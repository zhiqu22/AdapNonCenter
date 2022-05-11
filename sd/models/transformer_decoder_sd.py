from typing import Any, Dict, List, Optional
from torch import Tensor
from torch import nn
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerDecoderBase
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules import transformer_layer
from sd.modules.transformer_layer_sd import TransformerDecoderSd


class SdTransformerDecoder(TransformerDecoderBase):

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn,
        output_projection,)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn, idx=i)
                for i in range(6)
            ]
        )

    def build_decoder_layer(self, cfg, no_encoder_attn=False, idx=0):
        if idx == 3:
            layer = TransformerDecoderSd(cfg, no_encoder_attn)
        else:
            layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)

        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        # for unify coding format, the reason is, in the Inference step, data comes from 'encoder_out'
        src_direction = encoder_out["src_direction"]
        tgt_direction = encoder_out["tgt_direction"]
        if src_direction is None:
            raise ValueError(
                "LanguageSpecificBasedTransformerEncoderLayer has no src_direction!"
            )
        if tgt_direction is None:
            raise ValueError(
                "LanguageSpecificBasedTransformerEncoderLayer has no tgt_direction!"
            )
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_direction=src_direction,
            tgt_direction=tgt_direction,
        )
        if not features_only:
            x = self.output_layer(x)
            return x, extra

    def extract_features(
                self,
                prev_output_tokens,
                encoder_out: Optional[Dict[str, List[Tensor]]],
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                full_context_alignment: bool = False,
                alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None,
                src_direction=None,
                tgt_direction=None,
        ):
        return self.extract_features_scriptable(
                prev_output_tokens,
                encoder_out,
                incremental_state,
                full_context_alignment,
                alignment_layer,
                alignment_heads,
                src_direction,
                tgt_direction,
            )

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_direction=None,
            tgt_direction=None,
    ):
        """
            src_direction:Tensor, dim is 1, len is batch_size
            tgt_direction:Tensor, dim is 1, len is batch_size
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                    enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            if idx == 3:
                x, layer_attn, _ = layer(
                    x,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                    src_direction=src_direction,
                    tgt_direction=tgt_direction,
                )
            else:
                x, layer_attn, _ = layer(
                    x,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}
