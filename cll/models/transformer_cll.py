from dataclasses import dataclass, field
from typing import Optional

from cll.models.transformer_encoder_cll import CllTransformerEncoder
from cll.models.transformer_decoder_cll import CllTransformerDecoder
from fairseq.models import register_model
from fairseq.models.transformer import TransformerConfig, TransformerModelBase
from fairseq.models.transformer.transformer_config import (
    DecoderConfig,
    EncDecBaseConfig,
)
from fairseq.dataclass.utils import gen_parser_from_dataclass


@dataclass
class EncDecBaseConfigCll(EncDecBaseConfig):
    layers: int = field(default=6, metadata={"help": "number of layers"})
    attention_heads: int = field(
        default=8, metadata={"help": "number of attention heads"}
    )
    ffn_embed_dim: int = field(
        default=1024, metadata={"help": "embedding dimension for FFN"}
    )
    pass


@dataclass
class DecoderConfigCll(EncDecBaseConfigCll, DecoderConfig):
    pass


@dataclass
class TransformerConfigCll(TransformerConfig):
    language_num: int = field(default=0, metadata={"help": "the number of language in this task"})
    dropout: float = field(default=0.3, metadata={"help": "dropout probability"})
    encoder: EncDecBaseConfigCll = EncDecBaseConfigCll()
    decoder: DecoderConfigCll = DecoderConfigCll()


@register_model("transformer_cll", dataclass=TransformerConfigCll)
class TransformerModelCll(TransformerModelBase):

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        parser.add_argument(
            '--language_num', type=int, metavar='N',
            help='the number of language this task has',
        )
        parser.add_argument(
            '--dropout', type=float, metavar='N',
            help='dropout possibility',
        )
        parser.add_argument(
            '--layers', type=int, metavar='N',
            help='the number of layers',
        )
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return CllTransformerEncoder(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return CllTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs,
    ):
        # kwargs.get return tuple, but we need tensor, so add [0]
        src_direction = kwargs.get("src_direction", None),
        tgt_direction = kwargs.get("tgt_direction", None),
        src_direction = src_direction[0]
        tgt_direction = tgt_direction[0]
        if src_direction is None:
            raise ValueError(
                "building model has no src_direction!"
            )
        if tgt_direction is None:
            raise ValueError(
                "building model has no tgt_direction!"
            )
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            src_direction=src_direction,
            tgt_direction=tgt_direction,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        return decoder_out
