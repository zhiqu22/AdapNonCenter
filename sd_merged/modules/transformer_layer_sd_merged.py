from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayerBase,
    TransformerEncoderLayerBase,
)
from torch import Tensor
from fairseq.modules.fairseq_dropout import FairseqDropout
import itertools
import numpy as np


class SdMergedTransformerEncoderLayer(TransformerEncoderLayerBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.flag = cfg.flag
        self.normalize_before = True

    def forward(
            self,
            x,
            encoder_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor] = None,
            src_direction=None,
            tgt_direction=None,
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        # x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class SdMergedTransformerDecoderLayer(TransformerDecoderLayerBase):

    def __init__(self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.language_num = cfg.language_num
        self.flag = cfg.flag
        self.relu = F.relu
        self.dropout_specific = FairseqDropout(
            float(0.3), module_name=self.__class__.__name__
        )
        self.language_specific1 = self.language_specific_ffn1(512, 1024, self.quant_noise, self.quant_noise_block_size)
        self.language_specific2 = self.language_specific_ffn2(1024, 512, self.quant_noise, self.quant_noise_block_size)
        self.information_weights = self.build_weights()

    def language_specific_ffn1(self, input_dim, output_dim, q_noise, qn_block_size):
        tmp = nn.ModuleList()
        for i in range(self.language_num):
            tmp.append(quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size))
            nn.init.xavier_uniform_(tmp[i].weight, gain=1)
        return tmp

    def language_specific_ffn2(self, input_dim, output_dim, q_noise, qn_block_size):
        tmp = nn.ModuleList()
        for i in range(self.language_num):
            tmp.append(quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size))
            nn.init.xavier_uniform_(tmp[i].weight, gain=1)
        return tmp

    def build_weights(self):
        tmp = [0.1 for _ in range(self.language_num - 1)]
        return nn.Parameter(torch.FloatTensor(tmp))

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
            src_direction=None,
            tgt_direction=None,
    ):
        print(self.information_weights.data)
        # if self.language_specific1[0].bias.requires_grad:
        #     self.language_specific1[0].requires_grad_(False)
        #     self.language_specific1[1].requires_grad_(False)
        #     self.language_specific1[2].requires_grad_(False)
        #     self.language_specific1[3].requires_grad_(False)
        #
        #     self.language_specific2[0].requires_grad_(False)
        #     self.language_specific2[1].requires_grad_(False)
        #     self.language_specific2[2].requires_grad_(False)
        #     self.language_specific2[3].requires_grad_(False)
        if need_head_weights:
            need_attn = True
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            # import numpy as np
            # np.set_printoptions(suppress=True,precision=4)
            # print(attn.shape)
            # tmp = torch.clone(attn.view(1, -1)).to("cpu")
            # tmp = np.array(tmp)
            # with open('heat-map/specific_decoder_map.txt','ab') as f:
            #     np.savetxt(f, tmp, delimiter=" ")
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        torch.save(self.fc1.state_dict(), "nocenter-fc1.pt")
        torch.save(self.fc2.state_dict(), "nocenter-fc2.pt")
        torch.save(self.language_specific1.state_dict(), "nocenter-s1.pt")
        torch.save(self.language_specific2.state_dict(), "nocenter-s2.pt")
        # import numpy as np
        # np.set_printoptions(suppress=True,precision=4)
        # tmp = torch.clone(x.view(1, -1)).to("cpu")
        # tmp = np.array(tmp)
        # with open('heat-map/input3.txt','ab') as f:
        #     np.savetxt(f, tmp, delimiter=" ")
        shared = self.fc1(x)
        # block start
        # to compute additional information out of english
        # tgt_direction = torch.Tensor([3]).type(torch.int).cuda()
        specific = language_specific_instruction(x, self.language_num, tgt_direction, self.language_specific1)
        specific = self.relu(specific)
        # original activation dropout = 0, skip
        specific = language_specific_instruction(specific, self.language_num, tgt_direction, self.language_specific2,
                                                 times=2, weights=self.information_weights)
        specific = self.dropout_specific(specific)
        # block over
        x = self.activation_fn(shared)
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        # x = x + specific
        # tmp = torch.clone(x.view(1, -1)).to("cpu")
        # tmp = np.array(tmp)
        # with open('heat-map/output3.txt', 'ab') as f:
        #     np.savetxt(f, tmp, delimiter=" ")
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        tmp = torch.clone(x).view(1, -1).to("cpu")
        return x, attn, None


# out of english
def language_specific_instruction(x, language_num, direction, linear_language_specific, times=1, weights=None):
    if times != 1 and weights is None:
        raise ValueError("language_specific_instruction has error")
    # initialize idx list
    list_batch_language_direction_idx = [[] for i in range(language_num)]
    # fill idx
    for idx, val in enumerate(direction):
        list_batch_language_direction_idx[(val.item() - 1)].append(idx)
    # initialize interval for slicing tensor
    list_len_slice = [len(list_batch_language_direction_idx[i]) for i in range(language_num)]
    # new_order is arranged by sequence of language kinds
    # flatten to 1 dim
    new_order = list(itertools.chain(*list_batch_language_direction_idx))
    old_order = [new_order.index(i) for i in range(len(new_order))]
    # to Tensor
    new_order, old_order = torch.LongTensor(new_order).cuda(), torch.LongTensor(old_order).cuda()
    x = torch.index_select(x, 1, new_order)
    # container to save slice
    container_slice, pre_position = [], 0
    for i in range(language_num):
        # skip, if this batch does not have slice_i
        if list_len_slice[i] == 0:
            continue
        post_position = pre_position + list_len_slice[i]
        # x_i + (1-w_i) * x_shared_i
        tmp_tensor = linear_language_specific[i](x[:, pre_position:post_position, :])
        # if english skip
        if i == 0:
            tmp_tensor = torch.zeros_like(tmp_tensor)
        # if not english and times = 2
        if i != 0 and times != 1:
            tmp_tensor = weights[i - 1:i] * tmp_tensor
        pre_position = post_position
        container_slice.append(tmp_tensor)
    # return with concatenating and reordering to original order
    return torch.cat(container_slice, dim=1).index_select(1, old_order)
