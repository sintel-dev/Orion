"""UniTS

This primitive is an pytorch implementation of "UNITS:
A Unified Multi-Task Time Series Model"
https://arxiv.org/abs/2403.00131

This is a modified version of the original code, which can be found
at https://github.com/mims-harvard/UniTS
"""
import io
import math
import warnings

import numpy as np
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from smart_open import open as smart_open
from timm.layers import DropPath, Mlp
from timm.layers.helpers import to_2tuple
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Signal(object):
    """Data object.

    Args:
        X (ndarray):
            An n-dimensional array of signal values
        seq_length(int):
            Size of input window
        pred_length(int):
            Size of prediction
    """

    def __init__(self, X, index, seq_length, pred_length):
        self.data = X
        self.index = index
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length]

        return x, y

    def first_index(self):
        """Return first index of each input sequence.
        """
        return self.index[self.seq_length: len(self.index) - self.pred_length + 1]


def calculate_unfold_output_length(input_length, size, step):
    # Calculate the number of windows
    num_windows = (input_length - size) // step + 1
    return num_windows


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            var_num=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if var_num is not None:
            self.template = nn.Parameter(
                torch.zeros(var_num, dim), requires_grad=True)
            torch.nn.init.normal_(self.template, std=.02)
        self.var_num = var_num

    def forward(self, x, query=None):
        B, N, C = x.shape
        if query is not None:
            q = self.q(query).reshape(
                B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            var_num = query.shape[1]
        else:
            q = self.q(self.template).reshape(1, self.var_num,
                                              self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            q = q.repeat(B, 1, 1, 1)
            var_num = self.var_num
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DynamicLinear(nn.Module):
    """
    A dynamic linear layer that can interpolate the weight size to support any given input and
    output feature dimension.
    """

    def __init__(self, in_features=None, out_features=None, fixed_in=0, bias=True):
        super(DynamicLinear, self).__init__()
        assert fixed_in < in_features, "fixed_in < in_features is required !!!"
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.fixed_in = fixed_in

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, out_features):
        """
        Forward pass for the dynamic linear layer.
        """
        fixed_weights = self.weights[:, :self.fixed_in]
        dynamic_weights = self.weights[:, self.fixed_in:]
        this_bias = self.bias
        in_features = x.shape[-1]

        if in_features != self.weights.size(1) or out_features != self.weights.size(0):
            dynamic_weights = F.interpolate(dynamic_weights.unsqueeze(0).unsqueeze(0), size=(
                out_features, in_features - self.fixed_in), mode='bilinear',
                align_corners=False).squeeze(0).squeeze(0)
            if self.fixed_in != 0:
                fixed_weights = F.interpolate(fixed_weights.unsqueeze(0).unsqueeze(0), size=(
                    out_features, self.fixed_in), mode='bilinear',
                    align_corners=False).squeeze(0).squeeze(0)
        if out_features != self.weights.size(0):
            this_bias = F.interpolate(this_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(
                1, out_features), mode='bilinear',
                align_corners=False).squeeze(0).squeeze(0).squeeze(0)
        return F.linear(x, torch.cat((fixed_weights, dynamic_weights), dim=1), this_bias)


class DynamicLinearMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            prefix_token_length=None,
            group=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Conv1d(in_features, hidden_features,
                             3, groups=group, bias=bias[0], padding=1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
        self.seq_fc = DynamicLinear(
            hidden_features // 4, hidden_features // 4, bias=bias[1], fixed_in=prefix_token_length)
        self.prompt_fc = DynamicLinear(
            hidden_features // 4, prefix_token_length, bias=bias[1], fixed_in=prefix_token_length)

        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.hidden_features = hidden_features
        self.prefix_token_length = prefix_token_length

    def dynamic_linear(self, x, prefix_seq_len):
        x_func = x[:, :, prefix_seq_len:]
        x_seq = x[:, :, :prefix_seq_len]
        x_seq_out = self.seq_fc(
            x_seq, x_seq.shape[-1] - self.prefix_token_length)
        x_prompt = self.prompt_fc(x_seq, self.prefix_token_length)
        x = torch.cat((x_prompt, x_seq_out, x_func), dim=-1)
        return x

    def split_dynamic_linear(self, x, prefix_seq_len):
        x1, x2 = x.chunk(2, dim=-2)
        x1 = self.dynamic_linear(x1, prefix_seq_len)
        return torch.cat((x1, x2), dim=-2)

    def forward(self, x, prefix_seq_len, dim=2):
        n, var, l, c = x.shape
        x = x.view(-1, l, c)
        x = x.transpose(-1, -2)
        x = self.fc1(x)
        x = self.split_dynamic_linear(x, prefix_seq_len)
        x = self.act(x)
        x = self.drop1(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.fc2(x).view(n, var, l, c)
        x = self.drop2(x)
        return x


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.zeros(
            1, 1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        return self.pe[:, :, offset:offset + x.size(2)]


class SeqAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,  # attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, P, C = x.shape

        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                  self.head_dim).permute(3, 0, 2, 4, 1, 5)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.mean(dim=1, keepdim=False)
        k = k.mean(dim=1, keepdim=False)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.view(B, self.num_heads, N, -1, P).permute(0,
                                                        2, 4, 1, 3).reshape(B, N, P, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GateLayer(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        gate_value = self.gate(x)
        return gate_value.sigmoid() * x


class SeqAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask):
        x_input = x
        x = self.norm1(x)
        n_vars, n_seqs = x.shape[1], x.shape[2]
        x = torch.reshape(
            x, (-1, x.shape[-2], x.shape[-1]))
        x = self.attn_seq(x, attn_mask)
        x = torch.reshape(
            x, (-1, n_vars, n_seqs, x.shape[-1]))
        x = x_input + self.drop_path1(self.ls1(x))
        return x


class VarAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn_var(self.norm1(x))))
        return x


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=None,
            prefix_token_length=0,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        if mlp_layer is DynamicLinearMlp:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
                prefix_token_length=prefix_token_length,
            )
        else:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )
        self.ls2 = GateLayer(dim, init_values=init_values)
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prefix_seq_len=None):
        if prefix_seq_len is not None:
            x = x + \
                self.drop_path2(
                    self.ls2(self.mlp(self.norm2(x), prefix_seq_len=prefix_seq_len)))
        else:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=8.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            prefix_token_length=0,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values,
                                         proj_drop=proj_drop, drop_path=drop_path,
                                         norm_layer=norm_layer)

        self.var_att_block = VarAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values,
                                         proj_drop=proj_drop, drop_path=drop_path,
                                         norm_layer=norm_layer)

        self.dynamic_mlp = MLPBlock(dim=dim, mlp_ratio=mlp_ratio, mlp_layer=DynamicLinearMlp,
                                    proj_drop=proj_drop, init_values=init_values,
                                    drop_path=drop_path,
                                    act_layer=act_layer, norm_layer=norm_layer,
                                    prefix_token_length=prefix_token_length)

    def forward(self, x, prefix_seq_len, attn_mask):
        x = self.seq_att_block(x, attn_mask)
        x = self.var_att_block(x)
        x = self.dynamic_mlp(x, prefix_seq_len=prefix_seq_len)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        assert self.patch_len == self.stride, "non-overlap"
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class CLSHead(nn.Module):
    def __init__(self, d_model, head_dropout=0):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.cross_att = CrossAttention(d_mid)

        self.mlp = MLPBlock(dim=d_mid, mlp_ratio=8, mlp_layer=Mlp,
                            proj_drop=head_dropout, init_values=None, drop_path=0.0,
                            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                            prefix_token_length=None)

    def forward(self, x, category_token=None, return_feature=False):
        x = self.proj_in(x)
        B, V, L, C = x.shape
        x = x.view(-1, L, C)
        cls_token = x[:, -1:]
        cls_token = self.cross_att(x, query=cls_token)
        cls_token = cls_token.reshape(B, V, -1, C)

        cls_token = self.mlp(cls_token)
        if return_feature:
            return cls_token
        m = category_token.shape[2]
        cls_token = cls_token.expand(B, V, m, C)
        distance = torch.einsum('nvkc,nvmc->nvm', cls_token, category_token)

        distance = distance.mean(dim=1)
        return distance


class ForecastHead(nn.Module):
    def __init__(self, d_model, patch_len, stride, pad, head_dropout=0, prefix_token_length=None):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=int(d_model * 4),
            act_layer=nn.GELU,
            drop=head_dropout,
        )
        self.proj_out = nn.Linear(d_model, patch_len)
        self.pad = pad
        self.patch_len = patch_len
        self.stride = stride
        self.pos_proj = DynamicLinear(
            in_features=128, out_features=128, fixed_in=prefix_token_length)

    def forward(self, x_full, pred_len, token_len):
        x_full = self.proj_in(x_full)
        x_pred = x_full[:, :, -token_len:]
        x = x_full.transpose(-1, -2)
        x = self.pos_proj(x, token_len)
        x = x.transpose(-1, -2)
        x = x + x_pred
        x = self.mlp(x)
        x = self.proj_out(x)

        bs, n_vars = x.shape[0], x.shape[1]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.fold(x, output_size=(
            pred_len, 1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1))
        x = x.squeeze(dim=-1)
        x = x.reshape(bs, n_vars, -1)
        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    """
    UniTS: Building a Unified Time Series Model
    """

    def __init__(self, args):
        super().__init__()

        # Tokens settings
        self.prompt_token = nn.Parameter(torch.zeros(1, 1, args['prompt_num'], args['d_model']))
        torch.nn.init.normal_(self.prompt_token, std=.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, args['d_model']))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, args['d_model']))
        torch.nn.init.normal_(self.cls_token, std=.02)

        remainder = args['seq_len'] % args['patch_len']
        if remainder == 0:
            padding = 0
        else:
            padding = args['patch_len'] - remainder
        input_token_len = calculate_unfold_output_length(
            args['seq_len'] + padding, args['stride'], args['patch_len'])
        input_pad = args['stride'] * \
            (input_token_len - 1) + args['patch_len'] - \
            args['seq_len']
        pred_token_len = calculate_unfold_output_length(
            args['pred_len'] - input_pad, args['stride'], args['patch_len'])
        real_len = args['seq_len'] + \
            args['pred_len']
        self.cls_nums = [pred_token_len, args['pred_len'], real_len]

        # model settings #
        self.prompt_num = args['prompt_num']
        self.stride = args['stride']
        self.pad = args['stride']
        self.patch_len = args['patch_len']

        # input processing
        self.patch_embeddings = PatchEmbedding(
            args['d_model'], args['patch_len'], args['stride'], args['stride'], args['dropout'])
        self.position_embedding = LearnablePositionalEmbedding(args['d_model'])
        self.prompt2forecat = DynamicLinear(128, 128, fixed_in=args['prompt_num'])

        # basic blocks
        self.block_num = args['e_layers']
        self.blocks = nn.ModuleList(
            [BasicBlock(dim=args['d_model'], num_heads=args['n_heads'], qkv_bias=False,
                        qk_norm=False,
                        mlp_ratio=8., proj_drop=args['dropout'], attn_drop=0., drop_path=0.,
                        init_values=None, prefix_token_length=args['prompt_num'])
             for _ in range(args['e_layers'])]
        )

        # output processing
        self.cls_head = CLSHead(args['d_model'], head_dropout=args['dropout'])
        self.forecast_head = ForecastHead(
            args['d_model'], args['patch_len'], args['stride'], args['stride'],
            prefix_token_length=args['prompt_num'], head_dropout=args['dropout'])

    def tokenize(self, x, mask=None):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x * x, dim=1) /
                               torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(dim=1)
        else:
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        x = x.permute(0, 2, 1)
        remainder = x.shape[2] % self.patch_len
        if remainder != 0:
            padding = self.patch_len - remainder
            x = F.pad(x, (0, padding))
        else:
            padding = 0
        x, n_vars = self.patch_embeddings(x)
        return x, means, stdev, n_vars, padding

    def prepare_prompt(self, x, n_vars, prefix_prompt, task_prompt, task_prompt_num):
        x = torch.reshape(
            x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        # append prompt tokens
        this_prompt = prefix_prompt.repeat(x.shape[0], 1, 1, 1)

        this_mask_prompt = task_prompt.repeat(
            x.shape[0], 1, task_prompt_num, 1)
        init_full_input = torch.cat(
            (this_prompt, x, this_mask_prompt), dim=-2)
        init_mask_prompt = self.prompt2forecat(init_full_input.transpose(
            -1, -2), init_full_input.shape[2] - prefix_prompt.shape[2]).transpose(-1, -2)
        this_function_prompt = init_mask_prompt[:, :, -task_prompt_num:]
        x = torch.cat((this_prompt, x, this_function_prompt), dim=2)
        x[:, :, self.prompt_num:] = x[:, :, self.prompt_num:] + \
            self.position_embedding(x[:, :, self.prompt_num:])

        return x

    def mark2token(self, x_mark):
        x_mark = x_mark.unfold(
            dimension=-1, size=self.patch_len, step=self.stride)
        x_mark = x_mark.mean(dim=-1)
        x_mark = (x_mark > 0).float()
        return x_mark

    def backbone(self, x, prefix_len, seq_len):
        attn_mask = None
        for block in self.blocks:
            x = block(x, prefix_seq_len=prefix_len +
                      seq_len, attn_mask=attn_mask)
        return x

    def forecast(self, x):
        task_prompt_num = self.cls_nums[0]
        task_seq_num = self.cls_nums[1]
        real_seq_len = self.cls_nums[2]
        x, means, stdev, n_vars, _ = self.tokenize(x)
        prefix_prompt = self.prompt_token.repeat(1, n_vars, 1, 1)
        task_prompt = self.mask_token.repeat(1, n_vars, 1, 1)

        x = self.prepare_prompt(
            x, n_vars, prefix_prompt, task_prompt, task_prompt_num)

        seq_token_len = x.shape[-2] - prefix_prompt.shape[2]
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        x = self.forecast_head(
            x, real_seq_len, seq_token_len)
        x = x[:, -task_seq_num:]

        # De-Normalization from Non-stationary Transformer
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        return x

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out  # [B, L, D]


class UniTS(object):
    """UniTS
    X: ndarray of input
    window_length: length of input window
    """

    def __init__(self,
                 window_size=250,
                 pred_length=1,
                 prompt_num=10,
                 d_model=64,
                 patch_len=1,
                 stride=1,
                 dropout=0.1,
                 e_layers=3,
                 n_heads=8,
                 pretrained_weight='units_x128_pretrain_checkpoint.pth'):
        super(UniTS, self).__init__()

        args = {'prompt_num': prompt_num, 'd_model': d_model, 'patch_len': patch_len,
                'stride': stride, 'dropout': dropout, 'e_layers': e_layers, 'n_heads': n_heads,
                'seq_len': window_size, 'pred_len': pred_length}
        self.args = args
        self.pretrained_weight = pretrained_weight

        self.model = self._build_model()

    def _build_model(self):
        model = Model(self.args)
        return model

    def predict(self, X, index):
        """Forecasting timeseries
        Args:
            X: ndarray of input
            index: index array of input
        Return:
            prediction, ground truth, index."""

        self.test_dataset = Signal(X, index, seq_length=self.args['seq_len'],
                                   pred_length=self.args['pred_len'])
        first_index = self.test_dataset.first_index()
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=False)

        # if os.path.exists(self.pretrained_weight):
        #     pretrain_weight_path = self.pretrained_weight
        #     print('loading pretrained model:', pretrain_weight_path)
        #     ckpt = torch.load(pretrain_weight_path, map_location='cpu')
        #     msg = self.model.load_state_dict(ckpt, strict=False)
        #     print(msg)
        #     total_params = sum(p.numel() for p in self.model.parameters())
        #     print(f'Total number of parameters: {total_params}')
        # else:
        #     print("no ckpt found!")
        #     print('loading pretrained model:', self.pretrained_weight)
        #     exit()

        load_path = "https://sintel-orion.s3.us-east-2.amazonaws.com/pretrained/units.pth"
        with smart_open(load_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            self.model.load_state_dict(torch.load(buffer), strict=False)

        pred_len = self.args['pred_len']

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for (batch_x, batch_y) in tqdm(self.test_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()

                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x)

                f_dim = 0
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:]

                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                del batch_x
                del batch_y

        preds = torch.stack(preds, dim=0)
        trues = torch.stack(trues, dim=0)
        preds = np.array(preds)
        trues = np.array(trues)

        torch.cuda.empty_cache()
        return preds[:, 0, 0], trues[:, 0, 0], first_index
