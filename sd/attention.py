import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embd: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        #projects input into Q, K, V 
        self.in_proj = nn.Linear(d_embd, 3 * d_embd, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embd // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        #x : (BS, Seq_Len, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embd = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (BS, Seq_Len, Dim) -> (BS, Seq_Len, Dim * 3) -> 3 tensors of shape (BS, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (BS, Seq_Len, Dim) -> (BS, Seq_Len, n_heads, Dim / n_heads) -> (BS, n_heads, Seq_len, Dim / n_heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (BS, n_heads, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            #Mask where the upper triangle is made up of 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (BS, n_heads, Seq_Len, Seq_Len) @ (BS, n_heads, Seq_Len, Dim / H) -> (BS, n_heads, Seq_Len, Dim/H)
        output = weight @ v

        # (BS, n_heads, Seq_Len, Dim / n_heads) -> 
        output = output.transpose(1, 2)

        # (BS, Seq_Len, Dim)
        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (BS, Seq_Len, Dim)
        return output