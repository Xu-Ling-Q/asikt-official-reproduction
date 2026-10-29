from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn


@dataclass
class Mamba2Config:
    d_model: int
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 10

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        if self.d_inner % self.headdim != 0:
            raise ValueError("Mamba2 d_inner must be divisible by headdim.")
        self.nheads = self.d_inner // self.headdim


class InferenceCache(NamedTuple):
    conv_state: Tensor
    ssm_state: Tensor


class Mamba2(nn.Module):
    """Minimal Mamba-2 block adapted from tommyip/mamba2-minimal."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        projection_size = (
            2 * config.d_inner + 2 * config.d_state + config.nheads
        )
        self.in_proj = nn.Linear(config.d_model, projection_size, bias=False)
        conv_size = config.d_inner + 2 * config.d_state
        self.conv1d = nn.Conv1d(
            conv_size,
            conv_size,
            kernel_size=config.d_conv,
            groups=conv_size,
            padding=config.d_conv - 1,
        )
        self.dt_bias = nn.Parameter(torch.zeros(config.nheads))
        self.A_log = nn.Parameter(torch.zeros(config.nheads))
        self.D = nn.Parameter(torch.ones(config.nheads))
        self.norm = RMSNorm(config.d_inner)
        self.out_proj = nn.Linear(
            config.d_inner, config.d_model, bias=False
        )

    def forward(
        self, inputs: Tensor, cache: Optional[InferenceCache] = None
    ):
        if cache is not None:
            return self.step(inputs, cache)
        if inputs.shape[1] % self.config.chunk_size != 0:
            raise ValueError(
                "ASIKT sequence length must be divisible by "
                f"Mamba2 chunk_size={self.config.chunk_size}."
            )

        decay = -torch.exp(self.A_log)
        projected = self.in_proj(inputs)
        z, xbc, dt = torch.split(
            projected,
            [
                self.config.d_inner,
                self.config.d_inner + 2 * self.config.d_state,
                self.config.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)

        conv_input = rearrange(xbc, "b l d -> b d l")
        conv_state = F.pad(
            conv_input, (self.config.d_conv - inputs.shape[1], 0)
        )
        xbc = silu(
            self.conv1d(conv_input).transpose(1, 2)[:, : inputs.shape[1]]
        )
        x, b_state, c_state = torch.split(
            xbc,
            [
                self.config.d_inner,
                self.config.d_state,
                self.config.d_state,
            ],
            dim=-1,
        )
        x = rearrange(
            x, "b l (h p) -> b l h p", p=self.config.headdim
        )
        output, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            decay * dt,
            rearrange(b_state, "b l n -> b l 1 n"),
            rearrange(c_state, "b l n -> b l 1 n"),
            self.config.chunk_size,
        )
        output = output + x * self.D.unsqueeze(-1)
        output = rearrange(output, "b l h p -> b l (h p)")
        output = self.out_proj(self.norm(output, z))
        return output, InferenceCache(conv_state, ssm_state)

    def step(self, inputs: Tensor, cache: InferenceCache):
        if inputs.shape[1] != 1:
            raise ValueError("Mamba2 cached inference accepts one token.")
        projected = self.in_proj(inputs.squeeze(1))
        z, xbc, dt = torch.split(
            projected,
            [
                self.config.d_inner,
                self.config.d_inner + 2 * self.config.d_state,
                self.config.nheads,
            ],
            dim=-1,
        )
        cache.conv_state.copy_(
            torch.roll(cache.conv_state, shifts=-1, dims=-1)
        )
        cache.conv_state[:, :, -1] = xbc
        xbc = torch.sum(
            cache.conv_state
            * rearrange(self.conv1d.weight, "d 1 w -> d w"),
            dim=-1,
        )
        xbc = silu(xbc + self.conv1d.bias)
        x, b_state, c_state = torch.split(
            xbc,
            [
                self.config.d_inner,
                self.config.d_state,
                self.config.d_state,
            ],
            dim=-1,
        )
        decay = -torch.exp(self.A_log)
        dt = F.softplus(dt + self.dt_bias)
        delta_decay = torch.exp(dt * decay)
        x = rearrange(x, "b (h p) -> b h p", p=self.config.headdim)
        delta_bx = torch.einsum("bh,bn,bhp->bhpn", dt, b_state, x)
        cache.ssm_state.copy_(
            cache.ssm_state
            * rearrange(delta_decay, "b h -> b h 1 1")
            + delta_bx
        )
        output = torch.einsum(
            "bhpn,bn->bhp", cache.ssm_state, c_state
        )
        output = output + rearrange(self.D, "h -> h 1") * x
        output = rearrange(output, "b h p -> b (h p)")
        output = self.out_proj(self.norm(output, z))
        return output.unsqueeze(1), cache


def segment_sum(values):
    length = values.size(-1)
    expanded = repeat(values, "... d -> ... d e", e=length)
    strict_lower = torch.tril(
        torch.ones(
            length, length, dtype=torch.bool, device=values.device
        ),
        diagonal=-1,
    )
    expanded = expanded.masked_fill(~strict_lower, 0)
    cumulative = torch.cumsum(expanded, dim=-2)
    lower = torch.tril(
        torch.ones(
            length, length, dtype=torch.bool, device=values.device
        )
    )
    return cumulative.masked_fill(~lower, -torch.inf)


def ssd(x, decay, b_state, c_state, chunk_size, initial_states=None):
    sequence_length = x.shape[1]
    x, decay, b_state, c_state = [
        rearrange(value, "b (c l) ... -> b c l ...", l=chunk_size)
        for value in (x, decay, b_state, c_state)
    ]
    decay = rearrange(decay, "b c l h -> b h c l")
    decay_cumsum = torch.cumsum(decay, dim=-1)

    lower_decay = torch.exp(segment_sum(decay))
    diagonal = torch.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp",
        c_state,
        b_state,
        lower_decay,
        x,
    )
    state_decay = torch.exp(
        decay_cumsum[:, :, :, -1:] - decay_cumsum
    )
    states = torch.einsum(
        "bclhn,bhcl,bclhp->bchpn", b_state, state_decay, x
    )
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    chunk_decay = torch.exp(
        segment_sum(F.pad(decay_cumsum[:, :, :, -1], (1, 0)))
    )
    new_states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    output_decay = torch.exp(decay_cumsum)
    off_diagonal = torch.einsum(
        "bclhn,bchpn,bhcl->bclhp",
        c_state,
        states,
        output_decay,
    )
    output = rearrange(
        diagonal + off_diagonal, "b c l h p -> b (c l) h p"
    )
    if output.shape[1] != sequence_length:
        raise RuntimeError("Mamba2 changed the ASIKT sequence length.")
    return output, final_state


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, inputs, gate=None):
        if gate is not None:
            inputs = inputs * silu(gate)
        scale = torch.rsqrt(
            inputs.pow(2).mean(-1, keepdim=True) + self.eps
        )
        return inputs * scale * self.weight


def silu(inputs):
    return inputs * torch.sigmoid(inputs)
