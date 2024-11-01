import itertools
from collections.abc import Callable
from typing import Any, Literal, cast

import delu
import rtdl_num_embeddings
import rtdl_revisiting_models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from .util import is_oom_exception


# ======================================================================================
# Initialization
# ======================================================================================
def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    return x.bernoulli_(0.5).mul_(2).add_(-1)


# ======================================================================================
# Modules
# ======================================================================================
class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class NLinear(nn.Module):
    """A stack of N linear layers. Each layer is applied to its own part of the input.

    **Shape**

    - Input: ``(B, N, in_features)``
    - Output: ``(B, N, out_features)``

    The i-th linear layer is applied to the i-th matrix of the shape (B, in_features).

    Technically, this is a simplified version of delu.nn.NLinear:
    https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html.
    The difference is that this layer supports only 3D inputs
    with exactly one batch dimension. By contrast, delu.nn.NLinear supports
    any number of batch dimensions.
    """

    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class _PiecewiseLinearEncodingV2(nn.Module):
    """A faster implementation of the piecewise-linear encoding.

    The original paper:
    "On Embeddings for Numerical Features in Tabular Deep Learning".

    The key observations that led to this implementation:

    * The piecewise-linear encoding is just
      a linear transformation followed by a clamp-based activation.
    * Aligning the *last* encoding channel across all features
      allows applying the activation simultaneously to all features without
      the loop over features.
    """

    weight: Tensor
    bias: Tensor
    single_bin_mask: None | Tensor
    mask: None | Tensor

    def __init__(self, bins: list[Tensor]) -> None:
        assert len(bins) > 0
        super().__init__()

        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)

        # Making the weight and bias trainable can lead to worse performance.
        self.register_buffer('weight', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias', torch.zeros(n_features, max_n_bins))

        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer(
            'single_bin_mask', single_bin_mask if single_bin_mask.any() else None
        )

        self.register_buffer(
            'mask',
            # The mask is needed if features have different number of bins.
            None
            if all(len(x) == len(bins[0]) for x in bins)
            else torch.row_stack(
                [
                    torch.cat(
                        [
                            # The number of bins for this feature, minus 1:
                            torch.ones((len(x) - 1) - 1, dtype=torch.bool),
                            # Inactive bins (always zeros):
                            torch.zeros(max_n_bins - (len(x) - 1), dtype=torch.bool),
                            # The last bin:
                            torch.ones(1, dtype=torch.bool),
                        ]
                    )
                    for x in bins
                ]
            ),
        )

        for i, bin_edges in enumerate(bins):
            # >>>
            # Recall that the piecewise-linear encoding of a single feature x
            # looks as follows:
            #
            # x --> [1, ..., 1, (x - left_bin_edge) / bin_width, 0, ..., 0]
            #
            # where the expression in the middle corresponds to the bin where
            # the feature belongs: left_bin_edge <= x < right_bin_edge
            # <<<
            bin_width = bin_edges.diff()
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
            # The last encoding channel is located in the same last column
            # for all features. Thus, when a feature hits its last bin,
            # the encoding look as follows:
            # x --> [1, ..., 1, 0, ..., 0, (x - left_bin_edge) / bin_width]
            self.weight[i, : len(bin_edges) - 2] = w[:-1]
            self.weight[i, -1] = w[-1]
            self.bias[i, : len(bin_edges) - 2] = b[:-1]
            self.bias[i, -1] = b[-1]

    def _encode1d(self, x: Tensor) -> Tensor:
        x = torch.addcmul(self.bias, self.weight, x[..., None])
        if x.shape[-1] > 1:
            x = torch.cat(
                [
                    x[..., :1].clamp_max(1.0),
                    x[..., 1:-1].clamp(0.0, 1.0),
                    (
                        x[..., -1:].clamp_min(0.0)
                        if self.single_bin_mask is None
                        else torch.where(
                            # For features with only one bin,
                            # the whole "piecewise-linear" encoding effectively behaves
                            # like MinMax scaling
                            # (assuming that the edges of the single bin
                            #  are the minimum and maximum feature values).
                            self.single_bin_mask[..., None],
                            x[..., -1:],
                            x[..., -1:].clamp_min(0.0),
                        )
                    ),
                ],
                dim=-1,
            )
        return x


class PiecewiseLinearEncoding0dV2(_PiecewiseLinearEncodingV2):
    """The flat version of the piecewise-linear encoding.

    This class is provided only for completeness.
    It is recommended to use the piecewise-linear *embeddings* (not the raw *encoding*).
    """

    def forward(self, x: Tensor) -> Tensor:
        x = self._encode1d(x)
        return x.flatten(-2) if self.mask is None else x[:, self.mask]


class PiecewiseLinearEncoding1dV2(_PiecewiseLinearEncodingV2):
    """The non-flat version of the piecewise-linear encoding.

    This class is provided only for completeness.
    It is recommended to use the piecewise-linear *embeddings* (not the raw *encoding*).
    """

    def forward(self, x: Tensor) -> Tensor:
        return self._encode1d(x)


class PiecewiseLinearEmbeddingsV2(nn.Module):
    """An improved version of the piecewise-linear embeddings.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, d_embedding)``

    Compared to the original piecewise-linear embeddings from the paper
    "On Embeddings for Numerical Features in Tabular Deep Learning":

    * This implementation uses different parametrization and initialization,
      which seems to improve performance on some datasets.
    * This implementation is *significantly* faster.

    The original implementation:
    https://github.com/yandex-research/rtdl-num-embeddings/blob/24173f7023088d48a1a81765029aedd632316d56/package/rtdl_num_embeddings.py#L550
    """

    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        assert d_embedding > 0

        super().__init__()
        self.linear1 = rtdl_num_embeddings.LinearEmbeddings(len(bins), d_embedding)
        self.encoding = PiecewiseLinearEncoding1dV2(bins)
        self.linear2 = NLinear(
            len(bins), self.encoding.weight.shape[-1], d_embedding, bias=False
        )
        # The piecewise-linear term is initialized with zeros.
        # Thus, at initialization, the whole embedding is actually linear,
        # and the piecewise-linear term is gradually learned during training.
        nn.init.zeros_(self.linear2.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear1(x) + self.linear2(self.encoding(x))


class OneHotEncoding0d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # NOTE
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                F.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


class ScaleEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        init: Literal['ones', 'normal', 'random-signs'],
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d))
        self._weight_init = init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._weight_init == 'ones':
            nn.init.ones_(self.weight)
        elif self._weight_init == 'normal':
            nn.init.normal_(self.weight)
        elif self._weight_init == 'random-signs':
            init_random_signs_(self.weight)
        else:
            raise ValueError(f'Unknown weight_init: {self._weight_init}')

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 2
        return x * self.weight


class LinearEfficientEnsemble(nn.Module):
    """
    This layer is a more configurable version of the "BatchEnsemble" layer
    from the paper
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    (link: https://arxiv.org/abs/2002.06715).

    First, this layer allows to select only some of the "ensembled" parts:
    - the input scaling  (r_i in the BatchEnsemble paper)
    - the output scaling (s_i in the BatchEnsemble paper)
    - the output bias    (not mentioned in the BatchEnsemble paper,
                          but is presented in public implementations)

    Second, the initialization of the scaling weights is configurable
    through the `scaling_init` argument.

    NOTE
    The term "adapter" is used in the TabM paper only to tell the story.
    The original BatchEnsemble paper does NOT use this term. So this class also
    avoids the term "adapter".
    """

    r: None | Tensor
    s: None | Tensor
    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x


def make_efficient_ensemble(module: nn.Module, **kwargs) -> None:
    """Replace torch.nn.Linear modules with LinearEfficientEnsemble.

    NOTE
    In the paper, there are no experiments with networks with normalization layers.
    Perhaps, their trainable weights (the affine transformations) also need
    "ensemblification" as in the paper about "FiLM-Ensemble".
    Additional experiments are required to make conclusions.
    """
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, nn.Linear):
            module.add_module(
                name,
                LinearEfficientEnsemble(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        else:
            make_efficient_ensemble(submodule, **kwargs)


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: None | int = None,
        d_out: None | int = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = 'ReLU',
    ) -> None:
        super().__init__()

        d_first = d_block if d_in is None else d_in
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_first if i == 0 else d_block, d_block),
                    getattr(nn, activation)(),
                    nn.Dropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


_CUSTOM_MODULES = {
    # https://docs.python.org/3/library/stdtypes.html#definition.__name__
    CustomModule.__name__: CustomModule
    for CustomModule in [
        rtdl_num_embeddings.LinearEmbeddings,
        rtdl_num_embeddings.LinearReLUEmbeddings,
        rtdl_num_embeddings.PeriodicEmbeddings,
        PiecewiseLinearEmbeddingsV2,
        MLP,
    ]
}


def make_module(type: str, *args, **kwargs) -> nn.Module:
    Module = getattr(nn, type, None)
    if Module is None:
        Module = _CUSTOM_MODULES[type]
    return Module(*args, **kwargs)


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


@torch.inference_mode()
def compute_parameter_stats(module: nn.Module) -> dict[str, dict[str, float]]:
    stats = {'norm': {}, 'gradnorm': {}, 'gradratio': {}}
    for name, parameter in module.named_parameters():
        stats['norm'][name] = parameter.norm().item()
        if parameter.grad is not None:
            stats['gradnorm'][name] = parameter.grad.norm().item()
            # Avoid computing statistics for zero-initialized parameters.
            if (parameter.abs() > 1e-6).any():
                stats['gradratio'][name] = (
                    (parameter.grad.abs() / parameter.abs().clamp_min_(1e-6))
                    .mean()
                    .item()
                )
    stats['norm']['model'] = (
        torch.cat([x.flatten() for x in module.parameters()]).norm().item()
    )
    stats['gradnorm']['model'] = (
        torch.cat([x.grad.flatten() for x in module.parameters() if x.grad is not None])
        .norm()
        .item()
    )
    return stats


# ======================================================================================
# Optimization
# ======================================================================================
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    from rtdl_num_embeddings import _Periodic

    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        nn.BatchNorm1d
        | nn.LayerNorm
        | nn.InstanceNorm1d
        | rtdl_revisiting_models.LinearEmbeddings
        | rtdl_num_embeddings.LinearEmbeddings
        | rtdl_num_embeddings.LinearReLUEmbeddings
        | _Periodic,
    )


def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_groups: None | list[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(
        itertools.chain.from_iterable(group['params'] for group in custom_groups)
    )
    assert len(custom_params) == sum(
        len(group['params']) for group in custom_groups
    ), 'Parameters in custom_groups must not intersect'
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {
        'params': [
            p
            for p in module.parameters()
            if p not in custom_params and p not in zero_wd_params
        ]
    }
    return [
        default_group,
        {'params': list(zero_wd_params), 'weight_decay': 0.0},
        *custom_groups,
    ]


def make_optimizer(type: str, **kwargs) -> torch.optim.Optimizer:
    Optimizer = getattr(torch.optim, type)
    return Optimizer(**kwargs)


# ======================================================================================
# Training
# ======================================================================================
def zero_grad_forward_backward(
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[Tensor], Tensor],  # step_fn: batch_idx -> loss
    batch_idx: Tensor,
    chunk_size: int,
    grad_scaler: None | torch.cuda.amp.GradScaler = None,  # type: ignore[code]
) -> tuple[Tensor, int]:
    """
    This is a standard training step. Additionally, it supports:

    - Training by chunks if the whole batch does not fit into GPU.
    - Gradient scaling for training in FP16.
    """
    backward = (
        Tensor.backward
        if grad_scaler is None
        else lambda x: grad_scaler.scale(x).backward()  # type: ignore[code]
    )
    batch_size = len(batch_idx)
    loss = None
    while chunk_size != 0:
        optimizer.zero_grad()

        try:
            if batch_size <= chunk_size:
                # The simple forward-backward.
                loss = step_fn(batch_idx)
                backward(loss)
            else:
                # Forward-backward by chunks.
                # Mathematically, this is equivalent to the simple forward-backward.
                # Technically, this implementations uses less memory.
                loss = None
                for chunk_idx in batch_idx.split(chunk_size):
                    chunk_loss = step_fn(chunk_idx)
                    chunk_loss = chunk_loss * (len(chunk_idx) / batch_size)
                    backward(chunk_loss)
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            delu.cuda.free_memory()
            chunk_size //= 2

        else:
            break

    if not chunk_size:
        raise RuntimeError('Not enough memory even for chunk_size=1')
    return cast(Tensor, loss), chunk_size