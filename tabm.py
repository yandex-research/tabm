# License: https://github.com/yandex-research/tabm/blob/main/LICENSE
"""TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (ICLR 2025)."""  # noqa: E501

__version__ = '0.0.1'

import abc
import collections.abc
import typing
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypedDict, Union

import rtdl_num_embeddings
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from typing_extensions import Unpack

# ======================================================================================
# Utilities
# ======================================================================================
_INTERNAL_ERROR_MESSAGE = (
    'Internal error (this code must be unreachable; please, report a bug)'
)


def _get_required_kwarg(kwargs, key: str) -> Any:
    if key not in kwargs:
        raise TypeError(f'The required argument `{key}` is missing')
    return kwargs[key]


def _check_kwargs(
    kwargs,
    *,
    forbidden: Optional[list[str]] = None,
) -> None:
    assert isinstance(kwargs, dict), _INTERNAL_ERROR_MESSAGE
    if forbidden is not None:
        for argname in forbidden:
            if argname in kwargs:
                raise TypeError(f'The argument {argname} must not be passed explicitly')


def _check_positive_integer(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer, however: {name}={value}')


def _check_input_ndim(value: Tensor, name: str, required_ndim: int) -> None:
    if value.ndim != required_ndim:
        raise ValueError(
            f'The input must have {required_ndim} dimensions, however:'
            f' {name}.ndim={value.ndim}'
        )


def _check_input_min_ndim(value: Tensor, name: str, required_min_ndim: int) -> None:
    if value.ndim < required_min_ndim:
        raise ValueError(
            f'The input must have at least {required_min_ndim} dimensions,'
            f' however: {name}.ndim={value.ndim}'
        )


def _check_ensemble_input_shape(name: str, value: Tensor, k: int) -> None:
    _check_input_ndim(value, name, 3)
    if value.shape[-2] != k:
        raise ValueError(
            f'The penultimate input dimension must be equal to the ensemble size {k=},'
            f' however: {name}.shape[-2]={value.shape[-2]}'
        )


# ======================================================================================
# Initialization
# ======================================================================================
def _init_rsqrt_uniform_(tensor: Tensor, d: int) -> Tensor:
    assert d > 0, _INTERNAL_ERROR_MESSAGE
    d_rsqrt = d**-0.5
    return nn.init.uniform_(tensor, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def _init_random_signs_(tensor: Tensor) -> Tensor:
    return tensor.bernoulli_(0.5).mul_(2).add_(-1)


ScalingRandomInitialization = Literal['random-signs', 'normal']
ScalingInitialization = Literal[ScalingRandomInitialization, 'ones']


def init_scaling_(
    x: Tensor,
    distribution: ScalingInitialization,
    chunks: Optional[list[int]] = None,
) -> Tensor:
    """Initialize scaling parameters.

    Args:
        x: the tensor storing the scaling parameters.
        distribution: the initialization distribution.
        chunks: the initialization chunks.
    """
    if distribution == 'ones':
        if chunks is not None:
            raise ValueError(f'When {distribution=}, chunks must be None')
        init_fn = nn.init.ones_
    elif distribution == 'normal':
        init_fn = nn.init.normal_
    elif distribution == 'random-signs':
        init_fn = _init_random_signs_
    else:
        raise ValueError(f'Unknown {distribution=}')

    if chunks is None:
        return init_fn(x)

    else:
        if x.ndim < 1:
            raise ValueError(
                'When chunks is not None, the input tensor must have at least one'
                f'dimension, however: {x.ndim=}'
            )
        if sum(chunks) != x.shape[-1]:
            raise ValueError(
                'The tensor shape and chunks are incompatible:'
                f' {x.shape[-1]=} != {sum(chunks)=}'
            )

        # More generally, for a given set of leading dimensions,
        # all values within one chunk are initialized with the same (random) value.
        #
        # Consider an example:
        # - x.shape      == (4, 5)
        # - distribution == 'normal'
        # - chunks       == [2, 3]
        #
        # Then, each of the four rows of x is split in two chunks of sizes 2 and 3.
        # And each chunk is initialized with the same value sampled randomly from
        # the normal distribution. As a result, x is initialized as follows:
        #
        # [
        #     [a1 a1 b1 b1 b1]
        #     [a2 a2 b2 b2 b2]
        #     [a3 a3 b3 b3 b3]
        #     [a4 a4 b4 b4 b4]
        # ]
        #
        # where ai and bi are the randomly sampled values (i denotes the row index,
        # and a/b refers to the first/second chunk of a given row).
        with torch.inference_mode():
            chunk_start = 0
            for chunk_size in chunks:
                x[..., chunk_start : chunk_start + chunk_size] = init_fn(
                    torch.empty(*x.shape[:-1], 1)
                )
                chunk_start += chunk_size
        return x


# ======================================================================================
# Basics modules
# ======================================================================================
class _OneHotEncoding(nn.Module):
    """One-hot encoding for categorical features.

    The output of the module is the concatenation of one-hot representations of the
    categorical features.

    **Shape**

    - Input: ``(*, len(cardinalities))``,
          where `*` denotes an arbitrary number of batch dimensions.
    - Output: ``(*, sum(cardinalities))``.

    **Examples**

    >>> cardinalities = [2, 3]
    >>> m = _OneHotEncoding(cardinalities)
    >>> x = torch.tensor([
    ...     [0, 0],
    ...     [1, 2],
    ... ])
    >>> m(x)
    tensor([[1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1]])
    """

    def __init__(self, cardinalities: list[int]) -> None:
        """
        Args:
            cardinalities: the cardinalities of the categorical features,
                i.e. ``cardinalities[i]`` must store the number of unique categories
                of the i-th categorical feature.
        """
        if not cardinalities:
            raise ValueError(
                f'cardinalities must be non-empty, however: {cardinalities=}'
            )
        for i, cardinality in enumerate(cardinalities):
            if cardinality <= 0:
                raise ValueError(
                    'cardinalities must be a list of positive'
                    f' integers, however: cardinalities[{i}]={cardinalities[i]}'
                )

        super().__init__()
        self._cardinalities = cardinalities

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        return torch.Size((sum(self._cardinalities),))

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass.

        Args:
            x: the categorical features. The data type must be `torch.long`.
               The i-th feature must take values in ``range(0, cardinalities[i])``,
               where ``cardinalities`` is the list passed to the constructor.
        """
        _check_input_min_ndim(x, 'x', 1)
        if x.shape[-1] != len(self._cardinalities):
            raise ValueError(
                'Based on the cardinalities passed to the constructor, the input must'
                f' have the shape (*, {len(self._cardinalities)}), however: {x.shape=}'
            )

        return torch.cat(
            [
                nn.functional.one_hot(x[..., i], cardinality)
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


class ElementwiseAffine(nn.Module):
    """Elementwise affine transformation.

    **Shape**

    - Input: ``(*, *shape)``,
          where ``*`` denotes an arbitrary number of batch dimensions.
    - Output: ``(*, *shape)``

    **Examples**

    >>> m = ElementwiseAffine(
    ...     (3, 4),
    ...     scaling_init='normal',
    ...     scaling_init_chunks=None,
    ...     bias=True,
    ... )
    >>> x = torch.randn((1, 2, 3, 4))
    >>> m(x).shape
    torch.Size([1, 2, 3, 4])
    """

    bias: Optional[Tensor]

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        bias: bool,
        scaling_init: ScalingInitialization,
        scaling_init_chunks: Optional[list[int]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        """
        Args:
            shape: the input shape without the batch dimensions.
            scaling_init: the initialization of the scaling.
            scaling_init_chunks: the initialization chunks of the scaling
                (see README of the package for details).
            bias: if True, the module will have a trainable bias.
            dtype: the parameter data type.
            device: the parameter device.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty(shape, **factory_kwargs))  # type: ignore[call-overload]
        self.register_parameter(
            'bias',
            Parameter(torch.empty(shape, **factory_kwargs)) if bias else None,  # type: ignore[call-overload]
        )
        self._weight_init = scaling_init
        self._weight_init_chunks = scaling_init_chunks
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_scaling_(self.weight, self._weight_init, self._weight_init_chunks)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        expected_shape = self.weight.shape
        expected_shape_ndim = len(expected_shape)
        _check_input_min_ndim(x, 'x', expected_shape_ndim)
        if x.shape[-expected_shape_ndim:] != expected_shape:
            raise ValueError(
                'The input must have the shape'
                f' (*, {", ".join(map(str, expected_shape))}), however: {x.shape=}'
            )

        return (
            x * self.weight
            if self.bias is None
            else torch.addcmul(self.bias, self.weight, x)
        )


# ======================================================================================
# Ensemble modules
# ======================================================================================
def ensemble_view(x: Tensor, k: int, training: bool) -> Tensor:
    """The functional form of `EnsembleView`.

    Args:
        x: the tensor.
        k: the ensemble size.
        training: the training status.
    """
    if x.ndim == 2:
        x = x.unsqueeze(-2).expand(-1, k, -1)

    elif x.ndim == 3:
        if x.shape[-2] != k:
            raise ValueError(
                f'The penultimate input dimension must be equal to k={k},'
                f' however: {x.shape[-2]=}'
            )
        if not training:
            warnings.warn(
                'When training=False, the input should usually have the shape'
                ' (batch_size, d), i.e. hold one representation per object.'
                f' However: {x.shape=}. Is this intentional?'
            )

    else:
        raise ValueError(
            f'The must must have either two or three dimensions, however: {x.ndim=}'
        )

    return x


class EnsembleView(nn.Module):
    """Turn a tensor to a valid ensemble input.

    More precisely:

    * A two-dimensional tensors of the shape ``(batch_size, d)`` are expanded to
      three-dimensional tensors of the shape ``(batch_size, k, d)`` holding
      ``k`` identical views of the original input.
    * Three-dimensional tensors are propagated as-is without any changes.
      In this case, the second input dimension must be equal to the ensemble size ``k``.

    `EnsembleView` is a starting module for ensembles of MLP-like models (i.e. where the
    *base* model takes a single tensor of the shape ``(batch_tensor, d)`` as input).
    Usually, `EnsembleView` should be placed right before the first ensemble module.

    .. note::
        The module uses `torch.expand` under the hood and thus does not create any
        copies.

    **Shape**

    - Input: ``(batch_size, d)`` or ``(batch_size, k, d)``
    - Output: ``(batch_size, k, d)``

    **Examples**

    Two-dimensional tensors are expanded to three-dimensional tensors:

    >>> m = EnsembleView(k=2)
    >>> x = torch.randn(3, 4)
    >>> m(x).shape
    torch.Size([3, 2, 4])
    >>> torch.all(m(x) == x.unsqueeze(-2)).item()
    True

    Three-dimensional tensors are returned as-is:

    >>> m = EnsembleView(k=2)
    >>> x = torch.randn(3, 2, 4)
    >>> m(x) is x
    True
    """

    def __init__(self, *, k: int) -> None:
        """
        Args:
            k: the ensemble size.
        """
        super().__init__()
        self._k = k

    @property
    def k(self) -> int:
        """The ensemble size."""
        return self._k

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        return ensemble_view(x, self.k, self.training)


class LinearEnsemble(nn.Module):
    """An ensemble of k linear layers applied in parallel to k inputs.

    The i-th linear layer is applied to the i-th input matrix of the shape
    ``(batch_size, in_features)`` and produces the i-th output matrix of the shape
    ``(batch_size, out_features)``.

    **Shape**

    - Input: ``(batch_size, k, in_features)``
    - Output: ``(batch_size, k, out_features)``

    **Examples**

    >>> m = LinearEnsemble(2, 3, k=4)
    >>> x = torch.randn(5, 4, 2)
    >>> m(x).shape
    torch.Size([5, 4, 3])
    """

    bias: Optional[Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        """
        Args:
            in_features: the input size of each layer.
            out_features: the output size of each layer.
            bias: determines if the layers have biases.
            k: the number of linear layers.
            dtype: the parameter data type.
            device: the parameter device.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(
            torch.empty(k, in_features, out_features, **factory_kwargs)  # type: ignore[call-overload]
        )
        self.register_parameter(
            'bias',
            Parameter(torch.empty(k, out_features, **factory_kwargs)) if bias else None,  # type: ignore[call-overload]
        )
        self.reset_parameters()

    @classmethod
    def from_linear(cls, module: nn.Linear, **kwargs) -> 'LinearEnsemble':
        """Create an instance from its non-ensemble version.

        Args:
            module: the original module.
            kwargs: the ensemble-specific arguments.
        """
        kwargs.setdefault('dtype', module.weight.dtype)
        kwargs.setdefault('device', module.weight.device)
        return cls(
            module.in_features, module.out_features, module.bias is not None, **kwargs
        )

    @property
    def in_features(self) -> int:
        """The input dimension."""
        return self.weight.shape[-2]

    @property
    def out_features(self) -> int:
        """The output dimension."""
        return self.weight.shape[-1]

    @property
    def k(self) -> int:
        """The ensemble size."""
        return self.weight.shape[0]

    def reset_parameters(self):
        d = self.in_features
        _init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            _init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        _check_ensemble_input_shape('x', x, self.k)
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f'The last input dimension must be equal to {self.in_features},'
                f' however: {x.shape[-1]}'
            )

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class LinearBatchEnsemble(nn.Module):
    """A parameter-efficient ensemble of $k$ linear layers applied in parallel to $k$ inputs.

    The implementation follows the paper
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning".
    The only addition is the arguments related to the initialization of the $R$ and $S$ matrices,
    which is used by TabM.

    .. note::
        The notation related to the $R$ and $S$ matrices may be not fully in sync
        between the TabM paper and this code.

    .. note::
        In the TabM paper, the matrices $R$ and $S$, as well as the bias, are called
        "adapters". This term was introduced in the paper only to tell the story, and was *not*
        used in the original BatchEnsemble paper. So the implementation of this class
        also avoids the term "adapter".

    **Shape**

    - Input: ``(batch_size, k, in_features)``
    - Output: ``(batch_size, k, out_features)``

    **Examples**

    >>> m = LinearBatchEnsemble(4, 5, k=3, scaling_init='random-signs')
    >>> x = torch.randn(2, 3, 4)
    >>> m(x).shape
    torch.Size([2, 3, 5])
    """  # noqa: E501

    bias: Optional[Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        scaling_init: Union[
            ScalingInitialization, tuple[ScalingInitialization, ScalingInitialization]
        ],
        first_scaling_init_chunks: Optional[list[int]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.dtype]] = None,
    ):
        """
        Args:
            in_features: the input size of each layer.
            out_features: the output size of each layer.
            bias: determines if the layers have biases.
            k: the number of linear layers.
            scaling_init: the initialization distribution for the scaling parameters.
                A tuple can be passed to set different distributions for the first
                and second scaling parameters.
            first_scaling_init_chunks: the initialization chunks of the first scaling
                (see README of the package for details).
            dtype: the parameter data type.
            device: the parameter device.
        """
        _check_positive_integer(in_features, 'in_features')
        _check_positive_integer(out_features, 'out_features')
        _check_positive_integer(k, 'k')

        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)  # type: ignore[call-overload]
        )
        self.r = Parameter(  # R
            torch.empty(k, in_features, **factory_kwargs)  # type: ignore[call-overload]
        )
        self.s = Parameter(  # S
            torch.empty(k, out_features, **factory_kwargs)  # type: ignore[call-overload]
        )
        self.register_parameter(
            'bias',
            Parameter(torch.empty(k, out_features, **factory_kwargs)) if bias else None,  # type: ignore[call-overload]
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        if isinstance(scaling_init, tuple):
            self._first_scaling_init = scaling_init[0]
            self._second_scaling_init = scaling_init[1]
        else:
            self._first_scaling_init = scaling_init
            self._second_scaling_init = scaling_init
        self._first_scaling_init_chunks = first_scaling_init_chunks

        self.reset_parameters()

    @classmethod
    def from_linear(cls, module: nn.Linear, **kwargs) -> 'LinearBatchEnsemble':
        """Create an instance from its non-ensemble version.

        Args:
            module: the original module.
            kwargs: the ensemble-specific arguments.
        """
        kwargs.setdefault('dtype', module.weight.dtype)
        kwargs.setdefault('device', module.weight.device)
        return cls(
            module.in_features, module.out_features, module.bias is not None, **kwargs
        )

    def reset_parameters(self) -> None:
        _init_rsqrt_uniform_(self.weight, self.in_features)
        init_scaling_(self.r, self._first_scaling_init, self._first_scaling_init_chunks)
        init_scaling_(self.s, self._second_scaling_init, None)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE
                # All k biases have the same initialization (this is why the shape of
                # bias_init is (out_features,) instead of (k, out_features)).
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = _init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_ndim(x, 'x', 3)  # (B, K, D)

        # >>> Equation (5) from the BatchEnsemble paper (arXiv v2).
        x = x * self.r
        x = x @ self.weight.T
        x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x


class BatchNorm1dEnsemble(nn.Module):
    """An ensemble of $k$ batch normalizations applied in parallel to $k$ inputs.

    The module can be used in ensembles of MLP-like models.
    Similarly to other ensemble modules in this package, the $k$ normalizations are
    fully independent, i.e. they do not share the running statistics nor the
    parameters of the affine transformations.

    **Shape**

    - Input: ``(batch_size, k, num_features)``
    - Output: ``(batch_size, k, num_features)``

    **Examples**

    >>> m = BatchNorm1dEnsemble(2, k=3)
    >>> x = torch.randn(4, 3, 2)
    >>> m(x).shape
    torch.Size([4, 3, 2])
    """

    def __init__(
        self, num_features: int, *, k: int, affine: bool = True, **kwargs
    ) -> None:
        """
        Args:
            num_features: same as in `BatchNorm1d`.
            affine: same as in `BatchNorm1d`.
            k: the ensemble size.
            kwargs: the (optional) keyword arguments for `BatchNorm1d`.
        """
        _check_positive_integer(k, 'k')

        super().__init__()
        self.impl = nn.BatchNorm1d(k * num_features, affine=False, **kwargs)
        self.affine = (
            ElementwiseAffine(
                (k, num_features),
                scaling_init='ones',
                scaling_init_chunks=None,
                bias=True,
                dtype=kwargs.get('dtype'),
                device=kwargs.get('device'),
            )
            if affine
            else None
        )
        self._k = k

    @classmethod
    def from_batchnorm1d(
        cls, module: nn.BatchNorm1d, **kwargs
    ) -> 'BatchNorm1dEnsemble':
        """Create an instance from its non-ensemble version.

        Args:
            module: the original module.
            kwargs: the ensemble-specific arguments.
        """
        kwargs.update(
            (key, getattr(module, key)) for key in nn.BatchNorm1d.__constants__
        )
        if module.weight is not None:
            kwargs['dtype'] = module.weight.dtype
            kwargs['device'] = module.weight.device
        elif module.running_mean is not None:
            kwargs['dtype'] = module.running_mean.dtype
            kwargs['device'] = module.running_mean.device
        return BatchNorm1dEnsemble(**kwargs)

    @property
    def k(self) -> int:
        """The ensemble size."""
        return self._k

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_ensemble_input_shape('x', x, self.k)

        x = self.impl(x.flatten(-2)).unflatten(-1, x.shape[-2:])
        if self.affine is not None:
            x = self.affine(x)
        return x


class LayerNormEnsemble(nn.Module):
    """An ensemble of $k$ layer normalizations applied in parallel to $k$ inputs.

    * Contrary to other ensemble modules in this package, this modules does not impose
      any restriction on the input shape of the base normalization.
    * Similarly to other ensemble modules in this package, the $k$ normalizations are
      fully independent, i.e. they do not share the running statistics nor the
      weights of the affine transformations.

    **Shape**

    - Input: ``(*, k, *normalized_shape)``, where ``*`` denotes an arbitrary number of
          batch dimensions
    - Output: ``(*, k, *normalized_shape)``

    **Examples**

    >>> m = LayerNormEnsemble(2, k=3)
    >>> x = torch.randn(4, 3, 2)
    >>> m(x).shape
    torch.Size([4, 3, 2])

    >>> m = LayerNormEnsemble((2, 3), k=4)
    >>> x = torch.randn(5, 6, 4, 2, 3)
    >>> m(x).shape
    torch.Size([5, 6, 4, 2, 3])
    """

    def __init__(
        self,
        normalized_shape: Union[int, list[int], torch.Size],
        *,
        k: int,
        elementwise_affine: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            normalized_shape: same as in `LayerNorm`.
            k: the ensemble size.
            elementwise_affine: same as in `LayerNorm`.
            kwargs: the (optional) keyword arguments for `LayerNorm`.
        """
        _check_positive_integer(k, 'k')

        super().__init__()
        self.impl = nn.LayerNorm(normalized_shape, elementwise_affine=False, **kwargs)
        self.affine = (
            ElementwiseAffine(
                (k, *self.impl.normalized_shape),
                scaling_init='ones',
                scaling_init_chunks=None,
                bias=True,
                dtype=kwargs.get('dtype'),
                device=kwargs.get('device'),
            )
            if elementwise_affine
            else None
        )
        self._k = k

    @classmethod
    def from_layernorm(cls, module: nn.LayerNorm, **kwargs) -> 'LayerNormEnsemble':
        """Create an instance from its non-ensemble version.

        Args:
            module: the original module.
            kwargs: the ensemble-specific arguments.
        """
        kwargs.update((key, getattr(module, key)) for key in nn.LayerNorm.__constants__)
        if module.weight is not None:
            kwargs['dtype'] = module.weight.dtype
            kwargs['device'] = module.weight.device
        elif module.running_mean is not None:
            kwargs['dtype'] = module.running_mean.dtype
            kwargs['device'] = module.running_mean.device
        return LayerNormEnsemble(**kwargs)

    @property
    def k(self) -> int:
        """The ensemble size."""
        return self._k

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        expected_shape = (self.k, *self.impl.normalized_shape)
        expected_shape_ndim = len(expected_shape)
        _check_input_min_ndim(x, 'x', expected_shape_ndim)
        if x.shape[-expected_shape_ndim:] != expected_shape:
            raise ValueError(
                'The input must have the shape'
                f' (*, {", ".join(map(str, expected_shape))}), however: {x.shape=}'
            )

        x = self.impl(x)
        if self.affine is not None:
            x = self.affine(x)
        return x


# ======================================================================================
# Ensemble functions
# ======================================================================================
def _check_include_exclude(
    module: nn.Module,
    *,
    Layer: type[nn.Module],
    include: Optional[list[nn.Module]],
    exclude: Optional[list[nn.Module]],
    **kwargs,
) -> None:
    del kwargs

    if include is not None and exclude is not None:
        raise ValueError('include and exclude cannot be provided simultaneously')
    if include is not None:
        if any(not isinstance(x, Layer) for x in include):
            raise ValueError(f'All modules in `include` must be instances of {Layer=}')
    if exclude is not None:
        if any(not isinstance(x, Layer) for x in exclude):
            raise ValueError(f'All modules in `exclude` must be instances of {Layer=}')

    for list_name, list_values in [('include', include), ('exclude', exclude)]:
        if list_values is not None:
            submodules = list(module.modules())
            if any(x not in submodules for x in list_values):
                raise ValueError(
                    f'Some of the modules in `{list_name}` are not submodules'
                    ' of the provided module'
                )


def _replace_layers_impl_(
    module: nn.Module,
    *,
    Layer: type[nn.Module],
    factory: collections.abc.Callable[..., nn.Module],
    include: Optional[list[nn.Module]],
    exclude: Optional[list[nn.Module]],
    **kwargs,
) -> None:
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, Layer):
            if (
                (exclude is None or all(submodule is not x for x in exclude))
                if include is None
                else any(submodule is x for x in include)
            ):
                module.add_module(name, factory(submodule, **kwargs))
        else:
            _replace_layers_impl_(
                module=submodule,
                Layer=Layer,
                factory=factory,
                **kwargs,
                include=include,
                exclude=exclude,
            )


def _replace_layers_(**kwargs) -> None:
    _check_include_exclude(**kwargs)
    return _replace_layers_impl_(**kwargs)


def ensemble_linear_layers_(
    module: nn.Module,
    *,
    include: Optional[list[nn.Linear]] = None,
    exclude: Optional[list[nn.Linear]] = None,
    **kwargs,
) -> None:
    """Replace submodules of the type `torch.nn.Linear` with `LinearEnsemble` in a given module.

    Args:
        module: the module.
        include: replace only these submodules.
        exclude: do not replace these submodules.
        kwargs: the arguments for `LinearEnsemble`.

    **Examples**

    >>> def make_model():
    ...     return nn.Sequential(
    ...         nn.Linear(2, 3),
    ...         nn.ReLU(),
    ...         nn.Linear(3, 4),
    ...     )
    ...
    >>> model = make_model()
    >>> ensemble_linear_layers_(model, k=5)
    >>> isinstance(model[0], LinearEnsemble)
    True
    >>> isinstance(model[2], LinearEnsemble)
    True
    >>>
    >>> model = make_model()
    >>> ensemble_linear_layers_(model, include=[model[0]], k=5)
    >>> isinstance(model[0], LinearEnsemble)
    True
    >>> isinstance(model[2], LinearEnsemble)
    False
    >>>
    >>> model = make_model()
    >>> ensemble_linear_layers_(model, exclude=[model[0]], k=5)
    >>> isinstance(model[0], LinearEnsemble)
    False
    >>> isinstance(model[2], LinearEnsemble)
    True
    """  # noqa: E501
    _replace_layers_(
        module=module,
        Layer=nn.Linear,
        factory=LinearEnsemble.from_linear,
        include=include,
        exclude=exclude,
        **kwargs,
    )


def batchensemble_linear_layers_(
    module: nn.Module,
    *,
    include: Optional[list[nn.Linear]] = None,
    exclude: Optional[list[nn.Linear]] = None,
    **kwargs,
) -> None:
    """Replace submodules of the type `torch.nn.Linear` with `LinearBatchEnsemble` in a given module.

    Args:
        module: the module.
        include: replace only these submodules.
        exclude: do not replace these submodules.
        kwargs: the arguments for `LinearBatchEnsemble`.

    **Examples**

    >>> def make_model():
    ...     return nn.Sequential(
    ...         nn.Linear(2, 3),
    ...         nn.ReLU(),
    ...         nn.Linear(3, 4),
    ...     )
    ...
    >>> model = make_model()
    >>> batchensemble_linear_layers_(model, k=5, scaling_init='normal')
    >>> isinstance(model[0], LinearBatchEnsemble)
    True
    >>> isinstance(model[2], LinearBatchEnsemble)
    True
    >>>
    >>> model = make_model()
    >>> batchensemble_linear_layers_(model, include=[model[0]], k=5, scaling_init='normal')
    >>> isinstance(model[0], LinearBatchEnsemble)
    True
    >>> isinstance(model[2], LinearBatchEnsemble)
    False
    >>>
    >>> model = make_model()
    >>> batchensemble_linear_layers_(model, exclude=[model[0]], k=5, scaling_init='normal')
    >>> isinstance(model[0], LinearBatchEnsemble)
    False
    >>> isinstance(model[2], LinearBatchEnsemble)
    True
    """  # noqa: E501
    _replace_layers_(
        module=module,
        Layer=nn.Linear,
        factory=LinearBatchEnsemble.from_linear,
        include=include,
        exclude=exclude,
        **kwargs,
    )


def ensemble_batchnorm1d_layers_(
    module: nn.Module,
    *,
    include: Optional[list[nn.BatchNorm1d]] = None,
    exclude: Optional[list[nn.BatchNorm1d]] = None,
    **kwargs,
) -> None:
    """Replace submodules of the type `torch.nn.BatchNorm1d` with `BatchNorm1dEnsemble` in a given module.

    Args:
        module: the module.
        include: replace only these submodules.
        exclude: do not replace these submodules.
        kwargs: the arguments for `BatchNorm1dEnsemble`.

    **Examples**

    >>> def make_model():
    ...     return nn.Sequential(
    ...         nn.Linear(2, 3),
    ...         nn.BatchNorm1d(3),
    ...         nn.ReLU(),
    ...         nn.Linear(2, 3),
    ...         nn.BatchNorm1d(3),
    ...         nn.ReLU(),
    ...     )
    ...
    >>> model = make_model()
    >>> ensemble_batchnorm1d_layers_(model, k=5)
    >>> isinstance(model[1], BatchNorm1dEnsemble)
    True
    >>> isinstance(model[4], BatchNorm1dEnsemble)
    True
    >>>
    >>> model = make_model()
    >>> ensemble_batchnorm1d_layers_(model, include=[model[1]], k=5)
    >>> isinstance(model[1], BatchNorm1dEnsemble)
    True
    >>> isinstance(model[4], BatchNorm1dEnsemble)
    False
    >>>
    >>> model = make_model()
    >>> ensemble_batchnorm1d_layers_(model, exclude=[model[1]], k=5)
    >>> isinstance(model[1], BatchNorm1dEnsemble)
    False
    >>> isinstance(model[4], BatchNorm1dEnsemble)
    True
    """  # noqa: E501
    _replace_layers_(
        module=module,
        Layer=nn.BatchNorm1d,
        factory=BatchNorm1dEnsemble.from_batchnorm1d,
        include=include,
        exclude=exclude,
        **kwargs,
    )


def ensemble_layernorm_layers_(
    module: nn.Module,
    *,
    include: Optional[list[nn.LayerNorm]] = None,
    exclude: Optional[list[nn.LayerNorm]] = None,
    **kwargs,
) -> None:
    """Replace submodules of the type `torch.nn.LayerNorm` with `LayerNormEnsemble` in a given module.

    Args:
        module: the module.
        include: replace only these submodules.
        exclude: do not replace these submodules.
        kwargs: the arguments for `LayerNormEnsemble`.

    **Examples**

    >>> def make_model():
    ...     return nn.Sequential(
    ...         nn.Linear(2, 3),
    ...         nn.LayerNorm(3),
    ...         nn.ReLU(),
    ...         nn.Linear(2, 3),
    ...         nn.LayerNorm(3),
    ...         nn.ReLU(),
    ...     )
    ...
    >>> model = make_model()
    >>> ensemble_layernorm_layers_(model, k=5)
    >>> isinstance(model[1], LayerNormEnsemble)
    True
    >>> isinstance(model[4], LayerNormEnsemble)
    True
    >>>
    >>> model = make_model()
    >>> ensemble_layernorm_layers_(model, include=[model[1]], k=5)
    >>> isinstance(model[1], LayerNormEnsemble)
    True
    >>> isinstance(model[4], LayerNormEnsemble)
    False
    >>>
    >>> model = make_model()
    >>> ensemble_layernorm_layers_(model, exclude=[model[1]], k=5)
    >>> isinstance(model[1], LayerNormEnsemble)
    False
    >>> isinstance(model[4], LayerNormEnsemble)
    True
    """  # noqa: E501
    _replace_layers_(
        module=module,
        Layer=nn.LayerNorm,
        factory=LayerNormEnsemble.from_layernorm,
        include=include,
        exclude=exclude,
        **kwargs,
    )


# ======================================================================================
# MLP modules
# ======================================================================================
class MLPBackboneKwargs(TypedDict, total=False):
    """
    The arguments have the same meaning as in `make_tabm_backbone`.
    """

    d_in: int
    n_blocks: int
    d_block: int
    dropout: float
    activation: str


class MLPBackboneEnsembleKwargs(MLPBackboneKwargs):
    """
    The arguments have the same meaning as in `make_tabm_backbone`.
    """

    k: int


class _MLPBackboneBase(nn.Module, abc.ABC):
    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = 'ReLU',
        # NOTE
        # The following argument can be used by children classes for their needs,
        # but it must not be a part of the public API.
        options: Optional[Any] = None,
    ) -> None:
        _check_positive_integer(d_in, 'd_in')
        _check_positive_integer(n_blocks, 'n_blocks')
        _check_positive_integer(d_block, 'd_block')

        try:
            Activation = getattr(nn, activation)
        except AttributeError:
            raise ValueError(
                f'The activation "{activation}" does not exist in the torch.nn package'
            )

        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    self._make_linear(i, d_in if i == 0 else d_block, d_block, options),
                    Activation(),
                    nn.Dropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )

    @abc.abstractmethod
    def _make_linear(
        self,
        index: int,
        in_features: int,
        out_features: int,
        options: Any,
    ) -> nn.Module: ...


class MLPBackbone(_MLPBackboneBase):
    """MLP backbone."""

    def __init__(self, **kwargs: Unpack[MLPBackboneKwargs]) -> None:
        """
        Args:
            kwargs: see `MLPBackboneKwargs`.
        """
        super().__init__(**kwargs, options=None)

    def _make_linear(
        self, index: int, in_features: int, out_features: int, options: Any
    ) -> nn.Module:
        del index
        assert options is None, _INTERNAL_ERROR_MESSAGE
        return nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        for block in self.blocks:
            x = block(x)
        return x


class MLPBackboneEnsembleBase(_MLPBackboneBase):
    """The base class for MLP ensembles."""

    def __init__(
        self,
        *,
        k: int,
        options: Optional[Any] = None,
        **kwargs: Unpack[MLPBackboneKwargs],
    ) -> None:
        _check_positive_integer(k, 'k')

        super().__init__(**kwargs, options=options)
        self._k = k

    @property
    def k(self) -> int:
        """The ensemble size."""
        return self._k

    @abc.abstractmethod
    def get_original_output_shape(self) -> torch.Size:
        """Get the output shape of one underlying MLP, without the batch dimensions."""
        ...


class MLPBackboneEnsemble(MLPBackboneEnsembleBase):
    """An ensemble of $k$ MLP backbones applied in parallel to $k$ inputs.

    **Shape**

    - Input: ``(batch_size, k, d_in)``
    - Output: ``(batch_size, k, d_block)``
    """

    def __init__(self, **kwargs: Unpack[MLPBackboneEnsembleKwargs]) -> None:
        """
        Args:
            kwargs: see `MLPBackboneEnsembleKwargs`.
        """
        super().__init__(**kwargs, options=None)

    def get_original_output_shape(self) -> torch.Size:
        return torch.Size((self.blocks[-1][0].weight.shape[-1],))

    def _make_linear(
        self,
        index: int,
        in_features: int,
        out_features: int,
        options: Any,
    ) -> nn.Module:
        assert options is None, _INTERNAL_ERROR_MESSAGE
        del index
        return LinearEnsemble(in_features, out_features, k=self.k)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_ndim(x, 'x', 3)  # (B, K, D)
        for block in self.blocks:
            x = block(x)
        return x


class MLPBackboneMiniEnsemble(MLPBackboneEnsembleBase):
    """An ensemble of $k$ almost fully shared MLP backbones applied in parallel.

    All non-shared parameters are concentrated in one affine transformation
    applied at the very beginning of the forward pass. This can be informally viewed
    as a minimum possible architectural approach to ensembling, hence the name.
    """

    def __init__(
        self,
        *,
        affine_bias: bool,  # TabM uses False
        affine_scaling_init: ScalingRandomInitialization,
        affine_scaling_init_chunks: Optional[list[int]] = None,
        **kwargs: Unpack[MLPBackboneEnsembleKwargs],
    ) -> None:
        """
        Args:
            affine_scaling_init: the initialization of the scaling.
            affine_scaling_init_chunks: the initialization chunks of the scaling
                (see README of the package for details).
            affine_bias: if True, the affine transformation will have a trainable bias.
            kwargs: see `MLPBackboneEnsembleKwargs`.
        """
        super().__init__(**kwargs)
        self.affine = ElementwiseAffine(
            (self.k, self.blocks[0][0].weight.shape[-1]),
            scaling_init=affine_scaling_init,
            scaling_init_chunks=affine_scaling_init_chunks,
            bias=affine_bias,
        )

    def get_original_output_shape(self) -> torch.Size:
        return torch.Size((self.blocks[-1][0].weight.shape[0],))

    def _make_linear(
        self,
        index: int,
        in_features: int,
        out_features: int,
        options: Any,
    ) -> nn.Module:
        assert options is None, _INTERNAL_ERROR_MESSAGE
        del index
        # The same linear layer will be used by all k backbones.
        return nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_min_ndim(x, 'x', 2)
        x = self.affine(x)
        for block in self.blocks:
            x = block(x)
        return x


class MLPBackboneBatchEnsemble(MLPBackboneEnsembleBase):
    """A parameter-efficient ensemble of $k$ MLP backbones applied in parallel.

    This class uses the BatchEnsemble technique described in the paper "BatchEnsemble:
    An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    and allows one to use the TabM-style initialization.
    """

    @dataclass(frozen=True)
    class _Options:
        k: int
        tabm_init: bool
        scaling_init: ScalingRandomInitialization
        start_scaling_init_chunks: Optional[list[int]]

    def __init__(
        self,
        *,
        tabm_init: bool,
        scaling_init: ScalingRandomInitialization,
        start_scaling_init_chunks: Optional[list[int]],
        **kwargs: Unpack[MLPBackboneEnsembleKwargs],
    ):
        """
        Args:
            tabm_init: If True, the TabM-style initialization is used for the non-shared
                scaling parameters of the backbones
                (random initialization for the very first scaling and deterministic
                initialization with ones for all other scalings).
            scaling_init: the initialization of the non-shared scalings parameters.
            start_scaling_init_chunks: the initialization chunks of the very first
                scaling.
            kwargs: see `MLPBackboneEnsembleKwargs`.
        """
        super().__init__(
            **kwargs,
            options=MLPBackboneBatchEnsemble._Options(
                kwargs['k'], tabm_init, scaling_init, start_scaling_init_chunks
            ),
        )

    def get_original_output_shape(self) -> torch.Size:
        return torch.Size((self.blocks[-1][0].weight.shape[0],))

    def _make_linear(
        self,
        index: int,
        in_features: int,
        out_features: int,
        options: _Options,
    ) -> nn.Module:
        return LinearBatchEnsemble(
            in_features,
            out_features,
            k=options.k,
            scaling_init=(
                ((options.scaling_init, 'ones') if index == 0 else 'ones')
                if options.tabm_init
                else options.scaling_init
            ),
            first_scaling_init_chunks=(
                options.start_scaling_init_chunks if index == 0 else None
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_min_ndim(x, 'x', 3)
        for block in self.blocks:
            x = block(x)
        return x


# ======================================================================================
# TabM modules
# ======================================================================================
TabMArchitectureType = Literal['tabm', 'tabm-mini', 'tabm-packed']


def make_tabm_backbone(
    *,
    d_in: int,
    n_blocks: int,
    d_block: int,
    dropout: float,
    activation: str = 'ReLU',
    k: int,
    arch_type: TabMArchitectureType = 'tabm',
    start_scaling_init: Optional[ScalingRandomInitialization],
    start_scaling_init_chunks: Optional[list[int]],
) -> MLPBackboneEnsembleBase:
    """Make the TabM backbone.

    Args:
        d_in: the input dimension.
        n_blocks: the number of blocks (depth).
        d_block: the latent representation size (width).
        dropout: the dropout rate.
        activation: the activation (must be a valid attribute of the `torch.nn`
            package).
        k: the number of the underlying MLP backbones.
        arch_type: the TabM architecture type.
        start_scaling_init: the initialization of the very first
            scaling (see README of the package for details).
        start_scaling_init_chunks: the initialization chunks of the very first
            scaling (see README of the package for details).
    """
    if arch_type == 'tabm-packed':
        if start_scaling_init is not None:
            raise ValueError(f'When {arch_type=}, start_scaling_init must be None')
    else:
        if start_scaling_init is None:
            raise ValueError(f'When {arch_type=}, start_scaling_init must not be None')
    if start_scaling_init is None and start_scaling_init_chunks is not None:
        raise ValueError(
            f'When {start_scaling_init=}, start_scaling_init_chunks must be None'
        )

    mlp_ensemble_kwargs = {
        'd_in': d_in,
        'n_blocks': n_blocks,
        'd_block': d_block,
        'dropout': dropout,
        'activation': activation,
        'k': k,
    }
    backbone: MLPBackboneEnsembleBase

    if arch_type == 'tabm':
        assert start_scaling_init is not None, _INTERNAL_ERROR_MESSAGE
        if start_scaling_init_chunks is None:
            warnings.warn(
                'start_scaling_init_chunks is not provided,'
                ' which may lead to suboptimal performance in some cases'
            )
        backbone = MLPBackboneBatchEnsemble(
            **mlp_ensemble_kwargs,  # type: ignore[arg-type]
            tabm_init=True,
            scaling_init=start_scaling_init,
            start_scaling_init_chunks=start_scaling_init_chunks,
        )

    elif arch_type == 'tabm-mini':
        assert start_scaling_init is not None, _INTERNAL_ERROR_MESSAGE
        if start_scaling_init_chunks is None:
            warnings.warn(
                'start_scaling_init_chunks is not provided,'
                ' which may lead to suboptimal performance in some cases'
            )
        backbone = MLPBackboneMiniEnsemble(
            **mlp_ensemble_kwargs,  # type: ignore[arg-type]
            affine_scaling_init=start_scaling_init,
            affine_scaling_init_chunks=start_scaling_init_chunks,
            affine_bias=False,
        )

    elif arch_type == 'tabm-packed':
        backbone = MLPBackboneEnsemble(**mlp_ensemble_kwargs)  # type: ignore[arg-type]

    else:
        raise ValueError(f'Unknown {arch_type=}')

    return backbone


_NumEmbeddings = Union[
    rtdl_num_embeddings.LinearReLUEmbeddings,
    rtdl_num_embeddings.PiecewiseLinearEmbeddings,
    rtdl_num_embeddings.PeriodicEmbeddings,
]


class TabM(nn.Module):
    """TabM -- a Tabular DL model that makes Multiple predictions.

    Technically, one TabM efficiently represents an ensemble of $k$ MLPs.
    The two key differences of TabM compared to a conventional deep ensemble of MLPs:

    * **Parallel training** of all $k$ MLPs. This allows monitoring the performance
      of the ensemle during training and stopping the training when it is optimal
      for the ensemble, not for individual MLPs.
    * **Weight sharing** between the MLPs. Not only this significantly improves the
      runtime and memory efficiency, but also surves as an effective regularization
      that further improves the performance.

    **Shape**

    (A) When passing the same batch to each of the $k$ submodels:

    - Input:
        - ``x_num``: ``(batch_size, n_num_features)``
        - ``x_cat``: ``(batch_size, len(cat_cardinalities))``
    - Output: ``(batch_size, k, d_block if d_out is None else d_out)``

    (B) When passing different batches to the $k$ submodels:

    .. note::

        One use case for this strategy is training the $k$ underlying MLPs on
        different batches. In this case, the training input should consist of $k$
        full-fledged batches, **NOT** of one batch reshaped to $k$ small batches.
        The official notebook provides an example of how to implement this.

    - Input:
        - ``x_num``: ``(batch_size, k, n_num_features)``
        - ``x_cat``: ``(batch_size, k, len(cat_cardinalities))``
    - Output: ``(batch_size, k, d_block if d_out is None else d_out)`` (no change)
    """

    num_module: Optional[nn.Module]
    cat_module: Optional[nn.Module]

    def __init__(
        self,
        *,
        n_num_features: int = 0,
        cat_cardinalities: Optional[list[int]] = None,
        d_out: Optional[int],
        num_embeddings: Optional[_NumEmbeddings] = None,
        **backbone_kwargs,
    ) -> None:
        """
        .. note::
            Consider using `TabM.make` instead of this method.

        Args:
            n_num_features: the number of numerical (continuous) features.
            cat_cardinalities: the cardinalities of the categorical features,
                i.e. ``cardinalities[i]`` must store the number of unique categories
                of the i-th categorical feature. The categorical features are encoded
                with the one-hot encoding.
            d_out: the output size. If None, the module's output will be
                the output of the $k$ underlying backbones.
            num_embeddings: the embeddings for numerical features that transform the
                ``x_num`` input before applying the MLP backbones.
                This module is fully shared between the $k$ MLPs.
            backbone_kwargs: all arguments of `make_tabm_backbone`, except for
                ``d_in`` and ``start_scaling_init_chunks`` (they will be inferred
                automatically).
        """
        _check_kwargs(backbone_kwargs, forbidden=['d_in', 'start_scaling_init_chunks'])

        if cat_cardinalities is None:
            cat_cardinalities = []

        if n_num_features < 0:
            raise ValueError(
                f'n_num_features must be a non-negative integer, however:'
                f' {n_num_features=}'
            )
        if n_num_features == 0 and not cat_cardinalities:
            raise ValueError(
                f'{n_num_features=} and {cat_cardinalities=} at the same time,'
                ' which is not allowed'
            )
        if n_num_features == 0 and num_embeddings is not None:
            raise ValueError(f'When {n_num_features=}, num_embeddings must be None')
        if num_embeddings is not None:
            # Unknown embedding types can result in all kinds of weird things,
            # so checking the type explicitly.
            if not isinstance(num_embeddings, typing.get_args(_NumEmbeddings)):
                raise TypeError(
                    f'num_embeddings of the type {type(num_embeddings)} is not'
                    ' supported'
                )
            if (
                isinstance(
                    num_embeddings, rtdl_num_embeddings.PiecewiseLinearEmbeddings
                )
                and num_embeddings.linear0 is None
            ):
                raise ValueError(
                    'When using PiecewiseLinearEmbeddings as num_embeddings,'
                    ' set the version argument to "B":'
                    '\nnum_embeddings = PiecewiseLinearEmbeddings(..., version="B")'
                )

        super().__init__()

        d_features: list[int] = []  # Representation sizes of all features.

        if num_embeddings is None:
            d_features.extend(1 for _ in range(n_num_features))
        else:
            num_embeddings_n_features, d_num_embedding = (
                num_embeddings.get_output_shape()
            )
            if num_embeddings_n_features != n_num_features:
                raise ValueError(
                    'num_embeddings was created for a different number of features'
                    f' than {n_num_features=}'
                )
            d_features.extend(d_num_embedding for _ in range(n_num_features))
        self.num_module = num_embeddings

        if cat_cardinalities:
            # The one-hot representation size of a categorical feature
            # equals its cardinality.
            d_features.extend(cat_cardinalities)
            self.cat_module = _OneHotEncoding(cat_cardinalities)
        else:
            self.cat_module = None

        self.ensemble_view = EnsembleView(k=_get_required_kwarg(backbone_kwargs, 'k'))
        self.backbone = make_tabm_backbone(
            d_in=sum(d_features),  # Flat representation size.
            start_scaling_init_chunks=(
                None
                if _get_required_kwarg(backbone_kwargs, 'start_scaling_init') is None
                else d_features
            ),
            **backbone_kwargs,  # type: ignore[misc]
        )
        self.output = (
            None
            if d_out is None
            else LinearEnsemble(
                self.backbone.get_original_output_shape()[0], d_out, k=self.backbone.k
            )
        )

        self._n_num_features = n_num_features
        self._n_cat_features = len(cat_cardinalities)

    @classmethod
    def make(cls, **kwargs) -> 'TabM':
        """Create TabM.

        Compared to `TabM.__init__`, this function does not require setting all
        model-related arguments (data-related arguments are still required).
        More precisely:

        * The missing model arguments will be set to their default values.
        * The default model argument values are *not* constant, i.e. they depend on the
          provided arguments.
        * Explicitly provided model arguments take precedence over their default values.

        Args:
            kwargs: the arguments for `TabM.__init__`.
        """
        has_num_embeddings = kwargs.get('num_embeddings') is not None
        default_arch_type: TabMArchitectureType = 'tabm'
        defaults = {
            'n_blocks': 2 if has_num_embeddings else 3,
            'd_block': 512,
            'dropout': 0.1,
            'activation': 'ReLU',
            'k': 32,
            'arch_type': default_arch_type,
            'start_scaling_init': (
                None
                if kwargs.get('arch_type', default_arch_type) == 'tabm-packed'
                else 'normal'
                if has_num_embeddings
                else 'random-signs'
            ),
        }
        return TabM(**(defaults | kwargs))

    @property
    def k(self) -> int:
        """The number of submodels."""
        return self.backbone.k

    @staticmethod
    def _get_batch_info(
        x_num: Optional[Tensor], x_cat: Optional[Tensor]
    ) -> tuple[int, int]:
        if x_num is not None:
            batch_size = len(x_num)
            ndim = x_num.ndim
            if x_cat is not None and (x_cat.ndim != ndim or len(x_cat) != batch_size):
                raise ValueError(
                    f'Incompatible shapes of x_num and x_cat:'
                    f' {len(x_num)=} and {len(x_cat)=}'
                )
            return batch_size, ndim
        elif x_cat is not None:
            return len(x_cat), x_cat.ndim
        else:
            raise ValueError('Both x_num and x_cat are None')

    def _check_input(
        self, x_name: str, argname: str, n_features: int, *, x: Optional[Tensor]
    ) -> None:
        if n_features == 0:
            if x is not None:
                raise RuntimeError(
                    f'Based on {argname} passed to the constructor,'
                    f' {x_name} must be None'
                )
        else:
            if x is None:
                raise RuntimeError(
                    f'Based on {argname} passed to the constructor,'
                    f' {x_name} must not be None'
                )
            if x.ndim != 2 and x.ndim != 3:
                raise ValueError(
                    'The input must have either two or three dimensions, however:'
                    f' {x_name}.ndim={x.ndim}'
                )
            if x.shape[-1] != n_features:
                raise RuntimeError(
                    f'Based on {argname} passed to the constructor,'
                    f'the input {x_name} must be a tensor of the shape'
                    f' (batch_size, {n_features}), however: {x_name}.shape={x.shape}'
                )

    def _reshape_input_to_2d(self, x: Optional[Tensor]) -> Optional[Tensor]:
        if x is None:
            return x
        elif x.ndim == 3:
            # (B, K, D) -> (B * K, D)
            return x.flatten(0, -2)
        else:
            assert x.ndim == 2, _INTERNAL_ERROR_MESSAGE
            return x

    def forward(
        self, x_num: Optional[Tensor] = None, x_cat: Optional[Tensor] = None
    ) -> Tensor:
        """Do the forward pass.

        Args:
            x_num: the numerical features. The data type must be float.
            x_cat: the categorical features. The data type must be long.
                The i-th feature must take values in ``range(0, cat_cardinalities[i])``,
                where ``cat_cardinalities`` is the list passed to the constructor.
        """
        self._check_input('x_num', 'n_num_features', self._n_num_features, x=x_num)
        self._check_input('x_cat', 'cat_cardinalities', self._n_cat_features, x=x_cat)

        batch_size, ndim = self._get_batch_info(x_num, x_cat)
        x_num = self._reshape_input_to_2d(x_num)
        x_cat = self._reshape_input_to_2d(x_cat)

        x_: list[Tensor] = []
        if x_num is not None:
            x_.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is not None:
            assert self.cat_module is not None, _INTERNAL_ERROR_MESSAGE
            x_.append(self.cat_module(x_cat))
        x = torch.column_stack([x.flatten(1, -1) for x in x_])

        if ndim == 3:
            # Unflatten the first dimension back to the original shape.
            x = x.unflatten(0, (batch_size, self.backbone.k))  # (B * K, D) -> (B, K, D)

        x = self.ensemble_view(x)
        x = self.backbone(x)
        if self.output is not None:
            x = self.output(x)
        return x
