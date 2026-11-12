"""Convert the named Keras H5 weights used by Shorkie and ExoShorkie.

The shared loader covers the trunk only. Each model loads its own head after
calling :func:`load_trunk_weights`. Tensor shapes, per-layer keys, and the set
of weighted layers are checked so incompatible checkpoints fail explicitly.
"""

from collections import defaultdict

import h5py
import numpy as np
import torch
from torch import nn


def _weight_root(root: h5py.Group) -> h5py.Group:
    return root.get("model_weights", root)


def get_tf_layer(root: h5py.Group, name: str) -> dict[str, np.ndarray]:
    """Read a layer from a full-model or weights-only Keras H5 checkpoint."""
    root = _weight_root(root)
    if name not in root:
        raise ValueError(f"Checkpoint is missing layer {name!r}")
    group = root[name]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Expected an H5 group for layer {name!r}")  # noqa: TRY004
    # Keras commonly nests layer_name/layer_name/parameter:0. Visit the
    # outer group too, so unexpected sibling weights cannot be skipped.
    prefix = f"{name}/" if name in group and isinstance(group[name], h5py.Group) else ""
    params = {}

    def collect(key, value):
        if isinstance(value, h5py.Dataset):
            key = key.removeprefix(prefix)
            if key in params:
                raise ValueError(f"Duplicate weight {key!r} in layer {name!r}")
            params[key] = np.asarray(value)

    group.visititems(collect)
    return params


def _require_keys(params: dict, expected: set[str]) -> None:
    actual = set(params)
    if actual != expected:
        raise ValueError(
            "Checkpoint layer has incompatible weights: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _copy(target: torch.Tensor, value: np.ndarray, name: str) -> None:
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f"Checkpoint tensor {name!r} has shape {tuple(value.shape)}; "
            f"expected {tuple(target.shape)}"
        )
    with torch.no_grad():
        target.copy_(torch.from_numpy(value))


def load_linear(layer: nn.Linear, params: dict[str, np.ndarray]) -> None:
    """Load a Keras Dense layer, transposing its input/output dimensions."""
    _require_keys(params, {"kernel:0", "bias:0"})
    _copy(layer.weight, params["kernel:0"].T, "dense.kernel")
    _copy(layer.bias, params["bias:0"], "dense.bias")


def _load_conv1d(layer: nn.Conv1d, params: dict[str, np.ndarray]) -> None:
    _require_keys(params, {"kernel:0", "bias:0"})
    # Keras (kernel, input, output) -> PyTorch (output, input, kernel).
    _copy(layer.weight, params["kernel:0"].transpose(2, 1, 0), "conv1d.kernel")
    _copy(layer.bias, params["bias:0"], "conv1d.bias")


def _load_batch_norm(layer: nn.BatchNorm1d, params: dict[str, np.ndarray]) -> None:
    _require_keys(params, {"gamma:0", "beta:0", "moving_mean:0", "moving_variance:0"})
    for target, key in (
        (layer.weight, "gamma:0"),
        (layer.bias, "beta:0"),
        (layer.running_mean, "moving_mean:0"),
        (layer.running_var, "moving_variance:0"),
    ):
        _copy(target, params[key], f"batch_normalization.{key}")


def _load_layer_norm(layer: nn.LayerNorm, params: dict[str, np.ndarray]) -> None:
    _require_keys(params, {"gamma:0", "beta:0"})
    _copy(layer.weight, params["gamma:0"], "layer_normalization.gamma")
    _copy(layer.bias, params["beta:0"], "layer_normalization.beta")


def _weighted_layers(root: h5py.Group) -> set[str]:
    """Find every top-level layer containing a weight dataset."""
    names = set()
    for name, group in root.items():
        if isinstance(group, h5py.Dataset):
            names.add(name)
        else:
            has_weight = group.visititems(
                lambda _, obj: True if isinstance(obj, h5py.Dataset) else None
            )
            if has_weight:
                names.add(name)
    return names


def load_trunk_weights(
    backbone: nn.Module,
    root: h5py.Group,
    *,
    head_name: str | None = None,
) -> int:
    """Load the common trunk and return the next numbered Dense-layer index.

    ``head_name`` declares the one weighted layer left for the caller to load.
    Its default is Shorkie's final numbered Dense layer; ExoShorkie passes
    ``"distill_head"``. Additional or missing weighted layers are rejected.
    """
    root = _weight_root(root)
    indices = defaultdict(int)
    loaded_layers = set()

    def next_layer(kind):
        index = indices[kind]
        indices[kind] += 1
        name = kind if index == 0 else f"{kind}_{index}"
        loaded_layers.add(name)
        return get_tf_layer(root, name)

    _load_conv1d(backbone.conv_dna.conv, next_layer("conv1d"))

    for block in backbone.res_tower.blocks:
        for conv in (block.conv0, *block.conv_stack):
            _load_batch_norm(conv.norm, next_layer("batch_normalization"))
            _load_conv1d(conv.conv, next_layer("conv1d"))
        params = next_layer("scale")
        _require_keys(params, {"scale:0"})
        _copy(block.res_scale, params["scale:0"][:, None], "scale")

    for block in backbone.transformer_tower.blocks:
        _load_layer_norm(block.attn_norm, next_layer("layer_normalization"))
        params = next_layer("multihead_attention")
        _require_keys(params, {
            "q_layer/kernel:0", "k_layer/kernel:0", "v_layer/kernel:0",
            "r_k_layer/kernel:0", "r_w_bias:0", "r_r_bias:0",
            "embedding_layer/kernel:0", "embedding_layer/bias:0",
        })
        mha = block.mha
        for name in ("q_layer", "k_layer", "v_layer", "r_k_layer", "embedding_layer"):
            _copy(getattr(mha, name).weight, params[f"{name}/kernel:0"].T, name)
        _copy(mha.r_w_bias, params["r_w_bias:0"], "r_w_bias")
        _copy(mha.r_r_bias, params["r_r_bias:0"], "r_r_bias")
        _copy(mha.embedding_layer.bias, params["embedding_layer/bias:0"], "embedding_bias")

        _load_layer_norm(block.ffn_norm, next_layer("layer_normalization"))
        load_linear(block.ffn_dense1, next_layer("dense"))
        load_linear(block.ffn_dense2, next_layer("dense"))

    for unet in backbone.unet_convs:
        _load_batch_norm(unet.norm1, next_layer("batch_normalization"))
        _load_batch_norm(unet.norm2, next_layer("batch_normalization"))
        if unet.upsample_conv:
            load_linear(unet.dense1, next_layer("dense"))
        load_linear(unet.dense2, next_layer("dense"))

        params = next_layer("separable_conv1d")
        _require_keys(params, {"depthwise_kernel:0", "pointwise_kernel:0", "bias:0"})
        _copy(
            unet.sep_conv.depthwise.weight,
            params["depthwise_kernel:0"].transpose(1, 2, 0),
            "separable_conv1d.depthwise_kernel",
        )
        _copy(
            unet.sep_conv.pointwise.weight,
            params["pointwise_kernel:0"].transpose(2, 1, 0),
            "separable_conv1d.pointwise_kernel",
        )
        _copy(unet.sep_conv.pointwise.bias, params["bias:0"], "separable_conv1d.bias")

    dense_index = indices["dense"]
    if head_name is None:
        head_name = "dense" if dense_index == 0 else f"dense_{dense_index}"
    expected = loaded_layers | {head_name}
    actual = _weighted_layers(root)
    if actual != expected:
        raise ValueError(
            "Checkpoint has incompatible weighted layers: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return dense_index
