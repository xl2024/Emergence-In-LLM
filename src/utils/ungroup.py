from typing import Any, Dict, Iterable

from src.utils.tools import _resolve_layer_path, get_layer_paths, get_num_hidden_layers, _resolve_text_model_dims


# These helpers ungroup K/V at the projection-output level. Any linear bias is
# already included in k_proj.output/v_proj.output, so there is no separate bias
# handling here. Bias expansion is only needed for weight-level surgery.


def needs_kv_head_ungrouping(num_heads: int, num_kv_heads: int, ungroup: bool = True) -> bool:
    """
    Return whether GQA KV activations should be expanded from KV heads to Q heads.
    """
    if not ungroup:
        return False
    if num_heads < num_kv_heads:
        raise ValueError(f"num_heads ({num_heads}) must be >= num_kv_heads ({num_kv_heads}).")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}) "
            "to ungroup GQA activations."
        )
    return num_heads > num_kv_heads


def ungroup_kv_head_activation(act: Any, hidden_size: int, num_heads: int, num_kv_heads: int) -> Any:
    """
    Expand a KV projection activation from grouped KV heads to per-query-head KV heads.

    This is intended for nnsight/NDIF traces where the remote model weights cannot be
    surgically replaced. `act` may be a concrete tensor or an nnsight proxy tensor.
    Expected shape is (..., num_kv_heads * head_dim). The returned shape is
    (..., num_heads * head_dim).
    """
    if not needs_kv_head_ungrouping(num_heads, num_kv_heads):
        return act

    head_dim = hidden_size // num_heads
    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")

    num_groups = num_heads // num_kv_heads
    prefix_shape = tuple(act.shape[:-1])
    return (
        act.view(*prefix_shape, num_kv_heads, head_dim)
        .repeat_interleave(num_groups, dim=-2)
        .view(*prefix_shape, num_heads * head_dim)
    )


def maybe_ungroup_kv_head_activation(
    act: Any,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    ungroup: bool = False,
) -> Any:
    """
    Conditionally expand a KV activation when an ungroup flag is enabled and GQA is present.
    """
    if not needs_kv_head_ungrouping(num_heads, num_kv_heads, ungroup=ungroup):
        return act
    return ungroup_kv_head_activation(act, hidden_size, num_heads, num_kv_heads)


def set_remote_attention_to_mha(attn_module: Any, num_heads: int) -> None:
    """
    Best-effort update of attention routing metadata inside an nnsight trace.

    HF attention modules commonly repeat KV heads according to num_key_value_groups.
    After expanding k_proj/v_proj outputs to num_heads, the attention block should
    treat KV as already ungrouped.
    """
    for attr_name, attr_value in (
        ("num_key_value_heads", num_heads),
        ("num_key_value_groups", 1),
        ("num_key_value_groups_per_head", 1),
    ):
        if hasattr(attn_module, attr_name):
            setattr(attn_module, attr_name, attr_value)

def reset_remote_attention_to_gqa_metadata(attn_module: Any, num_heads: int, num_kv_heads: int) -> None:
    """
    Restore native grouped-query attention routing metadata inside an nnsight trace.

    This does not regroup already-expanded k/v projection outputs. If k/v outputs
    have been ungrouped in the current trace, resetting metadata would make the
    attention block inconsistent, so callers should fail before doing that.
    """
    num_groups = num_heads // num_kv_heads
    for attr_name, attr_value in (
        ("num_key_value_heads", num_kv_heads),
        ("num_key_value_groups", num_groups),
        ("num_key_value_groups_per_head", num_groups),
    ):
        if hasattr(attn_module, attr_name):
            setattr(attn_module, attr_name, attr_value)


def ungroup_remote_kv_projection_output(
    target_module: Any,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    ungroup: bool = False,
    save: bool = False,
) -> Any:
    """
    Intercept a remote k_proj/v_proj module output and replace it with ungrouped KV heads.

    Call this inside `with model.trace(..., remote=True):` after resolving the target
    k/v projection module. When `save=True`, the patched output is saved and returned.
    Otherwise the patched nnsight value is returned.
    """
    if needs_kv_head_ungrouping(num_heads, num_kv_heads, ungroup=ungroup):
        patched_output = ungroup_kv_head_activation(
            target_module.output,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        target_module.output = patched_output

    if save:
        return target_module.output.save()
    return target_module.output


def ungroup_remote_layer_kv_heads(
    model: Any,
    layer_idx: int,
    ungroup: bool = False,
    activation_name: str = False,
    save: bool = False,
) -> Dict[str, Any]:
    """
    Ungroup k/v projection outputs for one layer inside an NDIF trace.

    `layer_paths` should be the same mapping used by RSA/CMA scripts:
    `{layer_idx: {"k": "...k_proj", "v": "...v_proj"}}`.
    """
    hidden_size, num_heads = _resolve_text_model_dims(model)
    _, num_kv_heads = _resolve_text_model_dims(model, kv_heads=True)
    layer_paths = get_layer_paths(model, ["k", "v"]) if ungroup else None

    patched_output = None
    if not needs_kv_head_ungrouping(num_heads, num_kv_heads, ungroup=ungroup):
        return patched_output

    if activation_name not in ("k", "v"):
        return patched_output
    if activation_name not in layer_paths[layer_idx]:
        return patched_output

    target_module = _resolve_layer_path(model, layer_paths[layer_idx][activation_name])
    parent_path = layer_paths[layer_idx][activation_name].rsplit(".", 1)[0]
    attn_module = _resolve_layer_path(model, parent_path)
    set_remote_attention_to_mha(attn_module, num_heads)
    patched_output = ungroup_remote_kv_projection_output(
        target_module,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        ungroup=True,
        save=save,
    )
    # custom attribute for tracking status
    ungrouped_outputs = getattr(attn_module, "_ungrouped_kv_projection_outputs", set())
    ungrouped_outputs.add(activation_name)
    setattr(attn_module, "_ungrouped_kv_projection_outputs", ungrouped_outputs)

    return patched_output

def maybe_ungroup_remote_layer_kv_heads(
    model: Any,
    layer_idx: int,
    ungroup: bool = False,
    activation_names: Iterable[str] = ("k", "v"),
    save: bool = False,
) -> Dict[str, Any]:
    """
    Flag-gated wrapper for remote KV-head ungrouping.
    """
    _, num_heads = _resolve_text_model_dims(model)
    _, num_kv_heads = _resolve_text_model_dims(model, kv_heads=True)
    patched_outputs = {}
    if not needs_kv_head_ungrouping(num_heads, num_kv_heads, ungroup=ungroup):
        return patched_outputs
    for activation_name in activation_names:
        patched_outputs[activation_name] = ungroup_remote_layer_kv_heads(
                model=model,
                layer_idx=layer_idx,
                ungroup=True,
                activation_name=activation_name,
                save=save,
            )
    return patched_outputs

def apply_remote_kv_ungrouping(
        model,
        ungroup_grouped_query_attention=False,
        hook_fn=None,
        **hook_kwargs,
        ):
    _, num_heads = _resolve_text_model_dims(model)
    _, num_kv_heads = _resolve_text_model_dims(model, kv_heads=True)
    activation_name = hook_kwargs.get('activation_name')

    if not needs_kv_head_ungrouping(
        num_heads,
        num_kv_heads,
        ungroup=ungroup_grouped_query_attention,
    ) or not activation_name in ['k','v']:
        if hook_fn is not None:
            hook_fn(model, need_ungroup=False, **hook_kwargs)
        return

    hook_layer_idx = hook_kwargs.get('layer_idx')
    for act in ['k','v']:
        patched_output = ungroup_remote_layer_kv_heads(
            model,
            layer_idx=hook_layer_idx,
            ungroup=True,
            activation_name=act,
        )     # hook both k,v
        if act == activation_name and hook_fn is not None:
            hook_fn(model, need_ungroup=True, patched_output=patched_output, **hook_kwargs)

def check_remote_layer_kv_metadata(
    model: Any,
    layer_idx: int,
    ungroup_grouped_query_attention: bool
) -> None:
    """
    Restore native GQA metadata for one layer inside an NDIF trace.

    This does not regroup k/v activations. It asserts if k/v projection outputs
    were already expanded in the same trace.
    """
    _, num_heads = _resolve_text_model_dims(model)
    _, num_kv_heads = _resolve_text_model_dims(model, kv_heads=True)
    layer_paths = get_layer_paths(model, ["k", "v"]) if ungroup_grouped_query_attention else None
    
    if num_heads == num_kv_heads:
        return
    if "k" not in layer_paths[layer_idx]:
        return

    parent_path = layer_paths[layer_idx]["k"].rsplit(".", 1)[0]
    attn_module = _resolve_layer_path(model, parent_path)
    ungrouped_outputs = getattr(attn_module, "_ungrouped_kv_projection_outputs", set())
    if ungrouped_outputs:
        raise AssertionError(
            "Cannot reset GQA metadata after ungrouping k/v projection outputs "
            f"in the same trace. Already ungrouped: {sorted(ungrouped_outputs)}."
        )
    reset_remote_attention_to_gqa_metadata(attn_module, num_heads, num_kv_heads)

def check_remote_kv_metadata(model, ungroup_grouped_query_attention):
    total_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)
    _, num_kv_heads = _resolve_text_model_dims(model, kv_heads=True)

    if not needs_kv_head_ungrouping(num_heads, num_kv_heads, ungroup=True):
        return

    for layer_idx in range(total_layers):
        check_remote_layer_kv_metadata(
            model,
            layer_idx=layer_idx,
            ungroup_grouped_query_attention=ungroup_grouped_query_attention
        )
