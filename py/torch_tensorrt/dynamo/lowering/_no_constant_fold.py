from typing import Any, Callable, Collection, Iterable, Optional

import torch
from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor
from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot


NO_CONSTANT_FOLD_META_KEY = "_torch_tensorrt_no_constant_fold"
ATTENTION_MASK_ARANGE_RULE_ID = "attention_mask_arange"

NoConstantFoldRule = Callable[[torch.fx.Node], Iterable[torch.fx.Node]]
_NO_CONSTANT_FOLD_RULES: dict[str, NoConstantFoldRule] = {}


def _is_arange_node(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and getattr(node.target, "overloadpacket", None) is torch.ops.aten.arange
    )


def _find_ancestor_nodes(
    node: torch.fx.Node,
    predicate: Callable[[torch.fx.Node], bool],
) -> list[torch.fx.Node]:
    """Find ancestors of ``node`` that satisfy ``predicate``."""
    nodes_to_visit: list[torch.fx.Node] = [node]
    visited: set[torch.fx.Node] = set()
    matching_nodes: list[torch.fx.Node] = []

    while nodes_to_visit:
        current = nodes_to_visit.pop()
        if current in visited:
            continue

        visited.add(current)
        if predicate(current):
            matching_nodes.append(current)

        nodes_to_visit.extend(current.all_input_nodes)

    return matching_nodes


def _mark_no_constant_fold(nodes: Iterable[torch.fx.Node], rule_id: str) -> None:
    """Record which rule wants ``nodes`` kept out of constant folding.

    The marks carry their rule ID rather than a bare flag so that
    :func:`mark_no_constant_fold_nodes` can revoke the ones belonging to
    disabled rules, whichever marking path produced them.
    """
    for node in nodes:
        node.meta.setdefault(NO_CONSTANT_FOLD_META_KEY, set()).add(rule_id)


def register_no_constant_fold_rule(
    rule_id: str,
) -> Callable[[NoConstantFoldRule], NoConstantFoldRule]:
    """Register a named rule that selects FX nodes to exclude from folding."""
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError("A no-constant-fold rule ID must be a non-empty string")

    def register(rule: NoConstantFoldRule) -> NoConstantFoldRule:
        if rule_id in _NO_CONSTANT_FOLD_RULES:
            raise ValueError(
                f"No-constant-fold rule {rule_id!r} is already registered"
            )

        _NO_CONSTANT_FOLD_RULES[rule_id] = rule
        return rule

    return register


def validate_disabled_no_constant_fold_rules(
    rule_ids: Collection[str],
) -> set[str]:
    """Validate disabled rule IDs and return them as a set."""
    if isinstance(rule_ids, str):
        raise TypeError(
            "disabled_no_constant_fold_rules must be a collection of rule IDs, "
            "not a single string"
        )

    disabled_rule_ids = set(rule_ids)
    unknown_rule_ids = disabled_rule_ids - _NO_CONSTANT_FOLD_RULES.keys()
    if unknown_rule_ids:
        available_rule_ids = ", ".join(sorted(_NO_CONSTANT_FOLD_RULES))
        raise ValueError(
            "Unknown no-constant-fold rule IDs: "
            f"{sorted(unknown_rule_ids)}. Available rule IDs: "
            f"[{available_rule_ids}]"
        )

    return disabled_rule_ids


def mark_attn_mask_aranges_no_constant_fold(attn_mask: torch.Tensor) -> None:
    """Mark aranges behind an attention mask while tracing a decomposition.

    ``run_decompositions`` invokes decompositions once for functionalization and
    again while proxy tracing. Only the proxy-tracing invocation has an FX graph
    on which metadata can be set.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is None:
        return

    unwrapped_mask = mb_unwrap_functional_tensor(attn_mask)
    tracked_mask = get_proxy_slot(
        unwrapped_mask,
        proxy_mode.tracer,
        default=None,
    )
    proxy = getattr(tracked_mask, "proxy", tracked_mask)
    mask_node = getattr(proxy, "node", None)

    if isinstance(mask_node, torch.fx.Node):
        _mark_no_constant_fold(
            _find_ancestor_nodes(mask_node, predicate=_is_arange_node),
            ATTENTION_MASK_ARANGE_RULE_ID,
        )


@register_no_constant_fold_rule(ATTENTION_MASK_ARANGE_RULE_ID)
def _attention_mask_arange_rule(node: torch.fx.Node) -> Iterable[torch.fx.Node]:
    """Select aranges feeding an attention mask."""
    # Every SDPA overload that takes a mask keeps it at positional index 3.
    # _scaled_dot_product_flash_attention is absent because it has no mask.
    attention_mask_args: dict[Any, tuple[int, str]] = {
        torch.ops.aten.scaled_dot_product_attention: (3, "attn_mask"),
        torch.ops.aten._scaled_dot_product_efficient_attention: (3, "attn_bias"),
        torch.ops.aten._scaled_dot_product_cudnn_attention: (3, "attn_bias"),
    }

    if node.op != "call_function":
        return ()

    overload_packet = getattr(node.target, "overloadpacket", None)
    mask_arg = attention_mask_args.get(overload_packet)
    if mask_arg is None:
        return ()

    mask_index, mask_kwarg = mask_arg
    mask = node.kwargs.get(
        mask_kwarg,
        node.args[mask_index] if len(node.args) > mask_index else None,
    )
    if not isinstance(mask, torch.fx.Node):
        return ()

    return _find_ancestor_nodes(mask, predicate=_is_arange_node)


def mark_no_constant_fold_nodes(
    gm: torch.fx.GraphModule, settings: Optional[Any] = None
) -> torch.fx.GraphModule:
    """Apply registered rules that label FX nodes as non-foldable.

    This pass is the single authority on which rules are in effect. It runs
    immediately before ``constant_fold`` and is the only marking path that sees
    ``settings``: rules that mark nodes while a decomposition is traced run
    during ``run_decompositions``, long before a settings object is reachable.
    Those marks are therefore revoked here rather than suppressed where they are
    made, so a caller only has to communicate the disabled rules once.
    """
    disabled_rule_ids = validate_disabled_no_constant_fold_rules(
        settings.disabled_no_constant_fold_rules if settings is not None else ()
    )

    for node in gm.graph.nodes:
        for rule_id, rule in _NO_CONSTANT_FOLD_RULES.items():
            if rule_id in disabled_rule_ids:
                continue
            _mark_no_constant_fold(rule(node), rule_id)

    if disabled_rule_ids:
        for node in gm.graph.nodes:
            marking_rule_ids = node.meta.get(NO_CONSTANT_FOLD_META_KEY)
            if marking_rule_ids:
                marking_rule_ids -= disabled_rule_ids

    return gm
