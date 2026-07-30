import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import (
    CompilationSettings,
    settings_are_compatible,
)
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
)
from torch_tensorrt.dynamo.lowering._no_constant_fold import (
    ATTENTION_MASK_ARANGE_RULE_ID,
    NO_CONSTANT_FOLD_META_KEY,
    mark_no_constant_fold_nodes,
    register_no_constant_fold_rule,
)


class TestAttentionMaskNoConstantFold(TestCase):
    class AttentionWithCausalMask(torch.nn.Module):
        def forward(self, query, key, value, attention_mask):
            sequence_length = query.shape[-2]
            row = torch.arange(sequence_length, device=query.device)
            col = torch.arange(sequence_length, device=query.device)
            causal_mask = col.unsqueeze(0) <= row.unsqueeze(1)
            combined_mask = causal_mask & attention_mask
            unrelated_arange = torch.arange(3, device=query.device)
            attention = torch.ops.aten.scaled_dot_product_attention.default(
                query,
                key,
                value,
                combined_mask,
            )
            return attention, unrelated_arange

    def _export(self):
        inputs = (
            torch.randn(1, 2, 8, 16, device="cuda"),
            torch.randn(1, 2, 8, 16, device="cuda"),
            torch.randn(1, 2, 8, 16, device="cuda"),
            torch.ones(8, 8, dtype=torch.bool, device="cuda"),
        )
        return torch.export.export(self.AttentionWithCausalMask(), inputs)

    def _assert_only_attention_aranges_survive(
        self, decompose_attention, disabled_no_constant_fold_rules=()
    ):
        exported_program = self._export().run_decompositions(
            get_decompositions(
                decompose_attention=decompose_attention,
                disabled_no_constant_fold_rules=disabled_no_constant_fold_rules,
            )
        )
        gm = post_lowering(
            exported_program.module(),
            CompilationSettings(
                disabled_no_constant_fold_rules=disabled_no_constant_fold_rules
            ),
        )

        arange_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and getattr(node.target, "overloadpacket", None)
            is torch.ops.aten.arange
        ]
        self.assertEqual(len(arange_nodes), 2)
        self.assertTrue(
            all(
                node.meta.get(NO_CONSTANT_FOLD_META_KEY, False)
                for node in arange_nodes
            )
        )

    def test_disabled_rules_default_to_empty(self):
        self.assertEqual(CompilationSettings().disabled_no_constant_fold_rules, set())
        self.assertEqual(
            CompilationSettings(
                disabled_no_constant_fold_rules=[ATTENTION_MASK_ARANGE_RULE_ID]
            ).disabled_no_constant_fold_rules,
            {ATTENTION_MASK_ARANGE_RULE_ID},
        )
        with self.assertRaisesRegex(TypeError, "collection of rule IDs"):
            CompilationSettings(
                disabled_no_constant_fold_rules=ATTENTION_MASK_ARANGE_RULE_ID
            )

    def test_old_serialized_setting_defaults_to_no_disabled_rules(self):
        state = CompilationSettings().__dict__.copy()
        state.pop("disabled_no_constant_fold_rules")
        restored = CompilationSettings.__new__(CompilationSettings)
        restored.__setstate__(state)
        self.assertEqual(restored.disabled_no_constant_fold_rules, set())

    def test_setting_changes_engine_compatibility(self):
        compatible, incompatible_settings = settings_are_compatible(
            CompilationSettings(),
            CompilationSettings(
                disabled_no_constant_fold_rules={ATTENTION_MASK_ARANGE_RULE_ID}
            ),
        )
        self.assertFalse(compatible)
        self.assertIn(
            "disabled_no_constant_fold_rules",
            incompatible_settings,
        )

    def test_decomposed_attention_mask_aranges_are_not_folded(self):
        self._assert_only_attention_aranges_survive(decompose_attention=True)

    def test_ia_attention_mask_aranges_are_not_folded(self):
        self._assert_only_attention_aranges_survive(decompose_attention=False)

    def test_decomposed_attention_rules_can_be_disabled(self):
        exported_program = self._export().run_decompositions(
            get_decompositions(
                decompose_attention=True,
                disabled_no_constant_fold_rules={ATTENTION_MASK_ARANGE_RULE_ID},
            )
        )
        gm = post_lowering(
            exported_program.module(),
            CompilationSettings(
                disabled_no_constant_fold_rules={ATTENTION_MASK_ARANGE_RULE_ID}
            ),
        )
        self.assertFalse(
            any(
                node.op == "call_function"
                and getattr(node.target, "overloadpacket", None)
                is torch.ops.aten.arange
                for node in gm.graph.nodes
            )
        )

    def test_native_attention_rules_can_be_disabled(self):
        exported_program = self._export().run_decompositions(
            get_decompositions(decompose_attention=False)
        )
        gm = post_lowering(
            exported_program.module(),
            CompilationSettings(
                disabled_no_constant_fold_rules={ATTENTION_MASK_ARANGE_RULE_ID}
            ),
        )
        self.assertFalse(
            any(
                node.op == "call_function"
                and getattr(node.target, "overloadpacket", None)
                is torch.ops.aten.arange
                for node in gm.graph.nodes
            )
        )

    def test_decomposed_attention_rules_can_be_disabled_at_post_lowering_only(self):
        """post_lowering alone is enough to disable a rule.

        get_decompositions is public API, so the manual lowering sequence used
        by the examples can build the decompositions without knowing about the
        disabled rules. The trace-time marks must be revoked at post_lowering
        rather than relying on the caller to pass the setting to both.
        """
        exported_program = self._export().run_decompositions(
            get_decompositions(decompose_attention=True)
        )
        gm = post_lowering(
            exported_program.module(),
            CompilationSettings(
                disabled_no_constant_fold_rules={ATTENTION_MASK_ARANGE_RULE_ID}
            ),
        )
        self.assertFalse(
            any(
                node.op == "call_function"
                and getattr(node.target, "overloadpacket", None)
                is torch.ops.aten.arange
                for node in gm.graph.nodes
            )
        )

    def test_registered_rule_can_mark_an_arbitrary_node(self):
        def custom_target():
            return torch.ones(1)

        graph = torch.fx.Graph()
        custom_node = graph.call_function(custom_target)
        graph.output(custom_node)
        gm = torch.fx.GraphModule({}, graph)

        @register_no_constant_fold_rule("test_arbitrary_node")
        def custom_rule(node):
            return (node,) if node.target is custom_target else ()

        mark_no_constant_fold_nodes(gm)
        self.assertTrue(custom_node.meta[NO_CONSTANT_FOLD_META_KEY])

    def test_unknown_disabled_rule_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown no-constant-fold rule IDs",
        ):
            get_decompositions(
                disabled_no_constant_fold_rules={"unknown_rule"},
            )


class TestAttentionMaskArangeRuleCoverage(TestCase):
    """The rule must cover every SDPA overload that carries an attention mask.

    These graphs are hand-built and never executed, so the coverage check does
    not depend on which SDPA backend the local GPU happens to dispatch to.
    """

    MASKED_ATTENTION_OPS = (
        (torch.ops.aten.scaled_dot_product_attention.default, "attn_mask"),
        (torch.ops.aten._scaled_dot_product_efficient_attention.default, "attn_bias"),
        (torch.ops.aten._scaled_dot_product_cudnn_attention.default, "attn_bias"),
    )

    def _attention_graph(self, target, mask_kwarg):
        graph = torch.fx.Graph()
        query = graph.placeholder("query")
        key = graph.placeholder("key")
        value = graph.placeholder("value")
        arange = graph.call_function(torch.ops.aten.arange.default, (8,))
        mask = graph.call_function(torch.ops.aten.unsqueeze.default, (arange, 0))
        if mask_kwarg is None:
            attention = graph.call_function(target, (query, key, value, mask))
        else:
            attention = graph.call_function(
                target, (query, key, value), {mask_kwarg: mask}
            )
        graph.output(attention)
        return torch.fx.GraphModule({}, graph), arange

    def test_mask_aranges_are_marked_for_every_masked_attention_op(self):
        for target, mask_kwarg in self.MASKED_ATTENTION_OPS:
            for passed_as_kwarg in (False, True):
                with self.subTest(target=target, passed_as_kwarg=passed_as_kwarg):
                    gm, arange = self._attention_graph(
                        target, mask_kwarg if passed_as_kwarg else None
                    )
                    mark_no_constant_fold_nodes(gm)
                    self.assertTrue(arange.meta.get(NO_CONSTANT_FOLD_META_KEY))


if __name__ == "__main__":
    run_tests()
