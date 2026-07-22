# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext

from .harness import DispatchTestCase


class TestArangeConverter(DispatchTestCase):
    def test_static_arange_uses_linspace_fill(self):
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        ctx = ConversionContext(network)

        output = impl.arange.arange(
            ctx,
            torch.ops.aten.arange.start_step,
            SourceIR.ATEN,
            "arange",
            start=0,
            end=5,
            step=1,
        )

        fill_layers = [
            network.get_layer(index)
            for index in range(network.num_layers)
            if network.get_layer(index).type == trt.LayerType.FILL
        ]
        self.assertEqual(len(fill_layers), 1)
        self.assertEqual(
            fill_layers[0].name,
            "[FILL]-[aten_ops.arange.start_step]-[arange_arange_fill]",
        )
        self.assertEqual(tuple(output.shape), (5,))

    @parameterized.expand(
        [
            (0, 5, 1),
            (1, 5, 2),
            (3, 5, 3),
            (5, 0, -1),
            (5, 1, -2),
            (5, 3, -3),
            (5, -2, -1),
            (-5, -2, 2),
            (-5, -3, 1),
            (-2, -5, -1),
            (1.2, 5, 1.3),
            (1.2, 5.0, 1.3),
            (1, 5.0, 1.3),
            (-1.2, -5.0, -1.3),
            (-5.0, -1.2, 1.3),
            (-5, 1.2, 1.3),
            (-5.0, 1, 1.3),
        ]
    )
    def test_arange(self, start, end, step):
        class Arange(nn.Module):
            def forward(self, x):
                return torch.ops.aten.arange.start_step(start, end, step)

        inputs = [torch.randn(1, 1)]
        self.run_test(
            Arange(),
            inputs,
            use_dynamo_tracer=True,
        )

    def test_arange_static_non_numpy_type(self):
        class Arange(nn.Module):
            def forward(self, x):
                return torch.ops.aten.arange.start_step(
                    0, 5, 1, dtype=torch.bfloat16, device=x.device
                )

        self.run_test(
            Arange(),
            [torch.randn(1, 1)],
            use_dynamo_tracer=True,
        )

    def test_arange_dynamic_int32(self):
        class Arange(nn.Module):
            def forward(self, end_tensor):
                return torch.ops.aten.arange.start_step(0, end_tensor, 1)

        pyt_input = 7
        inputs = [
            torch_tensorrt.Input(
                min_shape=(5,),
                opt_shape=(7,),
                max_shape=(10,),
                dtype=torch.int32,
                torch_tensor=torch.tensor(pyt_input, dtype=torch.int32).cuda(),
                is_shape_tensor=True,
            )
        ]
        self.run_test_with_dynamic_shape(
            Arange(),
            inputs,
            use_example_tensors=False,
            check_dtype=False,
            pyt_inputs=[pyt_input],
            use_dynamo_tracer=False,
        )

    def test_arange_dynamic_int64(self):
        class Arange(nn.Module):
            def forward(self, end_tensor):
                return torch.ops.aten.arange.start_step(0, end_tensor, 1)

        pyt_input = 7
        inputs = [
            torch_tensorrt.Input(
                min_shape=(5,),
                opt_shape=(7,),
                max_shape=(10,),
                dtype=torch.int64,
                torch_tensor=torch.tensor(pyt_input, dtype=torch.int64).cuda(),
                is_shape_tensor=True,
            )
        ]
        self.run_test_with_dynamic_shape(
            Arange(),
            inputs,
            use_example_tensors=False,
            check_dtype=False,
            pyt_inputs=[pyt_input],
            use_dynamo_tracer=False,
        )

    def test_arange_data_dependent_start(self):
        class Arange(torch.nn.Module):
            def forward(self, x, mask):
                end = mask.nonzero().size(0)
                start = end // 2
                indices = torch.arange(start, end, 1, device=x.device)
                return x.index_select(0, indices)

        previous_capture_setting = torch._dynamo.config.capture_dynamic_output_shape_ops
        try:
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.reset()

            model = Arange().eval().cuda()
            x = torch.randn((16, 8), device="cuda")
            mask = torch.arange(16, device="cuda") % 2 == 0
            expected = model(x, mask)

            compiled = torch.compile(
                model,
                backend="tensorrt",
                options={
                    "pass_through_build_failures": True,
                    "min_block_size": 1,
                },
            )
            torch.testing.assert_close(compiled(x, mask), expected)
        finally:
            torch._dynamo.config.capture_dynamic_output_shape_ops = (
                previous_capture_setting
            )
            torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
