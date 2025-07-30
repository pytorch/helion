from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .experiment_util import BenchmarkOperator
from .experiment_util import ExperimentConfig

BUILDIN_SHAPES = [
    (256, 256, 256),
    (384, 384, 384),
    (512, 512, 512),
    (640, 640, 640),
    (768, 768, 768),
    (896, 896, 896),
    (1024, 1024, 1024),
    (1152, 1152, 1152),
    (1280, 1280, 1280),
    (1408, 1408, 1408),
    (1536, 1536, 1536),
    (1664, 1664, 1664),
    (1792, 1792, 1792),
    (1920, 1920, 1920),
    (2048, 2048, 2048),
    (2176, 2176, 2176),
    (2304, 2304, 2304),
    (2432, 2432, 2432),
    (2560, 2560, 2560),
    (2688, 2688, 2688),
    (2816, 2816, 2816),
    (2944, 2944, 2944),
    (3072, 3072, 3072),
    (3200, 3200, 3200),
    (3328, 3328, 3328),
    (3456, 3456, 3456),
    (3584, 3584, 3584),
    (3712, 3712, 3712),
    (3840, 3840, 3840),
    (3968, 3968, 3968),
    (4096, 4096, 4096),
]


class AGMatmulBench(BenchmarkOperator):
    def gen_configs(self, args: argparse.Namespace) -> list[ExperimentConfig]:
        all_configs = []
        for sz in args.shape:
            all_configs.append(
                ExperimentConfig(
                    shape=sz,
                    dtype=args.dtype,
                    backends=args.backend,
                    device=self.device,
                )
            )

        return all_configs

    def gen_inputs(self, config: ExperimentConfig) -> tuple:
        M, N, K = config.shape
        a = symm_mem.empty(
            (M, K),
            dtype=config.dtype,
            device=config.device,
        )
        b = (
            torch.randn((K, N), device=config.device, dtype=config.dtype)
            .T.contiguous()
            .T
        )
        assert dist.group.WORLD is not None
        symm_mem.rendezvous(a, dist.group.WORLD.group_name)
        return (a, b)

    def additional_parser_args(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        def matmul_shape_type(s: str) -> tuple[int, int, int]:
            try:
                M, N, K = map(int, s.split(","))
                return M, N, K
            except Exception as e:
                raise argparse.ArgumentTypeError(
                    "Matmul shape must be M, N, K. (M, K) @ (K, N) -> (M, N)"
                ) from e

        parser.add_argument(
            "--shape",
            type=matmul_shape_type,
            nargs="+",
            default=BUILDIN_SHAPES,
            help="matmul shapes: (M, N, K). (M, K) @ (K, N) -> (M, N)",
        )
        return parser

    def __init__(self) -> None:
        self.op_name = "ag_matmul"
        self.baseline = "nccl"
        super().__init__()

        def nccl_mem_ag_mm(
            a_shared: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            from torch.distributed._functional_collectives import all_gather_tensor

            a_gathered = all_gather_tensor(a_shared, 0, "0")
            return a_gathered, torch.matmul(a_gathered, b)

        def torch_symm_mem_ag_mm(
            a_shared: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert dist.group.WORLD is not None
            a_gathered, c = torch.ops.symm_mem.fused_all_gather_matmul(
                a_shared, [b], gather_dim=0, group_name=dist.group.WORLD.group_name
            )
            return a_gathered, c[0]

        assert dist.group.WORLD is not None

        AG_MATMUL_DICT = {
            "nccl": nccl_mem_ag_mm,
            "torch_symm_mem": torch_symm_mem_ag_mm,
            "helion": ("examples.all_gather_matmul", "helion_all_gather_matmul"),
            "kraken": ("kraken.all_gather", "all_gather_matmul"),
        }
        self.backend_dict = AG_MATMUL_DICT
