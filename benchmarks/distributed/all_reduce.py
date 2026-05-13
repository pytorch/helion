from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .experiment_util import BenchmarkOperator
from .experiment_util import ExperimentConfig

if TYPE_CHECKING:
    import argparse

BUILDTIN_SHAPES = [
    4093,
    4096,
    5000,
    8192,
    8193,
    16384,
    16380,
    16387,
]
LARGE_K_SHAPES = [2**exp for exp in range(15, 21)]


class AllReduceBench(BenchmarkOperator):
    def gen_configs(self, args: argparse.Namespace) -> list[ExperimentConfig]:
        all_configs = []
        for sz in args.shape:
            all_configs.append(
                ExperimentConfig(
                    shape=(sz,),
                    dtype=args.dtype,
                    backends=args.backend,
                    device=self.device,
                )
            )

        return all_configs

    def gen_inputs(self, config: ExperimentConfig) -> tuple:
        input_tensor = symm_mem.empty(
            config.shape,
            dtype=config.dtype,
            device=config.device,
        )
        assert dist.group.WORLD is not None
        symm_mem.rendezvous(input_tensor, dist.group.WORLD.group_name)
        input_tensor = input_tensor.normal_()
        return (input_tensor,)

    def additional_parser_args(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        parser.add_argument(
            "--shape",
            type=int,
            nargs="+",
            default=BUILDTIN_SHAPES + LARGE_K_SHAPES,
            help="Tensor lengths",
        )
        return parser

    def __init__(self) -> None:
        self.op_name = "allreduce"
        self.baseline = "nccl"
        super().__init__()

        def nccl_ring(msg: torch.Tensor) -> torch.Tensor:
            dist.all_reduce(msg)
            return msg

        assert dist.group.WORLD is not None

        ALLREDUCE_DICT = {
            "multimem": functools.partial(
                torch.ops.symm_mem.multimem_all_reduce_,
                reduce_op="sum",
                group_name=dist.group.WORLD.group_name,
            ),
            "oneshot": functools.partial(
                torch.ops.symm_mem.one_shot_all_reduce,
                reduce_op="sum",
                group_name=dist.group.WORLD.group_name,
            ),
            "twoshot": functools.partial(
                torch.ops.symm_mem.two_shot_all_reduce_,
                reduce_op="sum",
                group_name=dist.group.WORLD.group_name,
            ),
            "nccl": nccl_ring,
            "helion_oneshot": ("examples.all_reduce", "helion_one_shot_all_reduce"),
            "kraken_oneshot": ("kraken.all_reduce", "one_shot_all_reduce"),
        }
        self.backend_dict = ALLREDUCE_DICT
