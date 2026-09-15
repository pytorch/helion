# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public compatibility surface for transactional direct-affine replay."""

from __future__ import annotations

from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR as HELION_LANE_LOOP_VAR_ATTR
from .direct_affine_emission import (
    _split_feature_reduction_loop as _split_feature_reduction_loop,
)
from .direct_affine_emission import compose_direct_affine_replacement
from .direct_affine_replay_core import DirectAffineAsyncEntryReplay
from .direct_affine_replay_core import DirectAffineCoordinates
from .direct_affine_replay_core import DirectAffineEffectReplay
from .direct_affine_replay_core import DirectAffineEmission
from .direct_affine_replay_core import DirectAffineNodeReplay
from .direct_affine_replay_core import DirectAffineOrdinaryAxis
from .direct_affine_replay_core import DirectAffineReplay
from .direct_affine_replay_core import DirectAffineReplayBindings
from .direct_affine_replay_core import DirectAffineReplayProof
from .direct_affine_replay_core import DirectAffineResolvedTemplates
from .direct_affine_replay_core import DirectAffineSharedBuffer
from .direct_affine_replay_core import DirectAffineStateStoreReplay
from .direct_affine_replay_core import DirectAffineStepReplay
from .direct_affine_replay_core import DirectAffineValueReplay
from .direct_affine_replay_core import instantiate_direct_affine_value
from .direct_affine_replay_core import replace_direct_affine_replay
from .direct_affine_replay_core import resolve_direct_affine_coordinates
from .direct_affine_replay_core import resolve_direct_affine_replay
from .direct_affine_templates import (
    _dependency_value_replay as _dependency_value_replay,
)
from .direct_affine_templates import (
    _feature_reduction_markers_are_supported as _feature_reduction_markers_are_supported,
)
from .direct_affine_templates import (
    _pointer_has_tensor_layout as _pointer_has_tensor_layout,
)
from .direct_affine_templates import (
    _template_reduction_placement_is_supported as _template_reduction_placement_is_supported,
)
from .direct_affine_templates import (
    _validate_direct_affine_templates as _validate_direct_affine_templates,
)
from .direct_affine_templates import resolve_direct_affine_templates

__all__ = [
    "DirectAffineAsyncEntryReplay",
    "DirectAffineCoordinates",
    "DirectAffineEffectReplay",
    "DirectAffineEmission",
    "DirectAffineNodeReplay",
    "DirectAffineOrdinaryAxis",
    "DirectAffineReplay",
    "DirectAffineReplayBindings",
    "DirectAffineReplayProof",
    "DirectAffineResolvedTemplates",
    "DirectAffineSharedBuffer",
    "DirectAffineStateStoreReplay",
    "DirectAffineStepReplay",
    "DirectAffineValueReplay",
    "compose_direct_affine_replacement",
    "instantiate_direct_affine_value",
    "replace_direct_affine_replay",
    "resolve_direct_affine_coordinates",
    "resolve_direct_affine_replay",
    "resolve_direct_affine_templates",
]
