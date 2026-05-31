# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

import os
from typing import Optional

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .intern_encoder import InternVisionTower, InternVisionTowerS2
from .lasar_encoder import LASARVisionTower, LASARVisionTowerS2
from .radio_encoder import RADIOVisionTower
from .siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2


def _lasar_config_path(pretrained_path: str) -> Optional[str]:
    for name in ("lasar_config.pt", "veme_config.pt"):
        path = os.path.join(pretrained_path, name)
        if os.path.exists(path):
            return path
    return None


def build_vision_tower(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    ## skip vision tower instantiation
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    # Check if this is a local checkpoint path (not a HuggingFace model name)
    is_local_path = config.resume_path and os.path.exists(model_name_or_path)

    if is_local_path and "radio" not in model_name_or_path and "+" not in model_name_or_path:
        lasar_config_path = _lasar_config_path(model_name_or_path)
        if lasar_config_path is not None:
            print(f"✓ Detected LASAR dual-stream checkpoint at {model_name_or_path}")
            use_s2 = getattr(config, "s2", False)
            if use_s2:
                return LASARVisionTowerS2.from_pretrained(model_name_or_path, config)
            return LASARVisionTower.from_pretrained(model_name_or_path, config)

        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = vision_tower_arch if vision_tower_arch is not None else model_name_or_path

    use_s2 = getattr(config, "s2", False)

    # Dual encoder: Siglip + VGGT (LASAR)
    if "+" in vision_tower_name.lower():
        siglip_model_path, vggt_model_path = model_name_or_path.split("+")
        if use_s2:
            vision_tower = LASARVisionTowerS2(siglip_model_path, config, vggt_model_path)
        else:
            vision_tower = LASARVisionTower(siglip_model_path, config, vggt_model_path)
    elif "intern" in vision_tower_name.lower():
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        if use_s2:
            vision_tower = InternVisionTowerS2(model_name_or_path, config=config, drop_path_rate=drop_path_rate)
        else:
            vision_tower = InternVisionTower(model_name_or_path, config=config, drop_path_rate=drop_path_rate)
    elif "radio" in vision_tower_name:
        vision_tower = RADIOVisionTower(model_name_or_path, config)
    elif "clip" in vision_tower_name:
        if use_s2:
            vision_tower = CLIPVisionTowerS2(model_name_or_path, config)
        else:
            vision_tower = CLIPVisionTower(model_name_or_path, config)
    elif "siglip" in vision_tower_name:
        if use_s2:
            vision_tower = SiglipVisionTowerS2(model_name_or_path, config)
        else:
            vision_tower = SiglipVisionTower(model_name_or_path, config)
    else:
        raise ValueError(f"Unknown vision tower: {model_name_or_path}")

    config.mm_hidden_size = vision_tower.config.hidden_size if not use_s2 else vision_tower.hidden_size
    return vision_tower
