#!/usr/bin/env python3
"""Export LASAR submodule weights from a DeepSpeed or consolidated checkpoint directory."""

import argparse
import glob
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llava.model.language_model.lasar_llama import LASAR_MODULES_FILENAME, LASAR_MODULE_KEY_PREFIXES


def _is_lasar_key(name: str) -> bool:
    return any(name.startswith(prefix) or name == prefix.rstrip(".") for prefix in LASAR_MODULE_KEY_PREFIXES)


def load_deepspeed_state(ckpt_dir: str) -> dict:
    pattern = os.path.join(ckpt_dir, "global_step*", "zero_pp_rank_*_mp_rank_00_model_states.pt")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"No DeepSpeed shards under {ckpt_dir}")

    merged = {}
    for shard_path in shard_files:
        shard = torch.load(shard_path, map_location="cpu")
        module_state = shard.get("module", shard)
        for key, value in module_state.items():
            if _is_lasar_key(key) and key not in merged:
                merged[key] = value
    return merged


def main():
    parser = argparse.ArgumentParser(description="Export LASAR modules into ckpt/lasar/lasar_modules.bin")
    parser.add_argument("checkpoint_dir", help="Checkpoint directory (e.g. ckpt/exp_R02 or checkpoint-34900)")
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    lasar_dir = os.path.join(ckpt_dir, "lasar")
    os.makedirs(lasar_dir, exist_ok=True)
    out_path = os.path.join(lasar_dir, LASAR_MODULES_FILENAME)

    if os.path.isfile(out_path):
        print(f"Already exists: {out_path}")
        return

    lasar_state = load_deepspeed_state(ckpt_dir)
    if not lasar_state:
        raise RuntimeError("No LASAR tensors found in DeepSpeed shards.")

    torch.save(lasar_state, out_path)
    print(f"Exported {len(lasar_state)} tensors to {out_path}")


if __name__ == "__main__":
    main()
