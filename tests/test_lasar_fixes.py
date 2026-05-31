"""Regression tests for LASAR navigation-safe defaults."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tempfile
from types import SimpleNamespace

import torch

from llava.model.language_model.lasar_llama import (
    LASAR_MODULES_FILENAME,
    LASAR_MODULE_KEY_PREFIXES,
    LlavaLlamaModel,
)


def _is_lasar_key(name: str) -> bool:
    return any(name.startswith(prefix) or name == prefix.rstrip(".") for prefix in LASAR_MODULE_KEY_PREFIXES)


def test_lasar_key_filter():
    keys = [
        "cognitive_map_encoder.semantic_atlas.primitives",
        "belief_projection.weight",
        "belief_gate",
        "mm_projector.layers.0.weight",
    ]
    lasar = [k for k in keys if _is_lasar_key(k)]
    assert len(lasar) == 3


def test_resolve_inject_mask():
    model = LlavaLlamaModel.__new__(LlavaLlamaModel)
    model.config = SimpleNamespace(use_lasar_injection=False)
    mask = model._resolve_lasar_inject_mask(4, torch.device("cpu"), torch.tensor([True, True]))
    assert mask.sum().item() == 0

    model.config.use_lasar_injection = True
    mask2 = model._resolve_lasar_inject_mask(
        3, torch.device("cpu"), torch.tensor([False, True, False], dtype=torch.bool)
    )
    assert mask2.tolist() == [False, True, False]

    mask3 = model._resolve_lasar_inject_mask(2, torch.device("cpu"), None)
    assert mask3.sum().item() == 0


def test_save_load_lasar_modules_unit():
    class DummyLasarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cognitive_map_encoder = torch.nn.Linear(4, 4)
            self.belief_projection = torch.nn.Linear(4, 4)
            self.belief_gate = torch.nn.Parameter(torch.tensor([0.5]))
            self.episode_projection = torch.nn.Linear(4, 2)

        def _is_lasar_param(self, name):
            return _is_lasar_key(name)

        def _lasar_state_dict(self):
            return {k: v for k, v in self.state_dict().items() if self._is_lasar_param(k)}

        def save_lasar_modules(self, output_dir):
            lasar_state = self._lasar_state_dict()
            lasar_dir = os.path.join(output_dir, "lasar")
            os.makedirs(lasar_dir, exist_ok=True)
            torch.save(lasar_state, os.path.join(lasar_dir, LASAR_MODULES_FILENAME))

    model = DummyLasarModel()
    with tempfile.TemporaryDirectory() as tmp:
        model.save_lasar_modules(tmp)
        path = os.path.join(tmp, "lasar", LASAR_MODULES_FILENAME)
        assert os.path.isfile(path)
        loaded = torch.load(path, map_location="cpu")
        assert "belief_gate" in loaded


if __name__ == "__main__":
    test_lasar_key_filter()
    test_resolve_inject_mask()
    test_save_load_lasar_modules_unit()
    print("all tests passed")
