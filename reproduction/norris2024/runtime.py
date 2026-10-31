"""Compatibility loading; official RandomNoise implementation stays byte-for-byte intact."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


def load_upstream(upstream: Path):
    # TorchEEG's umbrella import loads unrelated C extensions with no Windows wheel.
    # Load only the original transform used by the paper, under its original names.
    import torcheeg
    transforms_path = Path(torcheeg.__file__).parent / 'transforms'
    if 'torcheeg.transforms' not in sys.modules:
        package = types.ModuleType('torcheeg.transforms')
        package.__path__ = [str(transforms_path)]
        sys.modules[package.__name__] = package
        torch_package = types.ModuleType('torcheeg.transforms.torch')
        torch_package.__path__ = [str(transforms_path / 'torch')]
        sys.modules[torch_package.__name__] = torch_package
        original = importlib.import_module('torcheeg.transforms.torch.random')
        package.RandomNoise = original.RandomNoise
    sys.path.insert(0, str(upstream / 'src'))
    return importlib.import_module('models')
