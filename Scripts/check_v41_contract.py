"""Run V4.1 metadata tests without importing the package's MLX inference runtime.

Usage: python3 Scripts/check_v41_contract.py
This exercises real contract code; it does not mock MLX or execute inference.
"""
from pathlib import Path
import sys
import types
import unittest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("deepseek_v4_ssd")
    package.__path__ = [str(root / "runtime/deepseek_v4_ssd")]
    sys.modules[package.__name__] = package
    suite = unittest.defaultTestLoader.discover(
        str(root / "runtime/tests"), pattern="test_v41_checkpoint_contract.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
