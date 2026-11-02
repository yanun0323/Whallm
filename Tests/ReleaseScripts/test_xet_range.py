"""Integrity checks for out-of-order Xet range assembly."""

from io import BytesIO
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


helper = Path(__file__).resolve().parents[2] / "runtime/deepseek_v4_ssd/model_support/xet_range.py"
spec = spec_from_file_location("xet_range", helper)
module = module_from_spec(spec)
spec.loader.exec_module(module)
write_unordered_range = module.write_unordered_range


class XetRangeAssemblyTests(unittest.TestCase):
    def test_unordered_chunks_reconstruct_the_requested_range(self):
        output = BytesIO()
        self.assertEqual(write_unordered_range([(4, b"efgh"), (0, b"abcd")], output, 8), 8)
        self.assertEqual(output.getvalue(), b"abcdefgh")

    def test_unordered_chunks_reject_gaps_overlaps_and_overruns(self):
        for chunks in (
            [(0, b"ab"), (3, b"d")],
            [(0, b"abc"), (2, b"cd")],
            [(0, b"abcde")],
        ):
            with self.subTest(chunks=chunks):
                with self.assertRaises(ValueError):
                    write_unordered_range(chunks, BytesIO(), 4)


if __name__ == "__main__":
    unittest.main()
