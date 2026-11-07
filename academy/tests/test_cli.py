"""Smoke-level tests for the CLI: splits command uses the disk cache and
exits cleanly."""

import unittest
from unittest.mock import patch

from academy.cli.main import main


class TestCLI(unittest.TestCase):
    def test_splits_command(self):
        with patch("sys.argv", ["academy", "splits"]):
            code = main()
        self.assertEqual(code, 0)

    def test_help_exits_1(self):
        with patch("sys.argv", ["academy"]):
            code = main()
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
