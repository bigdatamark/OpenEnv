# [MODULE DOC]
# Tests for openenv.cli.__main__.main error handling (typer app patched).
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the openenv __main__ module."""

from unittest.mock import patch

import pytest
from openenv.cli.__main__ import main
from typer.testing import CliRunner


runner = CliRunner()


def test_main_handles_keyboard_interrupt() -> None:
    # [FUNCTION DOC]
    # KeyboardInterrupt from the app exits with code 130.
    """Test that main handles KeyboardInterrupt gracefully."""
    with patch("openenv.cli.__main__.app") as mock_app:
        mock_app.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 130


def test_main_handles_generic_exception() -> None:
    # [FUNCTION DOC]
    # Any other exception exits with code 1.
    """Test that main handles generic exceptions gracefully."""
    with patch("openenv.cli.__main__.app") as mock_app:
        mock_app.side_effect = ValueError("Test error")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1


def test_main_entry_point() -> None:
    # [FUNCTION DOC]
    # main() invokes the app once.
    """Test that main() can be called as entry point."""
    # This tests the if __name__ == "__main__" block indirectly
    # by ensuring main() function works
    with patch("openenv.cli.__main__.app") as mock_app:
        main()
        mock_app.assert_called_once()
