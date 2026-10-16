# [MODULE DOC]
# In-memory notebook cell history for one episode, used to render the UI and to
# produce the text summary returned by the get_notebook_state tool.
"""
Notebook state tracker.

Maintains an ordered list of cells and their outputs, mirroring what a
Jupyter notebook would show. Used both for the Gradio UI rendering and
for producing the ``get_notebook_state`` summary returned to the LLM.
"""

from typing import Any, Dict, List
from uuid import uuid4

from .e2b_sandbox import CellResult


class NotebookTracker:
    # [CLASS DOC]
    # Holds an ordered list of cell dicts (keys: cell_id, cell_type, code, output,
    # error, execution_count, has_image, images, success) and a running execution
    # counter.
    """
    In-memory cell history for one episode.

    Cells are appended on every ``add_*_cell`` call. ``update_last_code_cell``
    replaces the most recent cell, implementing the edit-and-re-execute pattern
    used by agents to fix errors without cluttering the notebook with failed
    attempts.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Starts with no cells and counter 0.
        self.cells: List[Dict[str, Any]] = []
        self._exec_count: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Public mutation API
    # ──────────────────────────────────────────────────────────────────────────

    def add_code_cell(self, code: str, result: CellResult) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Increments the counter, appends a new "code" cell built from the code and
        # result, and returns the cell dict.
        """Append a new code cell with its execution result."""
        self._exec_count += 1
        cell = self._make_cell("code", code, result)
        self.cells.append(cell)
        return cell

    def update_last_code_cell(self, code: str, result: CellResult) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Removes the last cell (of any type) if present and decrements the counter
        # (not below 0), then appends the new code cell so it reuses the execution
        # number. Returns the new cell dict.
        """
        Replace the last cell with a new code + result pair.

        This implements the ``edit_and_execute_current_cell`` semantics —
        the agent fixes the last cell rather than adding a new failed one.
        """
        if self.cells:
            self.cells.pop()
            # Reuse the same exec count (it's a replacement, not a new cell)
            self._exec_count = max(0, self._exec_count - 1)
        return self.add_code_cell(code, result)

    def add_shell_cell(self, command: str, result: CellResult) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Increments the counter and appends a "shell" cell whose code is "$ <command>".
        # Returns the cell dict.
        """Append a shell command cell."""
        self._exec_count += 1
        cell = self._make_cell("shell", f"$ {command}", result)
        self.cells.append(cell)
        return cell

    def reset(self) -> None:
        # [FUNCTION DOC]
        # Clears all cells and resets the counter.
        """Clear all cells — called on episode reset."""
        self.cells = []
        self._exec_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Read API
    # ──────────────────────────────────────────────────────────────────────────

    def get_state_summary(
        self, max_cells: int = 10, include_images: bool = False
    ) -> str:
        # [FUNCTION DOC]
        # Returns "No cells executed yet." if empty; otherwise a newline-joined text of
        # the last max_cells cells: a header with the execution count and the first 120
        # code characters (newlines shown as an arrow), then the first 300 characters of
        # output and of error if present, and an image count line; with include_images
        # each image is appended as a PNG data URL.
        """
        Compact human-readable summary of recent cells for LLM context.

        Args:
            max_cells: Number of recent cells to include.
            include_images: If True, append base64 PNG data for multimodal
                models. If False (default), only note that images were
                generated — suitable for text-only models.

        Returns the last ``max_cells`` cells with truncated code + output.
        """
        if not self.cells:
            return "No cells executed yet."
        lines: List[str] = []
        for cell in self.cells[-max_cells:]:
            ec = cell["execution_count"]
            code_preview = cell["code"][:120].replace("\n", " ↵ ")
            lines.append(f"[Cell {ec}] {code_preview}")
            if cell["output"]:
                lines.append(f"  → {cell['output'][:300]}")
            if cell["error"]:
                lines.append(f"  ✗ {cell['error'][:300]}")
            if cell.get("images"):
                n = len(cell["images"])
                lines.append(f"  🖼 {n} image(s) generated")
                if include_images:
                    for i, b64 in enumerate(cell["images"]):
                        lines.append(f"  [image {i + 1}] data:image/png;base64,{b64}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _make_cell(
        self, cell_type: str, code: str, result: CellResult
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Builds a cell dict with a new UUID, the given type and code, formatted
        # output, the result's error, current counter, image flag/list and success.
        return {
            "cell_id": str(uuid4()),
            "cell_type": cell_type,
            "code": code,
            "output": self._format_output(result),
            "error": result.error,
            "execution_count": self._exec_count,
            "has_image": len(result.images) > 0,
            "images": result.images,
            "success": result.success,
        }

    @staticmethod
    def _format_output(result: CellResult) -> str:
        # [FUNCTION DOC]
        # HELPER: Static. Joins stripped stdout, the text results and an
        # "[N image(s) generated]" note with newlines.
        parts: List[str] = []
        if result.stdout:
            parts.append(result.stdout.strip())
        if result.text_results:
            parts.extend(result.text_results)
        if result.images:
            parts.append(f"[{len(result.images)} image(s) generated]")
        return "\n".join(parts)
