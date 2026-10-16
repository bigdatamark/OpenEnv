# [MODULE DOC]
# HELPER: icon set for the Harbor UI. Loads SVG path data from the packaged
# ui_assets/icons.json and renders it as inline SVG on the server side, embeds it in
# the page once, and provides a JavaScript prelude so client components render the same icons.
"""One small icon set for the Harbor UI (24px grid, 1.75 stroke, Lucide-style paths), shared by the
HTML rendered here and the components' JavaScript, so the page never falls back to emoji or glyphs."""

from __future__ import annotations

import html
import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def paths() -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: reads and parses openenv.harbor/ui_assets/icons.json (cached after first call).
    # Returns: mapping from icon name to SVG inner markup. Raises on missing file / bad JSON.
    return json.loads(
        resources.files("openenv.harbor")
        .joinpath("ui_assets", "icons.json")
        .read_text()
    )


def icon(name: str, size: int = 16, cls: str = "") -> str:
    # [FUNCTION DOC]
    # HELPER: returns an inline SVG string (24x24 viewBox, stroke-styled, aria-hidden) of
    # the given size, with CSS class "ic" plus optional extra class. Unknown names fall
    # back to the "file" icon (KeyError if that is also missing). Name/class are not escaped.
    return (
        f'<svg class="ic{" " + cls if cls else ""}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{paths().get(name) or paths()["file"]}</svg>'
    )


def json_tag() -> str:
    # [FUNCTION DOC]
    # HELPER: returns a hidden div with id "hb-icons" whose data-icons attribute holds the
    # HTML-escaped JSON of the whole icon set, to be emitted once per page.
    """The icon set, once per page, for every component's `icon()` to read.

    An attribute rather than a `<script type="application/json">`, which Gradio warns about in a
    `gr.HTML` because it cannot tell a data block from code that will never run.
    """
    return f'<div id="hb-icons" hidden data-icons="{html.escape(json.dumps(paths()))}"></div>'


def js_prelude() -> str:
    # [FUNCTION DOC]
    # HELPER: returns JavaScript source defining esc(text) (HTML escaping) and
    # icon(name, size, cls) that lazily reads the icon set from the #hb-icons element
    # (cached on window.__hbIcons) and builds the same SVG markup as icon().
    """`esc(text)` and `icon(name, size, cls)` for a component's `js_on_load`.

    The icons are read from the page (`json_tag`) the first time one is drawn, rather than inlined
    into each of the seven components' scripts.
    """
    return (
        "const esc = (s) => String(s ?? '').replace(/[&<>\"']/g, (c) => "
        "({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', \"'\": '&#39;' }[c]));\n"
        "const icon = (n, s = 16, c = '') => {\n"
        "  let I = window.__hbIcons;\n"
        "  if (!I) { const el = document.getElementById('hb-icons'); I = el ? (window.__hbIcons = JSON.parse(el.dataset.icons)) : {}; }\n"
        '  return `<svg class="ic${c ? \' \' + c : \'\'}" width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" '
        "aria-hidden=\"true\">${I[n] || I.file || ''}</svg>`;\n"
        "};\n"
    )
