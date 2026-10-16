# [MODULE DOC]
# HELPER: pure HTML-string builders for the tau2-bench Gradio UI: task explorer list, task detail page,
# conversation timeline, scoring result, run list and run comparison. All task/model-derived values are
# HTML-escaped; model messages are rendered as CommonMark with raw HTML disabled and images off.
# Defines COMPONENTS, a mapping from reward-basis name to a human description.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HTML for the task explorer, the task page and the conversations.

The values come from τ²-bench's task files and from models, so every one is escaped.
"""

import json
import re
from html import escape

from markdown_it import MarkdownIt

COMPONENTS = {
    "DB": "The database at the end matches the expected one",
    "COMMUNICATE": "The agent told the customer what it had to",
    "NL_ASSERTION": "Behaviour checks judged by an LLM",
    "ENV_ASSERTION": "Checks on the final state of the account and the phone",
    "ACTION": "The agent made the expected tool calls",
}


def points(text: str) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: splits loose instruction text into display points: one per non-empty line, and lines longer
    # than 160 chars are further split into sentences (at . ! ? followed by whitespace and a capital).
    # Returns a list of strings (empty for empty/None text).
    """τ²-bench writes instructions as loose paragraphs; one point per line or sentence reads better."""
    out = []
    for line in re.split(r"\n+", text or ""):
        line = line.strip()
        if len(line) > 160:
            out += [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if s]
        elif line:
            out.append(line)
    return out


def title(task, limit: int = 100) -> str:
    # [FUNCTION DOC]
    # HELPER: headline for a task: first sentence of the first point of the user's reason_for_call, with a
    # leading "first," / "you want to" / "you would like to" removed, first letter capitalised, truncated to
    # limit chars with an ellipsis. Raises IndexError if reason_for_call yields no points.
    """The first sentence of why the customer calls, as a headline."""
    first = points(task.user_scenario.instructions.reason_for_call)[0]
    first = re.split(r"(?<=[.!?])\s", first)[0]
    first = re.sub(r"^(first,?|you (want|would like) to)\s+", "", first, flags=re.I)
    first = first[:1].upper() + first[1:]
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def writes(task, tool_types: dict[str, str]) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns True if any expected action of the task names a tool whose type in tool_types is "write".
    return any(
        tool_types.get(a.name) == "write"
        for a in task.evaluation_criteria.actions or []
    )


def chips(task, tool_types: dict[str, str]) -> str:
    # [FUNCTION DOC]
    # HELPER: returns HTML chips for the task: one per reward-basis value, the count of expected calls, and
    # "changes the database" or "read-only".
    ec = task.evaluation_criteria
    out = [f'<span class="t2-chip">{escape(b.value)}</span>' for b in ec.reward_basis]
    n = len(ec.actions or [])
    out.append(
        f'<span class="t2-chip t2-muted">{n} expected call{"" if n == 1 else "s"}</span>'
    )
    if writes(task, tool_types):
        out.append('<span class="t2-chip t2-warn">changes the database</span>')
    else:
        out.append('<span class="t2-chip t2-muted">read-only</span>')
    return "".join(out)


def explorer_html(tasks: list, tool_types: dict[str, str]) -> str:
    # [FUNCTION DOC]
    # Returns the task explorer HTML: a search box, write/read-only filter buttons, a task count and one card
    # per task (first marked active) carrying data attributes for id, index, writes flag and lowercased
    # search text (id, reason for call, title).
    cards = []
    for i, task in enumerate(tasks):
        search = " ".join(
            [task.id, task.user_scenario.instructions.reason_for_call, title(task)]
        ).lower()
        cards.append(
            f'<li class="t2-tcard{" active" if i == 0 else ""}" data-task="{escape(task.id)}" data-index="{i}" '
            f'data-writes="{str(writes(task, tool_types)).lower()}" data-search="{escape(search)}">'
            f'<div class="t2-tcard-id">#{escape(task.id)}</div>'
            f'<div class="t2-tcard-title">{escape(title(task))}</div>'
            f'<div class="t2-chips">{chips(task, tool_types)}</div></li>'
        )
    return (
        '<div class="t2-explorer">'
        '<input class="t2-q" type="search" placeholder="Search tasks…" aria-label="Search tasks">'
        '<div class="t2-filters" role="group">'
        '<button class="t2-filter active" data-filter="all">All</button>'
        '<button class="t2-filter" data-filter="true">Changes the database</button>'
        '<button class="t2-filter" data-filter="false">Read-only</button></div>'
        f'<div class="t2-count">{len(tasks)} tasks</div>'
        f'<ul class="t2-tlist">{"".join(cards)}</ul></div>'
    )


def bullet_list(items: list[str]) -> str:
    # [FUNCTION DOC]
    # HELPER: returns an HTML unordered list with each item escaped.
    return "<ul>" + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"


def task_html(task, tool_types: dict[str, str], domain: str) -> str:
    # [FUNCTION DOC]
    # Returns the hidden task page HTML: breadcrumb (domain, id), title, optional purpose, chips, and sections
    # for what the customer wants, known/unknown info (if any), behaviour rules (collapsible, if any) and
    # scoring (reward components with descriptions, expected tool calls, NL assertions labelled as judged or
    # not part of the reward, and info the agent must communicate).
    ins = task.user_scenario.instructions
    ec = task.evaluation_criteria
    basis = [b.value for b in ec.reward_basis]
    purpose = (
        task.description.purpose
        if task.description and task.description.purpose
        else ""
    )

    scored = "".join(
        f'<li><span class="t2-chip">{escape(b)}</span> {escape(COMPONENTS.get(b, ""))}</li>'
        for b in basis
    )
    calls = "".join(
        f"<li><code>{escape(a.name)}({escape(json.dumps(a.arguments))})</code></li>"
        for a in ec.actions or []
    )
    sections = [
        f'<section class="t2-box"><h3>What the customer wants</h3>{bullet_list(points(ins.reason_for_call))}</section>'
    ]
    if ins.known_info or ins.unknown_info:
        known = (
            f"<h4>Knows</h4>{bullet_list(points(ins.known_info))}"
            if ins.known_info
            else ""
        )
        unknown = (
            f"<h4>Doesn't know</h4>{bullet_list(points(ins.unknown_info))}"
            if ins.unknown_info
            else ""
        )
        sections.append(
            f'<section class="t2-box"><h3>What they can tell the agent</h3>{known}{unknown}</section>'
        )
    if ins.task_instructions:
        rules = points(ins.task_instructions)
        sections.append(
            f'<details class="t2-box"><summary><h3>How they behave <span class="t2-muted">· {len(rules)} rules</span></h3></summary>'
            f"{bullet_list(rules)}</details>"
        )
    score = f'<h3>How it is scored</h3><ul class="t2-plain">{scored}</ul>'
    if calls:
        score += f'<h4>Expected tool calls</h4><ol class="t2-calls">{calls}</ol>'
    if ec.nl_assertions:
        label = (
            "Judged by an LLM"
            if "NL_ASSERTION" in basis
            else "Other checks, not part of the reward"
        )
        score += f"<h4>{label}</h4>{bullet_list(ec.nl_assertions)}"
    if ec.communicate_info:
        score += f"<h4>Must tell the customer</h4>{bullet_list(ec.communicate_info)}"
    sections.append(f'<section class="t2-box">{score}</section>')

    return (
        '<div class="t2-task">'
        f'<div class="t2-crumb">{escape(domain)} · task #{escape(task.id)} · hidden from the agent</div>'
        f"<h2>{escape(title(task))}</h2>"
        + (f'<p class="t2-purpose">{escape(purpose)}</p>' if purpose else "")
        + f'<div class="t2-chips">{chips(task, tool_types)}</div>'
        + "".join(sections)
        + "</div>"
    )


# The conversation ------------------------------------------------------------

# Model replies are markdown. Raw HTML is escaped and images are not loaded.
_markdown = MarkdownIt("commonmark", {"html": False}).disable("image")


def summary_html(task, tool_types: dict[str, str]) -> str:
    # [FUNCTION DOC]
    # Returns a compact task summary card (id, title, chips) for side panels.
    """The task in two lines, for the side cards of the Play and Run tabs."""
    return (
        f'<div class="t2-summary"><div class="t2-crumb">task #{escape(task.id)}</div>'
        f'<div class="t2-summary-title">{escape(title(task))}</div>'
        f'<div class="t2-chips">{chips(task, tool_types)}</div></div>'
    )


def timeline_html(events: list[dict], placeholder: str) -> str:
    # [FUNCTION DOC]
    # Returns conversation HTML from a list of event mappings. Empty list gives a placeholder div (placeholder
    # inserted unescaped). "tool" events become collapsible rows with name, JSON arguments and result
    # (pretty-printed if JSON; marked as error if it starts with "Error"; truncated to 6000 chars); other
    # events become customer/agent bubbles rendered from markdown.
    """The conversation: customer and agent messages, with tool calls as collapsible rows."""
    if not events:
        return f'<div class="t2-timeline t2-empty">{placeholder}</div>'
    rows = []
    for event in events:
        if event["kind"] == "tool":
            result = event["result"]
            try:
                result = json.dumps(json.loads(result), indent=2)
            except ValueError:  # not JSON, so it is shown as it came
                pass
            failed = result.startswith("Error")
            rows.append(
                f'<details class="t2-toolrow{" t2-err" if failed else ""}"><summary>'
                f'<span class="t2-toolname">{escape(event["name"])}</span>'
                f'<span class="t2-args">{escape(json.dumps(event["arguments"]))}</span>'
                f'<span class="t2-toolstate">{"error" if failed else "result"}</span></summary>'
                f"<pre>{escape(result[:6000])}</pre></details>"
            )
        else:
            who = "Customer" if event["kind"] == "customer" else "Agent"
            rows.append(
                f'<div class="t2-msg t2-{event["kind"]}"><div class="t2-who">{who}</div>'
                f'<div class="t2-bubble">{_markdown.render(event["text"])}</div></div>'
            )
    return '<div class="t2-timeline">' + "".join(rows) + "</div>"


def status_html(stats: str, running: bool) -> str:
    # [FUNCTION DOC]
    # Returns a status line with escaped stats text and a spinner when running.
    spinner = '<span class="t2-spinner"></span>' if running else ""
    return f'<div class="t2-status">{spinner}{escape(stats)}</div>'


def result_html(reward: float, info: dict, stats: str) -> str:
    # [FUNCTION DOC]
    # Returns the scoring panel: Solved if reward >= 1.0, the reward, stats, a check/cross row per entry of
    # info["reward_breakdown"] (required key; KeyError otherwise), rows for NL assertion and communicate
    # checks, and an expected tool calls list from action_checks when present.
    solved = reward >= 1.0
    rows = []
    for name, value in info["reward_breakdown"].items():
        mark = "✓" if value >= 1.0 else "✗"
        rows.append(
            f'<li class="{"t2-ok" if value >= 1.0 else "t2-no"}"><span class="t2-mark">{mark}</span>'
            f'<span class="t2-chip">{escape(name)}</span> {escape(COMPONENTS.get(name, ""))}</li>'
        )
    for check in info.get("nl_assertions") or []:
        met = check.get("met")
        rows.append(
            f'<li class="{"t2-ok" if met else "t2-no"} t2-sub"><span class="t2-mark">{"✓" if met else "✗"}</span>'
            f'{escape(str(check.get("nl_assertion")))}<div class="t2-muted">{escape(str(check.get("justification", "")))}</div></li>'
        )
    for check in info.get("communicate_checks") or []:
        met = check.get("met")
        rows.append(
            f'<li class="{"t2-ok" if met else "t2-no"} t2-sub"><span class="t2-mark">{"✓" if met else "✗"}</span>'
            f"Told the customer: {escape(str(check.get('info')))}</li>"
        )
    calls = ""
    if info.get("action_checks"):
        items = "".join(
            f'<li class="{"t2-ok" if c.get("action_match") else "t2-no"}"><span class="t2-mark">{"✓" if c.get("action_match") else "✗"}</span>'
            f"<code>{escape(c['action']['name'])}({escape(json.dumps(c['action']['arguments']))})</code></li>"
            for c in info["action_checks"]
        )
        calls = f'<h4>Expected tool calls</h4><ul class="t2-checks">{items}</ul>'
    return (
        f'<div class="t2-result {"t2-solved" if solved else "t2-failed"}">'
        f'<div class="t2-verdict">{"Solved" if solved else "Not solved"}<span>reward {reward:.2f}</span></div>'
        f'<div class="t2-muted t2-small">{escape(stats)}</div>'
        f'<ul class="t2-checks">{"".join(rows)}</ul>{calls}</div>'
    )


# Runs --------------------------------------------------------------------------


def run_label(run: dict) -> str:
    # [FUNCTION DOC]
    # Returns a one-line label for a run mapping: id, model short name, domain and task id, solved/not solved.
    verdict = "solved" if run["reward"] >= 1.0 else "not solved"
    return f"{run['id']} · {run['model'].split('/')[-1]} · {run['domain']} #{run['task_id']} · {verdict}"


def runs_html(runs: list[dict]) -> str:
    # [FUNCTION DOC]
    # Returns a list of run cards (newest first) with solved status/reward, model, domain/task and stats, or an
    # empty-state message if there are no runs.
    if not runs:
        return (
            '<div class="t2-timeline t2-empty">No runs yet in this session. Run a model on a '
            "task and it shows up here.</div>"
        )
    cards = []
    for i, run in enumerate(reversed(runs)):
        solved = run["reward"] >= 1.0
        cards.append(
            f'<li class="t2-tcard" data-task="{escape(run["id"])}" data-index="{i}" data-writes="all" data-search="">'
            f'<div class="t2-tcard-id">{escape(run["id"])} · {escape(run["finished"])}</div>'
            f'<div class="t2-tcard-title">{escape(run["title"])}</div>'
            '<div class="t2-chips">'
            f'<span class="t2-chip {"t2-okchip" if solved else "t2-warn"}">{"solved" if solved else "not solved"} · {run["reward"]:.2f}</span>'
            f'<span class="t2-chip t2-muted">{escape(run["model"].split("/")[-1])}</span>'
            f'<span class="t2-chip t2-muted">{escape(run["domain"])} #{escape(run["task_id"])}</span>'
            f'<span class="t2-chip t2-muted">{escape(run["stats"])}</span></div></li>'
        )
    return f'<div class="t2-explorer"><ul class="t2-tlist">{"".join(cards)}</ul></div>'


def compare_html(runs: list[dict]) -> str:
    # [FUNCTION DOC]
    # Returns side-by-side columns (one per run) with result panel and timeline, or a prompt to select runs if
    # fewer than two are given.
    if len(runs) < 2:
        return '<div class="t2-timeline t2-empty">Tick two to four runs above to compare them.</div>'
    columns = "".join(
        f'<div class="t2-col"><div class="t2-summary-title">{escape(run["model"].split("/")[-1])}</div>'
        f'<div class="t2-crumb">{escape(run["domain"])} · task #{escape(run["task_id"])} · {escape(run["id"])}</div>'
        f"{result_html(run['reward'], run['reward_info'], run['stats'])}"
        f"{timeline_html(run['events'], '')}</div>"
        for run in runs
    )
    return f'<div class="t2-compare" style="grid-template-columns: repeat({len(runs)}, minmax(0, 1fr))">{columns}</div>'
