# [MODULE DOC]
# Tests for harbor_env.harness.HarborSessionFactory / HarborSession, the TRL rollout-worker adapter
# (create -> wait_for_completion -> fetch_proxy_trace -> verify) over a hosted Harbor server, using a
# FakeEnv in place of the remote client. Covers trainability filtering, failure-as-return-code,
# ungraded rewards, prompt-to-task mapping, tool-call shape conversion, pickling, and client ownership.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`HarborSession` is what lets a stock TRL rollout worker train on a hosted Harbor server.

TRL's loop-owning path calls `create()` -> `wait_for_completion()` -> `fetch_proxy_trace()` ->
`verify()`, and knows nothing else. Satisfying that contract is what makes the training script the
stock one, with nothing added to TRL. These tests pin the parts where getting it subtly wrong would
look like a working run:

  * an EVAL rollout must yield NO trainable turns, not rows of zeros
  * a failed rollout must return non-zero rather than raise, because an exception in the rollout loop
    takes down every training rank waiting on the next batch
  * an ungraded rollout must report `None`, never 0 — a crashed rollout is not a wrong answer
  * a prompt must map back to the task the server actually has
"""

from __future__ import annotations

import pytest

harness = pytest.importorskip("harbor_env.harness")
models = pytest.importorskip("openenv.harbor.models")


def turn(**kw):
    # [FUNCTION DOC]
    # HELPER: Returns a models.HarborTurn with trainable defaults (prompt/completion ids, logprobs,
    # request messages, text) overridden by keyword arguments.
    defaults = dict(
        turn=0,
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[10, 11],
        per_token_logps=[-0.5, -0.25],
        request_messages=[{"role": "user", "content": "do the thing"}],
        text="done",
        trainable=True,
        discarded=False,
    )
    defaults.update(kw)
    return models.HarborTurn(**defaults)


def result(**kw):
    # [FUNCTION DOC]
    # HELPER: Returns an ok train/tokens HarborRolloutResult with reward 1.0 and one default turn,
    # overridden by keyword arguments.
    defaults = dict(
        ok=True,
        rollout_type="train",
        capture_level="tokens",
        reward=1.0,
        turns=[turn()],
    )
    defaults.update(kw)
    return models.HarborRolloutResult(**defaults)


class FakeEnv:
    # [CLASS DOC]
    # HELPER: Fake deployed server; returns a fixed outcome (or raises a given exception) from
    # run_rollout and records each call's kwargs. Defaults to two tasks.
    """Stands in for a deployed server. Records what it was asked for."""

    def __init__(self, outcome=None, tasks=None, raises=None):
        # [FUNCTION DOC]
        # HELPER: Stores outcome, exception, task list (default two tasks) and an empty call log.
        self._outcome = outcome
        self._raises = raises
        self._tasks = tasks or [
            {"index": 0, "task_name": "t0", "instruction": "first task"},
            {"index": 1, "task_name": "t1", "instruction": "second task"},
        ]
        self.calls: list[dict] = []

    def splits(self):
        # [FUNCTION DOC]
        # HELPER: Returns one split "train-split" with the task count.
        return [{"name": "train-split", "num_tasks": len(self._tasks)}]

    def num_tasks(self, split=""):
        # [FUNCTION DOC]
        # HELPER: Returns the number of tasks.
        return len(self._tasks)

    def get_task_range(self, split, start=None, stop=None):
        # [FUNCTION DOC]
        # HELPER: Returns the slice of task dicts.
        return self._tasks[start:stop]

    def run_rollout(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records kwargs, then raises the configured exception or returns the outcome.
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        return self._outcome


def factory_with(env, **kw):
    # [FUNCTION DOC]
    # HELPER: Builds a HarborSessionFactory for a dummy URL and split "train-split" with extra kwargs,
    # then replaces its metadata client and new_client so both return the fake env.
    """A factory whose metadata client AND per-session clients are the same fake.

    Sessions get their own client in production (a shared socket cannot carry overlapping rollouts),
    so the fake has to be returned from `new_client` too or these tests assert on a client nothing
    uses.
    """
    f = harness.HarborSessionFactory("http://server:8000", split="train-split", **kw)
    f._env = env
    f.new_client = lambda: env
    return f


def test_a_train_rollout_becomes_trace_entries():
    # [FUNCTION DOC]
    # A train rollout completes with 0 and yields one trace entry with completion ids, logprobs and the
    # request messages.
    env = FakeEnv(result())
    session = factory_with(env).create([{"role": "user", "content": "first task"}])
    assert session.wait_for_completion() == 0
    trace = session.fetch_proxy_trace()
    assert len(trace) == 1
    entry = trace[0]
    assert entry["completion_token_ids"] == [10, 11]
    assert entry["per_token_logps"] == [-0.5, -0.25]
    assert entry["request"]["messages"][0]["content"] == "do the thing", (
        "request_messages is what makes a TraceEntry buildable at all"
    )


def test_factory_passes_an_explicit_training_policy_to_each_rollout():
    # [FUNCTION DOC]
    # A sampling override reaches run_rollout with defaults top_p 1.0 and top_k -1 filled in.
    env = FakeEnv(result())
    session = factory_with(env, sampling={"temperature": 0.8}).create(
        [{"role": "user", "content": "first task"}]
    )
    assert session.wait_for_completion() == 0
    policy = env.calls[0]["sampling"]
    assert policy["temperature"] == 0.8
    assert policy["top_p"] == 1.0 and policy["top_k"] == -1


def test_factory_rejects_an_invalid_training_policy_before_opening_a_client():
    # [FUNCTION DOC]
    # temperature 0 raises ValueError at factory construction.
    with pytest.raises(ValueError, match="temperature"):
        harness.HarborSessionFactory(
            "http://unused.invalid", sampling={"temperature": 0}
        )


def test_an_eval_rollout_yields_nothing_trainable():
    # [FUNCTION DOC]
    # An eval result yields an empty trace but verify still reports reward 1.0.
    """It has a reward and a readable trace; what it has no business producing is training rows."""
    env = FakeEnv(result(rollout_type="eval", capture_level="text", turns=[]))
    session = factory_with(env).create([{"role": "user", "content": "first task"}])
    session.wait_for_completion()
    assert session.fetch_proxy_trace() == []
    assert session.verify([]).env_reward == 1.0, "the reward still stands"


def test_an_eval_rollout_that_somehow_carries_turns_still_yields_nothing():
    # [FUNCTION DOC]
    # An eval result with turns still yields an empty trace.
    """Guards the tier, not the emptiness: trusting turns present on an eval result would train on
    whatever the endpoint happened to return."""
    env = FakeEnv(result(rollout_type="eval", capture_level="logprobs", turns=[turn()]))
    session = factory_with(env).create([{"role": "user", "content": "first task"}])
    session.wait_for_completion()
    assert session.fetch_proxy_trace() == []


def test_untrainable_and_discarded_turns_are_dropped():
    # [FUNCTION DOC]
    # Untrainable, discarded and empty-completion turns are excluded from the trace.
    env = FakeEnv(
        result(
            turns=[
                turn(turn=0),
                turn(turn=1, trainable=False),
                turn(turn=2, discarded=True),
                turn(turn=3, completion_token_ids=[]),
            ]
        )
    )
    session = factory_with(env).create([{"role": "user", "content": "first task"}])
    session.wait_for_completion()
    assert len(session.fetch_proxy_trace()) == 1


def test_a_server_error_is_returned_not_raised():
    # [FUNCTION DOC]
    # A raising server makes wait_for_completion return 1, an empty trace and reward None.
    """An exception here kills the rollout loop, and a dead loop hangs every training rank."""
    env = FakeEnv(raises=RuntimeError("websocket closed"))
    session = factory_with(env).create([{"role": "user", "content": "first task"}])
    assert session.wait_for_completion() == 1
    assert session.fetch_proxy_trace() == []
    assert session.verify([]).env_reward is None, "no grade is None, never 0"


def test_a_not_ok_rollout_reports_non_zero():
    # [FUNCTION DOC]
    # A not-ok result returns 1 and reward None.
    env = FakeEnv(result(ok=False, error="sandbox died", reward=None))
    session = factory_with(env).create([{"role": "user", "content": "first task"}])
    assert session.wait_for_completion() == 1
    assert session.verify([]).env_reward is None


def test_the_prompt_selects_the_right_task():
    # [FUNCTION DOC]
    # The prompt text maps to task index 1.
    env = FakeEnv(result())
    session = factory_with(env).create([{"role": "user", "content": "second task"}])
    session.wait_for_completion()
    assert env.calls[0]["task_index"] == 1


def test_an_unknown_prompt_is_a_loud_error():
    # [FUNCTION DOC]
    # An unknown prompt raises KeyError ("does not match any task").
    """Silently defaulting to task 0 would train a whole run on one task and look like convergence."""
    env = FakeEnv(result())
    with pytest.raises(KeyError, match="does not match any task"):
        factory_with(env).create(
            [{"role": "user", "content": "not a task on this server"}]
        )


def test_the_engine_and_bounds_reach_the_server():
    # [FUNCTION DOC]
    # llm_url, model and agent_timeout_sec are forwarded to run_rollout.
    env = FakeEnv(result())
    f = factory_with(
        env, llm_url="http://vllm:8000", model="Qwen/Qwen3.5-2B", agent_timeout_sec=300
    )
    f.create([{"role": "user", "content": "first task"}]).wait_for_completion()
    call = env.calls[0]
    assert call["llm_url"] == "http://vllm:8000"
    assert call["model"] == "Qwen/Qwen3.5-2B"
    assert call["agent_timeout_sec"] == 300


def test_prompt_rows_round_trip_through_create():
    # [FUNCTION DOC]
    # prompt_rows gives task indices 0 and 1 and each row's prompt maps back to its index via create.
    """The dataset a trainer builds must be the one the factory can resolve."""
    env = FakeEnv(result())
    f = factory_with(env)
    rows = f.prompt_rows()
    assert [r["task_index"] for r in rows] == [0, 1]
    for row in rows:
        assert f.create(row["prompt"])._task_index == row["task_index"]


def test_the_agent_owns_its_loop_so_no_tools_are_exposed():
    # [FUNCTION DOC]
    # Sessions list no tools and call_tool returns an error result.
    session = factory_with(FakeEnv(result())).create(
        [{"role": "user", "content": "first task"}]
    )
    assert session.list_tools() == []
    assert session.call_tool("bash", {}).error


def test_prompt_skew_measures_against_the_engines_own_ids():
    # [FUNCTION DOC]
    # measure_prompt_skew yields exact_match_frac 1.0 for an exact tokenizer and, for a drifting one,
    # 0.0 with worst common prefix 2 and length delta 1.
    """The point of the measurement: an exact re-render scores 1.0, a wrong one does not."""

    class ExactTokenizer:
        # [CLASS DOC]
        # HELPER: Tokenizer stub reproducing the captured prompt ids.
        def apply_chat_template(self, messages, **kw):
            # [FUNCTION DOC]
            # HELPER: Returns [1, 2, 3].
            return [1, 2, 3]

    class DriftingTokenizer:
        # [CLASS DOC]
        # HELPER: Tokenizer stub producing different ids.
        def apply_chat_template(self, messages, **kw):
            # [FUNCTION DOC]
            # HELPER: Returns [1, 2, 99, 100].
            return [1, 2, 99, 100]

    res = result()
    assert harness.measure_prompt_skew(res, ExactTokenizer())["exact_match_frac"] == 1.0
    drifted = harness.measure_prompt_skew(res, DriftingTokenizer())
    assert drifted["exact_match_frac"] == 0.0
    assert drifted["worst_common_prefix"] == 2
    assert drifted["length_deltas"] == [1]


# --- shapes and process boundaries: four bugs the tests above all passed through ----------------


def test_tool_calls_reach_trl_in_the_shape_it_reads():
    # [FUNCTION DOC]
    # to_trace_entries converts flat {name, arguments} tool calls into the nested OpenAI function form
    # with a non-empty id and raw JSON arguments.
    """The capture flattens tool calls to `{name, arguments}`; TRL and chat templates want the nested
    OpenAI form. Handing over the flat shape makes `has_tool_call` false for every turn, so
    `train_turn_fn=has_tool_call` — the documented default for a coding agent — throws away the whole
    rollout while the run still looks healthy."""
    res = result(
        turns=[turn(tool_calls=[{"name": "bash", "arguments": '{"cmd":"ls"}'}])]
    )
    entry = harness.to_trace_entries(res)[0]
    call = entry["response"]["choices"][0]["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "bash"
    assert call["function"]["arguments"] == '{"cmd":"ls"}', (
        "arguments must stay the raw JSON string"
    )
    assert call["id"], (
        "the schema requires an id, and it is what pairs a call with its result"
    )


def test_has_tool_call_sees_the_converted_calls():
    # [FUNCTION DOC]
    # TRL's has_tool_call recognises the converted entry. Skipped without TRL's async_grpo module.
    """The actual consequence, asserted through TRL's own predicate rather than by inspection."""
    trl = pytest.importorskip("trl.experimental.async_grpo.openenv_harness")
    res = result(turns=[turn(tool_calls=[{"name": "bash", "arguments": "{}"}])])
    entry = harness.to_trace_entries(res)[0]
    assert trl.has_tool_call(trl._entry_to_turn(entry)) is True


def test_already_nested_tool_calls_pass_through_untouched():
    # [FUNCTION DOC]
    # Already-nested tool calls are passed through unchanged.
    nested = {
        "id": "call_x",
        "type": "function",
        "function": {"name": "bash", "arguments": "{}"},
    }
    res = result(turns=[turn(tool_calls=[nested])])
    entry = harness.to_trace_entries(res)[0]
    assert entry["response"]["choices"][0]["message"]["tool_calls"] == [nested]


def test_the_factory_survives_pickling():
    # [FUNCTION DOC]
    # A factory with a bound client pickles; the revived copy drops the client but keeps the
    # instruction map and llm_url.
    """TRL spawns the rollout loop and pickles the factory into it. Building the dataset in the parent
    binds a websocket first, so without dropping it the pickle fails or the child inherits a
    connection owned by another process."""
    import pickle

    env = FakeEnv(result())
    # Built directly, not via `factory_with`: that helper patches `new_client` with a lambda, and a
    # lambda on the instance is itself unpicklable — which would test the helper, not the factory.
    f = harness.HarborSessionFactory(
        "http://server:8000", split="train-split", llm_url="http://vllm:8000"
    )
    f._env = env
    f.prompt_rows()  # binds the client, as a trainer does before spawning
    assert f._env is not None
    revived = pickle.loads(pickle.dumps(f))
    assert revived._env is None, "a live client must not cross the process boundary"
    # The mapping is plain data and must survive, or the child cannot resolve the parent's dataset.
    assert revived._by_instruction == f._by_instruction
    assert revived.llm_url == "http://vllm:8000"


def test_duplicate_instructions_do_not_silently_shadow_each_other(caplog):
    # [FUNCTION DOC]
    # Duplicate instructions log a "share an instruction" warning and the first index wins.
    """Two tasks with one instruction produced two rows that both resolved to the LAST index, so a
    group trained on a task it was never given and nothing said so."""
    env = FakeEnv(
        result(),
        tasks=[
            {"index": 0, "task_name": "a", "instruction": "same text"},
            {"index": 1, "task_name": "b", "instruction": "same text"},
        ],
    )
    f = factory_with(env)
    with caplog.at_level("WARNING"):
        f.tasks()
    assert any("share an instruction" in r.message for r in caplog.records), (
        "a collision that changes which task runs must be said out loud"
    )
    # First occurrence wins, deterministically, rather than whichever happened to be last.
    assert f.create([{"role": "user", "content": "same text"}])._task_index == 0


def test_a_zero_timeout_defers_to_the_task_file():
    # [FUNCTION DOC]
    # wait_for_completion(timeout_s=0) forwards 0, not the factory default.
    """`0` is documented as "use the task's own timeout". `or` turns it into the factory default."""
    env = FakeEnv(result())
    session = factory_with(env, agent_timeout_sec=600).create(
        [{"role": "user", "content": "first task"}]
    )
    session.wait_for_completion(timeout_s=0)
    assert env.calls[0]["agent_timeout_sec"] == 0, "0 must reach the server as 0"


def test_no_timeout_argument_uses_the_factory_default():
    # [FUNCTION DOC]
    # Without a timeout argument, the factory's agent_timeout_sec is used.
    env = FakeEnv(result())
    session = factory_with(env, agent_timeout_sec=450).create(
        [{"role": "user", "content": "first task"}]
    )
    session.wait_for_completion()
    assert env.calls[0]["agent_timeout_sec"] == 450


def test_explicit_indices_normalise_like_the_range_path():
    # [FUNCTION DOC]
    # With explicit indices, model-like tasks from get_task are normalised via model_dump so prompt rows
    # and the instruction map work.
    """`get_task` returns a model, `get_task_range` returns dicts.

    Selecting specific tasks went through `get_task`, so every consumer doing `task.get(...)` hit
    `'HarborTaskRef' object has no attribute 'get'` — but only when indices were used, so the tests
    and the default path both passed. It surfaced on the first real training launch.
    """

    class ModelLike:
        # [CLASS DOC]
        # HELPER: Pydantic-like task with model_dump and no get.
        """Stands in for a pydantic HarborTaskRef: has model_dump, has no .get."""

        def __init__(self, index, instruction):
            # [FUNCTION DOC]
            # HELPER: Stores a dict with index, name "t<index>" and the instruction.
            self._d = {
                "index": index,
                "task_name": f"t{index}",
                "instruction": instruction,
            }

        def model_dump(self):
            # [FUNCTION DOC]
            # HELPER: Returns a copy of the stored dict.
            return dict(self._d)

    class ModelEnv(FakeEnv):
        # [CLASS DOC]
        # HELPER: FakeEnv with 10 tasks served one at a time via get_task.
        def num_tasks(self, split=""):
            # [FUNCTION DOC]
            # HELPER: Returns 10.
            return 10

        def get_task(self, split, index):
            # [FUNCTION DOC]
            # HELPER: Returns a ModelLike with instruction "task <index>".
            return ModelLike(index, f"task {index}")

    f = harness.HarborSessionFactory(
        "http://server:8000", split="train-split", indices=[3, 7]
    )
    f._env = ModelEnv(result())
    rows = f.prompt_rows()
    assert [r["task_index"] for r in rows] == [3, 7]
    assert rows[0]["prompt"][0]["content"] == "task 3"
    # And the instruction map must resolve back, or create() cannot find the task.
    assert f.create(rows[1]["prompt"])._task_index == 7


def test_out_of_range_indices_are_rejected_loudly():
    # [FUNCTION DOC]
    # An index beyond num_tasks makes tasks() raise IndexError ("out of range").
    """Silently dropping them would train on fewer tasks than asked for, invisibly."""

    class ModelEnv(FakeEnv):
        # [CLASS DOC]
        # HELPER: FakeEnv reporting 5 tasks.
        def num_tasks(self, split=""):
            # [FUNCTION DOC]
            # HELPER: Returns 5.
            return 5

    f = harness.HarborSessionFactory(
        "http://server:8000", split="train-split", indices=[1, 99]
    )
    f._env = ModelEnv(result())
    with pytest.raises(IndexError, match="out of range"):
        f.tasks()


def test_each_session_gets_its_own_client():
    # [FUNCTION DOC]
    # Temporarily replaces harness.HarborEnv with a recording subclass; two sessions construct two
    # distinct owned clients, neither the factory's metadata client. Restores the class afterwards.
    """Sharing one client across concurrent rollouts fails every rollout of every step.

    The MCP transport sends and then receives on one socket with no request-id correlation, so two
    overlapping `run_rollout` calls raise `ConcurrencyError: cannot call recv while another coroutine
    is already running recv`. With `num_generations` rollouts in flight that is all of them, instantly,
    each returning unscorable — a training run that logs happily and learns nothing. It also starves
    the socket's keepalive, so the next symptom is `keepalive ping timeout`, which looks like a network
    fault rather than a sharing bug.
    """
    env = FakeEnv(result())
    f = harness.HarborSessionFactory("http://server:8000", split="train-split")
    f._env = env  # metadata client only; new_client is deliberately NOT overridden here
    f.prompt_rows()
    built = []

    class Recording(harness.HarborEnv):
        # [CLASS DOC]
        # HELPER: HarborEnv subclass that records base URLs without connecting.
        def __init__(
            self, base_url, **kw
        ):  # no super().__init__: nothing here connects
            # [FUNCTION DOC]
            # HELPER: Appends base_url to the built list; does not call super.
            built.append(base_url)

    original = harness.HarborEnv
    harness.HarborEnv = Recording
    try:
        a = f.create([{"role": "user", "content": "first task"}])
        b = f.create([{"role": "user", "content": "second task"}])
    finally:
        harness.HarborEnv = original

    assert len(built) == 2, "each session must construct its own client"
    assert a._env is not b._env, "two sessions must not share one websocket"
    assert a._owns_env and b._owns_env, "and each must own (and therefore close) it"
    assert a._env is not f._env, "nor share the factory's metadata client"


def test_a_session_closes_the_client_it_owns():
    # [FUNCTION DOC]
    # An owning session's close calls the client's close and clears its reference.
    """Left open, each rollout holds an env session until `max_concurrent_envs` is exhausted."""
    closed = []

    class Closable:
        # [CLASS DOC]
        # HELPER: Client stub recording close calls.
        def close(self):
            # [FUNCTION DOC]
            # HELPER: Appends True to the closed list.
            closed.append(True)

    session = harness.HarborSession(
        env=Closable(),
        owns_env=True,
        split="s",
        task_index=0,
        instruction="i",
        harness="mini-swe-agent",
        sandbox="e2b",
        llm_url="",
        model="",
    )
    session.close()
    assert closed == [True]
    assert session._env is None, "and must not be reused after closing"


def test_a_borrowed_client_is_left_alone():
    # [FUNCTION DOC]
    # A non-owning session's close does not close the client.
    """A session handed someone else's client must not close it out from under them."""
    closed = []

    class Closable:
        # [CLASS DOC]
        # HELPER: Client stub recording close calls.
        def close(self):
            # [FUNCTION DOC]
            # HELPER: Appends True to the closed list.
            closed.append(True)

    borrowed = Closable()
    session = harness.HarborSession(
        env=borrowed,
        owns_env=False,
        split="s",
        task_index=0,
        instruction="i",
        harness="mini-swe-agent",
        sandbox="e2b",
        llm_url="",
        model="",
    )
    session.close()
    assert closed == [], "closing a borrowed client would break its owner"


def test_an_empty_indices_list_is_refused_rather_than_meaning_everything():
    # [FUNCTION DOC]
    # indices=[] raises ValueError ("no tasks would be selected").
    """A selection that matched nothing must not silently become the whole split.

    `indices=[]` arrives from a caller whose filter found no tasks. Treating it as "no selection"
    would train on all 2238 tasks while the caller believes it picked a handful.
    """
    with pytest.raises(ValueError, match="no tasks would be selected"):
        harness.HarborSessionFactory("http://server:8000", split="s", indices=[])


def test_omitting_indices_still_means_the_whole_split():
    # [FUNCTION DOC]
    # Without indices, the factory's private indices attribute is None.
    f = harness.HarborSessionFactory("http://server:8000", split="s")
    assert f._indices is None
