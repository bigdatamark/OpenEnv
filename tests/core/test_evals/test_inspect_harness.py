# [MODULE DOC]
# Tests for InspectAIHarness with inspect_ai replaced by mock modules injected into sys.modules:
# construction, the missing-dependency ImportError, how run maps eval_parameters to the
# inspect_ai.eval call, error handling for failed/empty logs, score extraction, and run_from_config.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for InspectAIHarness with mocked inspect_ai dependencies."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from openenv.core.evals import EvalConfig, EvalResult
from openenv.core.evals.inspect_harness import InspectAIHarness


# ---------------------------------------------------------------------------
# Helpers to build mock inspect_ai modules
# ---------------------------------------------------------------------------


def _make_mock_metric(name, value):
    # [FUNCTION DOC]
    # HELPER: Returns a MagicMock metric with the given name and value attributes.
    """Build a mock EvalMetric with name and value attributes."""
    metric = MagicMock()
    metric.name = name
    metric.value = value
    return metric


def _make_mock_eval_score(metrics):
    # [FUNCTION DOC]
    # HELPER: Returns a MagicMock score whose metrics attribute maps each name to a mock metric, built
    # from a list of (name, value) pairs.
    """Build a mock EvalScore with a metrics dict.

    Args:
        metrics: List of (name, value) tuples.

    Returns:
        Mock EvalScore with metrics as ``dict[str, EvalMetric]``.
    """
    score = MagicMock()
    score.metrics = {name: _make_mock_metric(name, val) for name, val in metrics}
    return score


def _make_mock_eval_log(*, status="success", metrics=None, results=None):
    # [FUNCTION DOC]
    # HELPER: Returns a MagicMock log with the given status and results: the explicit results if given,
    # else a results object holding one score built from metrics, else None.
    """Build a mock EvalLog object.

    Args:
        status: Log status string ("success" or "error").
        metrics: List of (name, value) tuples for a single scorer.
        results: Override results object (if None, built from metrics).
    """
    log = MagicMock()
    log.status = status

    if results is not None:
        log.results = results
    elif metrics is not None:
        mock_results = MagicMock()
        mock_results.scores = [_make_mock_eval_score(metrics)]
        log.results = mock_results
    else:
        log.results = None

    return log


def _make_mock_inspect_modules(*, eval_return=None):
    # [FUNCTION DOC]
    # HELPER: Builds a fake inspect_ai module whose eval attribute is a MagicMock returning eval_return
    # (default one successful log with accuracy 0.85). Returns (mapping of module name to module, the
    # eval mock).
    """Build a dict of mock modules that simulate inspect_ai's structure.

    Args:
        eval_return: Return value for inspect_ai.eval(). Defaults to a
            single successful log with {"accuracy": 0.85}.
    """
    if eval_return is None:
        eval_return = [_make_mock_eval_log(metrics=[("accuracy", 0.85)])]

    # Top-level
    inspect_ai_mod = ModuleType("inspect_ai")
    mock_eval = MagicMock(name="eval", return_value=eval_return)
    inspect_ai_mod.eval = mock_eval

    return {
        "inspect_ai": inspect_ai_mod,
    }, mock_eval


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestInspectAIHarnessConstruction:
    # [CLASS DOC]
    # Tests construction defaults.
    """Test instantiation and default values."""

    def test_default_construction(self):
        # [FUNCTION DOC]
        # log_dir defaults to None.
        harness = InspectAIHarness()
        assert harness.log_dir is None

    def test_custom_construction(self):
        # [FUNCTION DOC]
        # A custom log_dir is stored.
        harness = InspectAIHarness(log_dir="/tmp/logs")
        assert harness.log_dir == "/tmp/logs"

    def test_name_property(self):
        # [FUNCTION DOC]
        # name is "InspectAIHarness".
        harness = InspectAIHarness()
        assert harness.name == "InspectAIHarness"

    def test_is_eval_harness_subclass(self):
        # [FUNCTION DOC]
        # InspectAIHarness subclasses EvalHarness.
        from openenv.core.evals.base import EvalHarness

        assert issubclass(InspectAIHarness, EvalHarness)


class TestInspectAIHarnessImportGuard:
    # [CLASS DOC]
    # Tests the import guard.
    """Test that run() raises a clear ImportError when inspect-ai is missing."""

    def test_import_error_message(self):
        # [FUNCTION DOC]
        # With inspect_ai unimportable, run raises ImportError mentioning that inspect-ai is required.
        harness = InspectAIHarness()
        with patch.dict(sys.modules, {"inspect_ai": None}):
            with pytest.raises(ImportError, match="inspect-ai is required"):
                harness.run(
                    harness_version="0.3.0",
                    library_versions={},
                    dataset="mmlu",
                    eval_parameters={"model": "openai/gpt-4o"},
                )


class TestInspectAIHarnessRun:
    # [CLASS DOC]
    # Tests run with mocked inspect_ai.
    """Test the run() method with mocked inspect_ai."""

    def _run_harness(
        self, eval_parameters, dataset="mmlu", eval_return=None, **init_kwargs
    ):
        # [FUNCTION DOC]
        # HELPER: Runs a harness (constructed with init_kwargs) under mocked inspect_ai for the given
        # eval_parameters and dataset. Returns (scores, eval mock).
        """Helper to run the harness with mocked inspect_ai modules."""
        mock_modules, mock_eval = _make_mock_inspect_modules(
            eval_return=eval_return,
        )

        harness = InspectAIHarness(**init_kwargs)
        with patch.dict(sys.modules, mock_modules):
            scores = harness.run(
                harness_version="0.3.0",
                library_versions={"openai": "1.0.0"},
                dataset=dataset,
                eval_parameters=eval_parameters,
            )

        return scores, mock_eval

    def test_basic_run_returns_scores(self):
        # [FUNCTION DOC]
        # run returns the extracted scores.
        scores, _ = self._run_harness({"model": "openai/gpt-4o"})
        assert scores == {"accuracy": 0.85}

    def test_eval_called_with_correct_task_from_dataset(self):
        # [FUNCTION DOC]
        # eval is called with the dataset as the task and the model keyword.
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o"},
            dataset="hellaswag",
        )
        args, kwargs = mock_eval.call_args
        assert args[0] == "hellaswag"
        assert kwargs["model"] == "openai/gpt-4o"

    def test_task_parameter_overrides_dataset(self):
        # [FUNCTION DOC]
        # A "task" eval parameter replaces the dataset as the task.
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o", "task": "gsm8k"},
            dataset="hellaswag",
        )
        args, _ = mock_eval.call_args
        assert args[0] == "gsm8k"

    def test_missing_model_raises_value_error(self):
        # [FUNCTION DOC]
        # Missing model raises ValueError.
        harness = InspectAIHarness()
        mock_modules, _ = _make_mock_inspect_modules()
        with patch.dict(sys.modules, mock_modules):
            with pytest.raises(ValueError, match="model"):
                harness.run(
                    harness_version="0.3.0",
                    library_versions={},
                    dataset="mmlu",
                    eval_parameters={},
                )

    def test_optional_kwargs_passed_through(self):
        # [FUNCTION DOC]
        # max_samples, max_connections, temperature, max_tokens, epochs and model_base_url are forwarded.
        _, mock_eval = self._run_harness(
            {
                "model": "openai/gpt-4o",
                "max_samples": 100,
                "max_connections": 4,
                "temperature": 0.5,
                "max_tokens": 256,
                "epochs": 3,
                "model_base_url": "https://example.test/v1",
            }
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["max_samples"] == 100
        assert kwargs["max_connections"] == 4
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 256
        assert kwargs["epochs"] == 3
        assert kwargs["model_base_url"] == "https://example.test/v1"

    def test_none_optional_kwargs_omitted(self):
        # [FUNCTION DOC]
        # Unset optional keywords are not passed to eval.
        _, mock_eval = self._run_harness({"model": "openai/gpt-4o"})
        _, kwargs = mock_eval.call_args
        assert "max_samples" not in kwargs
        assert "max_connections" not in kwargs
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs
        assert "epochs" not in kwargs
        assert "model_base_url" not in kwargs

    def test_task_args_passed_through(self):
        # [FUNCTION DOC]
        # task_args are forwarded.
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o", "task_args": {"num_fewshot": 5}},
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["task_args"] == {"num_fewshot": 5}

    def test_model_args_passed_through(self):
        # [FUNCTION DOC]
        # model_args are forwarded.
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o", "model_args": {"api_key": "test"}},
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["model_args"] == {"api_key": "test"}

    def test_solver_passed_through(self):
        # [FUNCTION DOC]
        # solver is forwarded.
        solver = ["chain_of_thought", "generate"]
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o", "solver": solver},
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["solver"] == solver

    def test_scorer_passed_through(self):
        # [FUNCTION DOC]
        # scorer is forwarded.
        scorer = ["exact"]
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o", "scorer": scorer},
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["scorer"] == scorer

    def test_log_dir_passed_through(self):
        # [FUNCTION DOC]
        # The harness log_dir is forwarded.
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o"},
            log_dir="/tmp/logs",
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["log_dir"] == "/tmp/logs"

    def test_error_status_raises_runtime_error(self):
        # [FUNCTION DOC]
        # A log with error status raises RuntimeError ("failed with status").
        error_log = _make_mock_eval_log(status="error")
        harness = InspectAIHarness()
        mock_modules, _ = _make_mock_inspect_modules(eval_return=[error_log])
        with patch.dict(sys.modules, mock_modules):
            with pytest.raises(RuntimeError, match="failed with status"):
                harness.run(
                    harness_version="0.3.0",
                    library_versions={},
                    dataset="mmlu",
                    eval_parameters={"model": "openai/gpt-4o"},
                )

    def test_empty_logs_raises_runtime_error(self):
        # [FUNCTION DOC]
        # An empty log list raises RuntimeError ("returned no logs").
        harness = InspectAIHarness()
        mock_modules, _ = _make_mock_inspect_modules(eval_return=[])
        with patch.dict(sys.modules, mock_modules):
            with pytest.raises(RuntimeError, match="returned no logs"):
                harness.run(
                    harness_version="0.3.0",
                    library_versions={},
                    dataset="mmlu",
                    eval_parameters={"model": "openai/gpt-4o"},
                )


class TestInspectAIHarnessScoreExtraction:
    # [CLASS DOC]
    # Tests _extract_scores.
    """Test _extract_scores() parses EvalLog.results."""

    def test_extracts_single_metric(self):
        # [FUNCTION DOC]
        # A single metric becomes a one-entry dict.
        harness = InspectAIHarness()
        log = _make_mock_eval_log(metrics=[("accuracy", 0.92)])
        scores = harness._extract_scores(log)
        assert scores == {"accuracy": 0.92}

    def test_extracts_multiple_metrics(self):
        # [FUNCTION DOC]
        # All metrics of the score are extracted.
        harness = InspectAIHarness()
        log = _make_mock_eval_log(
            metrics=[("accuracy", 0.85), ("f1", 0.88), ("stderr", 0.02)]
        )
        scores = harness._extract_scores(log)
        assert scores == {"accuracy": 0.85, "f1": 0.88, "stderr": 0.02}

    def test_returns_empty_dict_when_results_none(self):
        # [FUNCTION DOC]
        # None results yield an empty dict.
        harness = InspectAIHarness()
        log = _make_mock_eval_log()
        assert log.results is None
        scores = harness._extract_scores(log)
        assert scores == {}

    def test_returns_empty_dict_when_no_metrics(self):
        # [FUNCTION DOC]
        # A score with no metrics yields an empty dict.
        harness = InspectAIHarness()
        # An EvalScore with an empty metrics dict
        log = _make_mock_eval_log(metrics=[])
        scores = harness._extract_scores(log)
        assert scores == {}


class TestInspectAIHarnessIntegration:
    # [CLASS DOC]
    # Tests run_from_config.
    """Test run_from_config produces correct EvalResult."""

    def test_run_from_config_returns_eval_result(self):
        # [FUNCTION DOC]
        # run_from_config returns an EvalResult with the same config and extracted scores.
        eval_return = [
            _make_mock_eval_log(metrics=[("accuracy", 0.85), ("stderr", 0.02)])
        ]
        mock_modules, _ = _make_mock_inspect_modules(eval_return=eval_return)

        harness = InspectAIHarness()
        config = EvalConfig(
            harness_name="InspectAIHarness",
            harness_version="0.3.0",
            library_versions={"openai": "1.0.0"},
            dataset="mmlu",
            eval_parameters={"model": "openai/gpt-4o"},
        )

        with patch.dict(sys.modules, mock_modules):
            result = harness.run_from_config(config)

        assert isinstance(result, EvalResult)
        assert result.config is config
        assert result.scores["accuracy"] == 0.85
        assert result.scores["stderr"] == 0.02
