# [MODULE DOC]
# Synchronous HTTP client (httpx) for the Calendar environment server plus a scenario benchmark runner and CLI.
# CalendarEnv talks to /reset, /step, /state and the /api database helpers, sending the database id, access token and
# extra context as x-* headers. ScenarioRunner drives an LLM (via openenv's create_llm_client) through tool calls
# against the env and scores runs with VerifierEngine (SQL state checks, LLM-judged response checks, tool-usage checks).
# Defines its own lightweight StepResult dataclass instead of using openenv's EnvClient.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Calendar Environment HTTP client.

This client uses the OpenEnv HTTP endpoints exposed by the calendar server:
/reset, /step, and /state. It also includes helpers for database seeding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TYPE_CHECKING, TypeVar

import httpx

if TYPE_CHECKING:
    from openenv.core.llm_client import LLMClient

try:
    from .models import CalendarAction, CalendarObservation
except ImportError:
    import sys
    from pathlib import Path

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from models import CalendarAction, CalendarObservation

ObsT = TypeVar("ObsT")


@dataclass
class StepResult(Generic[ObsT]):
    # [CLASS DOC]
    # Generic dataclass holding an observation, optional reward, done flag (default False) and optional metadata.
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
    metadata: Optional[Dict[str, Any]] = None


logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "response_output"


def _generate_database_id() -> str:
    # [FUNCTION DOC]
    # HELPER: Returns a new id "db_<epoch ms>_<9 random lowercase/digit chars>".
    timestamp = int(time.time() * 1000)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"db_{timestamp}_{suffix}"


class CalendarEnv:
    # [CLASS DOC]
    # HTTP client bound to a base URL with default database id ("default"), optional access token, extra context headers
    # and request timeout (60s). Lazily creates one httpx.Client; usable as a context manager (closes on exit).
    """HTTP client for the Calendar environment."""

    def __init__(
        self,
        base_url: str,
        database_id: str = "default",
        access_token: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_s: float = 60.0,
    ):
        # [FUNCTION DOC]
        # Stores configuration (trailing "/" stripped from base_url); no network I/O.
        self.base_url = base_url.rstrip("/")
        self.database_id = database_id
        self.access_token = access_token
        self.context = context or {}
        self.timeout_s = timeout_s
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "CalendarEnv":
        # [FUNCTION DOC]
        # Creates the HTTP client if needed and returns self.
        self._ensure_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # [FUNCTION DOC]
        # Closes the HTTP client.
        self.close()

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes and discards the HTTP client if one exists; idempotent.
        if self._client is not None:
            self._client.close()
            self._client = None

    def _ensure_client(self) -> httpx.Client:
        # [FUNCTION DOC]
        # HELPER: Returns the HTTP client, creating it with the configured timeout on first use.
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def _headers(
        self,
        database_id: Optional[str] = None,
        access_token: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        # [FUNCTION DOC]
        # HELPER: Builds request headers: JSON content type, x-database-id (override or default), x-access-token (override or
        # default) and every non-None context entry (default context merged with the override) as a lowercased header with "_"
        # replaced by "-" and an "x-" prefix added if missing; context never overrides the database/token headers.
        # Returns the header dict.
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        resolved_db = database_id if database_id is not None else self.database_id
        if resolved_db:
            headers["x-database-id"] = resolved_db

        resolved_token = access_token if access_token is not None else self.access_token
        if resolved_token:
            headers["x-access-token"] = resolved_token

        merged_context = dict(self.context)
        if context:
            merged_context.update(context)

        for key, value in merged_context.items():
            if value is None:
                continue
            header_key = str(key).strip().lower().replace("_", "-")
            if not header_key.startswith("x-"):
                header_key = f"x-{header_key}"
            if header_key in headers:
                continue
            headers[header_key] = str(value)

        return headers

    def _parse_step_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[CalendarObservation]:
        # [FUNCTION DOC]
        # HELPER: Builds a StepResult with a CalendarObservation from a server response (success default True, error message,
        # tools list, tool result, metadata preferring top-level "metadata"); reward may be None, done defaults to False.
        obs_data = payload.get("observation", {})
        observation = CalendarObservation(
            success=obs_data.get("success", True),
            error_message=obs_data.get("error_message"),
            tools_list=obs_data.get("tools_list"),
            tool_result=obs_data.get("tool_result"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def reset(
        self,
        database_id: Optional[str] = None,
        sql_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> StepResult[CalendarObservation]:
        # [FUNCTION DOC]
        # POSTs /reset with optional database_id and sql_content in the body (no body if neither) and headers. Returns the parsed
        # StepResult. Raises: httpx.HTTPStatusError on non-2xx responses, transport errors.
        payload: Dict[str, Any] = {}
        if database_id:
            payload["database_id"] = database_id
        if sql_content:
            payload["sql_content"] = sql_content

        client = self._ensure_client()
        response = client.post(
            f"{self.base_url}/reset",
            json=payload if payload else None,
            headers=self._headers(database_id=database_id, context=context),
        )
        response.raise_for_status()
        return self._parse_step_result(response.json())

    def reset_with_sql_file(
        self,
        sql_file_path: str | Path,
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> StepResult[CalendarObservation]:
        # [FUNCTION DOC]
        # Reads a SQL file (expanded and resolved path) and resets with its content. Raises: FileNotFoundError if missing.
        sql_path = Path(sql_file_path).expanduser().resolve()
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_path}")
        sql_content = sql_path.read_text(encoding="utf-8")
        return self.reset(
            database_id=database_id, sql_content=sql_content, context=context
        )

    def step(
        self,
        action: CalendarAction | Dict[str, Any],
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> StepResult[CalendarObservation]:
        # [FUNCTION DOC]
        # POSTs /step with the action (CalendarAction dumped without None fields, or a raw dict). Returns the parsed StepResult.
        # Raises: TypeError for other action types; httpx.HTTPStatusError on failure.
        if isinstance(action, CalendarAction):
            action_payload = action.model_dump(exclude_none=True)
        elif isinstance(action, dict):
            action_payload = action
        else:
            raise TypeError("action must be CalendarAction or dict")

        client = self._ensure_client()
        response = client.post(
            f"{self.base_url}/step",
            json=action_payload,
            headers=self._headers(database_id=database_id, context=context),
        )
        response.raise_for_status()
        return self._parse_step_result(response.json())

    def list_tools(
        self,
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # [FUNCTION DOC]
        # Steps with a ListToolsAction and returns the observation's tool list (empty list if None).
        result = self.step(
            CalendarAction(action_type="ListToolsAction"),
            database_id=database_id,
            context=context,
        )
        return result.observation.tools_list or []

    def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CalendarObservation:
        # [FUNCTION DOC]
        # Steps with a ToolCallAction for tool_name and arguments (default {}) and returns the observation.
        result = self.step(
            CalendarAction(
                action_type="ToolCallAction",
                tool_name=tool_name,
                arguments=arguments or {},
            ),
            database_id=database_id,
            context=context,
        )
        return result.observation

    def state(
        self,
        verify_queries: Optional[List[str]] = None,
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # GETs /state, passing each verification SQL query as a repeated "verify_queries" parameter. Returns the JSON dict
        # (including "verification_results" when queries are given). Raises httpx.HTTPStatusError on failure.
        params = (
            [("verify_queries", q) for q in verify_queries] if verify_queries else None
        )
        client = self._ensure_client()
        response = client.get(
            f"{self.base_url}/state",
            params=params,
            headers=self._headers(database_id=database_id, context=context),
        )
        response.raise_for_status()
        return response.json()

    def get_sample_sql(self) -> str:
        # [FUNCTION DOC]
        # GETs /api/sample-data and returns its sql_content. Raises: ValueError if absent; HTTP errors.
        client = self._ensure_client()
        response = client.get(f"{self.base_url}/api/sample-data")
        response.raise_for_status()
        data = response.json()
        sql_content = data.get("sql_content")
        if not sql_content:
            raise ValueError("sample-data response did not include sql_content")
        return sql_content

    def seed_database(
        self,
        sql_content: str,
        database_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        # [FUNCTION DOC]
        # POSTs /api/seed-database with the SQL, a database id (given or newly generated) and name/description defaults.
        # Returns the database id used. Raises HTTP errors. Side effects: replaces that server database's contents.
        db_id = database_id or _generate_database_id()
        payload = {
            "database_id": db_id,
            "sql_content": sql_content,
            "name": name or f"Calendar DB {db_id}",
            "description": description or "Seeded database",
        }
        client = self._ensure_client()
        response = client.post(f"{self.base_url}/api/seed-database", json=payload)
        response.raise_for_status()
        return db_id

    def seed_database_from_file(
        self,
        sql_file_path: str | Path,
        database_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        # [FUNCTION DOC]
        # Reads SQL from a file and seeds with it. Returns the database id. Raises FileNotFoundError if missing.
        sql_path = Path(sql_file_path).expanduser().resolve()
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_path}")
        sql_content = sql_path.read_text(encoding="utf-8")
        return self.seed_database(
            sql_content=sql_content,
            database_id=database_id,
            name=name,
            description=description,
        )

    def seed_database_from_sample(
        self,
        database_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        # [FUNCTION DOC]
        # Fetches the server's sample SQL and seeds with it. Returns the database id.
        sql_content = self.get_sample_sql()
        return self.seed_database(
            sql_content=sql_content,
            database_id=database_id,
            name=name,
            description=description,
        )


@dataclass
class ScenarioConfig:
    # [CLASS DOC]
    # Dataclass for one benchmark scenario: env URL (field name misspelled "gym_enviornment_url"), seed SQL file, system
    # and user prompts, LLM model/provider/key, database id, access token, context headers, execution mode ("openenv"),
    # expected/restricted tool lists, verifier definitions, number of runs, reset-between-runs flag, temperature, max
    # tokens, max iterations, seed mode ("reset" or "api") and output directory.
    # NOTE: expected_tools is loaded but never used by the runner.
    gym_enviornment_url: str
    seed_database_file: str
    system_prompt: str
    user_prompt: str
    llm_model: str
    llm_provider: str
    llm_api_key: str
    database_id: str
    access_token: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "openenv"
    expected_tools: List[str] = field(default_factory=list)
    restricted_tools: List[str] = field(default_factory=list)
    verifiers: List[Dict[str, Any]] = field(default_factory=list)
    number_of_runs: int = 1
    reset_database_between_runs: bool = True
    temperature: float = 0.0
    max_tokens: int = 4096
    max_iterations: int = 20
    seed_mode: str = "reset"
    output_dir: Path = DEFAULT_OUTPUT_DIR


class VerifierEngine:
    # [CLASS DOC]
    # Evaluates verifier definitions after a run, using the env client for SQL checks (via /state verify_queries) and an
    # LLM client as judge.
    def __init__(
        self,
        client: CalendarEnv,
        llm_client: "LLMClient",
        execution_mode: str = "openenv",
    ):
        # [FUNCTION DOC]
        # Stores the env client, LLM client and execution mode.
        self.client = client
        self.llm_client = llm_client
        self.execution_mode = execution_mode

    async def execute_verifier(
        self,
        verifier: Dict[str, Any],
        model_response: Dict[str, Any],
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Async. Dispatches by verifier["verifier_type"]: "database_state", "response_check" or "tool_execution" (each given the
        # verifier's validation_config). Returns the check's result dict (always with "passed"); unknown types return passed False
        # with an error.
        verifier_type = verifier.get("verifier_type")
        if verifier_type == "database_state":
            return await self._execute_database_state_verifier(
                verifier.get("validation_config", {}),
                database_id,
                context,
            )
        if verifier_type == "response_check":
            return await self._execute_response_check_verifier(
                verifier.get("validation_config", {}),
                model_response,
                database_id,
                context,
            )
        if verifier_type == "tool_execution":
            return await self._execute_tool_execution_verifier(
                verifier.get("validation_config", {}),
                model_response,
            )
        return {"passed": False, "error": f"Unsupported verifier type: {verifier_type}"}

    async def _execute_database_state_verifier(
        self,
        validation_config: Dict[str, Any],
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Runs the config's "query", extracts a scalar (single-column first row) or the full rows, and compares with
        # "expected_value" using "comparison_type" (default equals). Returns passed, expected, actual, comparison type, query and
        # details; or passed False with an error if the query is missing or fails.
        sql_query = validation_config.get("query")
        expected_value = validation_config.get("expected_value")
        comparison_type = validation_config.get("comparison_type", "equals")

        if not sql_query:
            return {"passed": False, "error": "No SQL query provided"}

        result = self._execute_sql_query(sql_query, database_id, context)
        if not result["success"]:
            return {
                "passed": False,
                "error": result.get("error", "SQL query failed"),
                "query": sql_query,
            }

        actual_value = self._extract_value_from_sql_result(result)
        comparison_result = self._compare_values(
            actual_value, expected_value, comparison_type
        )

        return {
            "passed": comparison_result["passed"],
            "expected": expected_value,
            "actual": actual_value,
            "comparison_type": comparison_type,
            "query": sql_query,
            "details": comparison_result.get("details"),
        }

    async def _execute_response_check_verifier(
        self,
        validation_config: Dict[str, Any],
        model_response: Dict[str, Any],
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Runs the config's "sql_query" and asks the LLM judge to score the model's final text against the result using
        # "comparison_prompt"; passes when the score >= minimum_comparison_value (default 7). Returns the judge result, or passed
        # False with an error for missing config or failed SQL.
        sql_query = validation_config.get("sql_query")
        comparison_prompt = validation_config.get("comparison_prompt")
        minimum_comparison_value = validation_config.get("minimum_comparison_value", 7)

        if not sql_query or not comparison_prompt:
            return {"passed": False, "error": "Missing sql_query or comparison_prompt"}

        sql_result = self._execute_sql_query(sql_query, database_id, context)
        if not sql_result["success"]:
            return {
                "passed": False,
                "error": sql_result.get("error", "SQL query failed"),
            }

        llm_response_text = self._extract_llm_content(model_response)
        return await self._compare_with_llm(
            sql_result,
            llm_response_text,
            comparison_prompt,
            minimum_comparison_value,
        )

    async def _execute_tool_execution_verifier(
        self, validation_config: Dict[str, Any], model_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Checks that every name in expected_tools was called and at least minimum_tool_calls (default 1) calls were
        # made. Returns passed plus expected/called/missing tools and counts.
        expected_tools = validation_config.get("expected_tools", [])
        minimum_tool_calls = validation_config.get("minimum_tool_calls", 1)

        tools_called = []
        if model_response.get("tool_calls"):
            tools_called = [tc.get("name") for tc in model_response["tool_calls"]]

        missing_tools = [tool for tool in expected_tools if tool not in tools_called]
        passed = len(missing_tools) == 0 and len(tools_called) >= minimum_tool_calls

        return {
            "passed": passed,
            "expected_tools": expected_tools,
            "tools_called": tools_called,
            "missing_tools": missing_tools,
            "minimum_tool_calls": minimum_tool_calls,
            "actual_tool_calls": len(tools_called),
        }

    def _execute_sql_query(
        self,
        query: str,
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Runs one SQL query through the env's /state endpoint. Returns {"success": True, "result": rows} or
        # {"success": False, "error": ...} when no results, the result has an error, or any exception occurs. Never raises.
        try:
            state_data = self.client.state(
                verify_queries=[query],
                database_id=database_id,
                context=context,
            )
            verification_results = state_data.get("verification_results", [])
            if not verification_results:
                return {"success": False, "error": "No verification results returned"}

            query_result = verification_results[0]
            if "error" in query_result:
                return {
                    "success": False,
                    "error": query_result.get("error", "Query failed"),
                }

            return {"success": True, "result": query_result.get("result", [])}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _extract_value_from_sql_result(self, sql_result: Dict[str, Any]) -> Any:
        # [FUNCTION DOC]
        # HELPER: Returns the single value of the first row when it is a one-column dict; otherwise the whole result list.
        result_data = sql_result.get("result", [])
        if isinstance(result_data, list) and result_data:
            first_row = result_data[0]
            if isinstance(first_row, dict) and len(first_row) == 1:
                return next(iter(first_row.values()))
        return result_data

    def _compare_values(
        self, actual: Any, expected: Any, comparison_type: str
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Compares actual vs expected with "equals", "greater_than", "less_than" or "contains" (expected substring of
        # str(actual)). Returns passed and a details string; unknown types or comparison exceptions give passed False.
        try:
            if comparison_type == "equals":
                passed = actual == expected
            elif comparison_type == "greater_than":
                passed = actual > expected
            elif comparison_type == "less_than":
                passed = actual < expected
            elif comparison_type == "contains":
                passed = expected in str(actual)
            else:
                return {
                    "passed": False,
                    "details": f"Unknown comparison type: {comparison_type}",
                }

            return {
                "passed": passed,
                "details": f"Comparison {comparison_type}: {actual} vs {expected}",
            }
        except Exception as exc:
            return {"passed": False, "details": f"Comparison error: {exc}"}

    def _extract_llm_content(self, model_response: Dict[str, Any]) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the first of "content", "text", "response" present in the model response (as str), else str of the dict.
        for key in ("content", "text", "response"):
            if key in model_response:
                return str(model_response[key])
        return str(model_response)

    async def _compare_with_llm(
        self,
        sql_result: Dict[str, Any],
        llm_response: str,
        comparison_prompt: str,
        minimum_score: int,
    ) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Async. Sends a system prompt requesting JSON {score 1-10, reasoning} and a user prompt with the DB result, the
        # assistant response and the comparison task to the LLM client (no tools). Parses the reply (extracting a ```json block
        # if present). Returns passed (score >= minimum), score and reasoning, or passed False with a parse error. LLM call
        # errors propagate.
        system_prompt = (
            "You are an evaluator comparing database results with an assistant response. "
            "Return only JSON in this format: "
            '{"score": <number 1-10>, "reasoning": "<short explanation>"}'
        )

        sql_result_str = json.dumps(sql_result.get("result", {}), indent=2)
        user_prompt = (
            f"Database result:\n{sql_result_str}\n\n"
            f"Assistant response:\n{llm_response}\n\n"
            f"Comparison task:\n{comparison_prompt}"
        )

        response = await self.llm_client.complete_with_tools(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            [],
        )

        try:
            response_text = response.content
            if "```json" in response_text:
                response_text = (
                    response_text.split("```json")[1].split("```")[0].strip()
                )
            judge_result = json.loads(response_text)
            score = int(judge_result.get("score", 0))
            return {
                "passed": score >= minimum_score,
                "score": score,
                "reasoning": judge_result.get("reasoning", ""),
            }
        except Exception as exc:
            return {"passed": False, "error": f"Judge parsing failed: {exc}"}


class ScenarioRunner:
    # [CLASS DOC]
    # Runs a scenario benchmark: prepares the database, lists (and filters) tools, runs the LLM tool-calling loop for each
    # run, runs verifiers and aggregates statistics.
    def __init__(self, config: ScenarioConfig):
        # [FUNCTION DOC]
        # Creates the env client from the config, an LLM client via create_llm_client (provider, model, key, temperature, max
        # tokens) and a VerifierEngine; initializes empty tool list and seed SQL cache.
        from openenv.core.llm_client import create_llm_client

        self.config = config
        self.client = CalendarEnv(
            base_url=config.gym_enviornment_url,
            database_id=config.database_id,
            access_token=config.access_token,
            context=config.context,
        )
        self.llm_client = create_llm_client(
            config.llm_provider,
            config.llm_model,
            config.llm_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.verifier_engine = VerifierEngine(
            self.client, self.llm_client, config.execution_mode
        )
        self.available_tools: List[Dict[str, Any]] = []
        self._seed_sql_content: Optional[str] = None

    def _prepare_database(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Loads seed SQL from the configured file (FileNotFoundError if missing). In "api" seed mode, seeds via
        # /api/seed-database (using the server sample SQL when no file) and then resets with that SQL; in "reset" mode, resets
        # with the file SQL (or none). Caches the SQL for later resets.
        sql_content = None
        seed_file = self.config.seed_database_file

        if seed_file:
            sql_path = Path(seed_file).expanduser().resolve()
            if not sql_path.exists():
                raise FileNotFoundError(f"SQL file not found: {sql_path}")
            sql_content = sql_path.read_text(encoding="utf-8")
            self._seed_sql_content = sql_content

        if self.config.seed_mode == "api":
            if sql_content is None:
                sql_content = self.client.get_sample_sql()
                self._seed_sql_content = sql_content
            self.client.seed_database(
                sql_content=sql_content,
                database_id=self.config.database_id,
            )
            self.client.reset(
                database_id=self.config.database_id, sql_content=sql_content
            )
        else:
            self.client.reset(
                database_id=self.config.database_id, sql_content=sql_content
            )

    async def initialize(self) -> None:
        # [FUNCTION DOC]
        # Async. Prepares the database and loads the tool list, removing tools named in restricted_tools.
        self._prepare_database()
        self.available_tools = self.client.list_tools()
        if self.config.restricted_tools:
            self.available_tools = [
                tool
                for tool in self.available_tools
                if tool["name"] not in self.config.restricted_tools
            ]

    async def execute_benchmark(self) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Async. Initializes, then executes number_of_runs runs (resetting the DB with the cached SQL before runs 2+ when
        # configured). Returns a dict with benchmark_config, per-run results and aggregated statistics.
        await self.initialize()
        runs: List[Dict[str, Any]] = []

        for run_number in range(1, self.config.number_of_runs + 1):
            if run_number > 1 and self.config.reset_database_between_runs:
                self.client.reset(
                    database_id=self.config.database_id,
                    sql_content=self._seed_sql_content,
                )

            run_result = await self.execute_single_run(run_number)
            runs.append(run_result)

        statistics = self._calculate_statistics(runs)
        return {
            "benchmark_config": {
                "execution_mode": self.config.execution_mode,
                "model": f"{self.config.llm_provider}/{self.config.llm_model}",
                "number_of_runs": self.config.number_of_runs,
                "user_prompt": self.config.user_prompt,
                "database_id": self.config.database_id,
                "seed_database_file": self.config.seed_database_file or "",
            },
            "runs": runs,
            "statistics": statistics,
        }

    async def execute_single_run(self, run_number: int) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # Async. Executes the task and verifiers and returns a run record: number, start time, duration ms, final response,
        # conversation flow, tools used, tool results, verification results/summary (total, passed, failed, pass rate) and
        # overall_success (all verifiers passed; True when there are none).
        start_time = datetime.now(timezone.utc)
        task_result = await self._execute_task()
        verification_results = await self._run_verifiers(task_result)

        execution_time_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
        overall_success = all(v.get("passed") for v in verification_results.values())

        total_verifiers = len(verification_results)
        passed_verifiers = sum(
            1 for v in verification_results.values() if v.get("passed", False)
        )

        return {
            "run_number": run_number,
            "started_at": start_time.isoformat(),
            "execution_time_ms": execution_time_ms,
            "model_response": task_result.get("final_response"),
            "conversation_flow": task_result.get("conversation_flow", []),
            "tools_used": task_result.get("tools_used", []),
            "tool_results": task_result.get("tool_results", []),
            "verification_results": verification_results,
            "verification_summary": {
                "total": total_verifiers,
                "passed": passed_verifiers,
                "failed": total_verifiers - passed_verifiers,
                "pass_rate": passed_verifiers / total_verifiers
                if total_verifiers > 0
                else 0.0,
            },
            "overall_success": overall_success,
        }

    async def _execute_task(self) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Async LLM loop for up to max_iterations: sends messages and available tools, records the assistant message,
        # stops when no tool calls are returned; otherwise executes each tool call against the env, appends a "tool" message with
        # the JSON tool result and records it. Returns the final assistant content, conversation flow, unique tools used (in
        # order) and all tool results. Side effects: network calls to the LLM and env.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": self.config.user_prompt},
        ]

        conversation_flow: List[Dict[str, Any]] = []
        tools_used: List[str] = []
        tool_results: List[Dict[str, Any]] = []

        for iteration in range(self.config.max_iterations):
            response = await self.llm_client.complete_with_tools(
                messages, self.available_tools
            )
            messages.append(response.to_message_dict())

            tool_calls = [
                {"id": tc.id, "name": tc.name, "args": tc.args}
                for tc in response.tool_calls
            ]
            conversation_flow.append(
                {
                    "type": "ai_message",
                    "content": response.content,
                    "tool_calls": tool_calls,
                }
            )

            if not tool_calls:
                break

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

                observation = self.client.call_tool(tool_name, arguments=tool_args)
                tool_result = {
                    "success": observation.success,
                    "result": observation.tool_result,
                    "error_message": observation.error_message,
                    "metadata": observation.metadata,
                }

                if tool_name and tool_name not in tools_used:
                    tools_used.append(tool_name)

                tool_results.append(
                    {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_result.get("result", {})),
                        "tool_call_id": tool_call.get("id", ""),
                    }
                )

                conversation_flow.append(
                    {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "result": tool_result,
                    }
                )

        # Find the last assistant message (not a tool result).
        final_response = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                final_response = msg.get("content", "")
                break
        return {
            "final_response": final_response,
            "conversation_flow": conversation_flow,
            "tools_used": tools_used,
            "tool_results": tool_results,
        }

    async def _run_verifiers(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Async. Builds a model-response dict (final content plus tool calls) and runs every configured verifier, keyed by
        # its "name" or "verifier_<n>". Returns the mapping of verifier name to result.
        verification_results: Dict[str, Any] = {}
        model_response = {
            "content": task_result.get("final_response", ""),
            "tool_calls": [
                {"name": tr["tool_name"], "args": tr["arguments"]}
                for tr in task_result.get("tool_results", [])
            ],
        }

        for i, verifier in enumerate(self.config.verifiers):
            verifier_name = verifier.get("name") or f"verifier_{i + 1}"
            verification_results[
                verifier_name
            ] = await self.verifier_engine.execute_verifier(
                verifier, model_response, self.config.database_id, self.config.context
            )

        return verification_results

    def _calculate_statistics(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Aggregates runs: totals and success rate, pass@1 (first run succeeded), verifier-level pass rate, per-verifier
        # pass counts/rates, mean execution time and per-tool usage counts (number of runs using each tool).
        successful_runs = [r for r in runs if r.get("overall_success")]
        total_runs = len(runs)
        overall_success_rate = (
            len(successful_runs) / total_runs if total_runs > 0 else 0.0
        )
        pass_at_1 = 1.0 if runs and runs[0].get("overall_success") else 0.0

        total_verifiers_count = 0
        passed_verifiers_count = 0
        verifier_pass_rates: Dict[str, Dict[str, int]] = {}

        for run in runs:
            summary = run.get("verification_summary")
            if summary:
                total_verifiers_count += summary.get("total", 0)
                passed_verifiers_count += summary.get("passed", 0)

            for verifier_name, result in run.get("verification_results", {}).items():
                if verifier_name not in verifier_pass_rates:
                    verifier_pass_rates[verifier_name] = {"passed": 0, "total": 0}
                verifier_pass_rates[verifier_name]["total"] += 1
                if result.get("passed", False):
                    verifier_pass_rates[verifier_name]["passed"] += 1

        verifier_stats = {}
        for verifier_name, counts in verifier_pass_rates.items():
            total = counts["total"]
            passed = counts["passed"]
            verifier_stats[verifier_name] = {
                "passed": passed,
                "total": total,
                "pass_rate": passed / total if total > 0 else 0.0,
            }

        verifier_level_pass_rate = (
            passed_verifiers_count / total_verifiers_count
            if total_verifiers_count > 0
            else 0.0
        )

        execution_times = [
            r.get("execution_time_ms", 0) for r in runs if "execution_time_ms" in r
        ]
        mean_time = (
            sum(execution_times) / len(execution_times) if execution_times else 0.0
        )

        tool_counts: Dict[str, int] = {}
        for run in runs:
            for tool in run.get("tools_used", []):
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        return {
            "total_runs": total_runs,
            "successful_runs": len(successful_runs),
            "overall_success_rate": overall_success_rate,
            "pass_at_1": pass_at_1,
            "verifier_level_pass_rate": verifier_level_pass_rate,
            "total_verifiers_checked": total_verifiers_count,
            "total_verifiers_passed": passed_verifiers_count,
            "individual_verifier_stats": verifier_stats,
            "mean_execution_time_ms": mean_time,
            "tool_usage": tool_counts,
        }


def _resolve_api_key(provider: str, config_key: Optional[str]) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the configured key, else the provider's environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY,
    # GOOGLE_API_KEY, else LLM_API_KEY), else LLM_API_KEY. Raises ValueError if none is found.
    if config_key:
        return config_key

    provider = provider.lower()
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    key = os.getenv(env_map.get(provider, "LLM_API_KEY")) or os.getenv("LLM_API_KEY")
    if not key:
        raise ValueError("LLM API key is required for scenario runs")
    return key


def _load_scenario_config(
    config_path: str,
    base_url_override: Optional[str],
    seed_sql_override: Optional[str],
    database_id_override: Optional[str],
    auto_db: bool,
    seed_mode_override: Optional[str],
    output_dir: Optional[str],
    access_token_override: Optional[str],
) -> ScenarioConfig:
    # [FUNCTION DOC]
    # HELPER: Loads a scenario JSON file (keys starting with "_" ignored) and applies CLI overrides to build a ScenarioConfig.
    # Base URL from override or gym_enviornment_url/gym_environment_url/base_url; seed_mode must be reset/api; system_prompt,
    # user_prompt, llm_provider and llm_model are required; only execution_mode "openenv" is allowed; auto_db generates a new
    # database id (else override/config/"default"); API key resolved via environment if absent.
    # Raises: FileNotFoundError, ValueError for missing/invalid fields, json errors.
    # KNOWN ISSUE: the relative-output-path branch is unreachable because the path is already resolved to absolute.
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Scenario config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config_data = json.load(file)

    config_data = {k: v for k, v in config_data.items() if not str(k).startswith("_")}

    base_url = (
        base_url_override
        or config_data.get("gym_enviornment_url")
        or config_data.get("gym_environment_url")
        or config_data.get("base_url")
    )
    if not base_url:
        raise ValueError("Missing gym_enviornment_url in scenario config")

    seed_database_file = (
        seed_sql_override
        if seed_sql_override is not None
        else config_data.get("seed_database_file", "")
    )
    seed_mode = seed_mode_override or config_data.get("seed_mode", "reset")
    if seed_mode not in {"reset", "api"}:
        raise ValueError(f"Invalid seed_mode: {seed_mode}")

    required_fields = ["system_prompt", "user_prompt", "llm_provider", "llm_model"]
    missing = [field for field in required_fields if not config_data.get(field)]
    if missing:
        raise ValueError(
            f"Missing required fields in scenario config: {', '.join(missing)}"
        )

    execution_mode = config_data.get("execution_mode", "openenv")
    if execution_mode != "openenv":
        raise ValueError("Only execution_mode 'openenv' is supported by this client")

    database_id = database_id_override or config_data.get("database_id")
    if auto_db:
        database_id = _generate_database_id()
    if not database_id:
        database_id = "default"

    access_token = access_token_override or config_data.get("access_token")

    llm_provider = config_data.get("llm_provider")
    llm_model = config_data.get("llm_model")
    llm_api_key = _resolve_api_key(llm_provider, config_data.get("llm_api_key"))

    output_path = (
        Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_OUTPUT_DIR
    )
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    return ScenarioConfig(
        gym_enviornment_url=base_url,
        seed_database_file=seed_database_file or "",
        system_prompt=config_data.get("system_prompt", ""),
        user_prompt=config_data.get("user_prompt", ""),
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        database_id=database_id,
        access_token=access_token,
        context=config_data.get("context", {}),
        execution_mode=execution_mode,
        expected_tools=config_data.get("expected_tools", []) or [],
        restricted_tools=config_data.get("restricted_tools", []) or [],
        verifiers=config_data.get("verifiers", []) or [],
        number_of_runs=config_data.get("number_of_runs", 1),
        reset_database_between_runs=config_data.get(
            "reset_database_between_runs", True
        ),
        temperature=config_data.get("temperature", 0.0),
        max_tokens=config_data.get("max_tokens", 4096),
        max_iterations=config_data.get("max_iterations", 20),
        seed_mode=seed_mode,
        output_dir=output_path,
    )


def _write_scenario_output(result: Dict[str, Any], output_dir: Path) -> Path:
    # [FUNCTION DOC]
    # HELPER: Creates the output directory and writes the result as indented JSON to benchmark_results_<UTC timestamp>.json.
    # Returns the file path.
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"benchmark_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, default=str)
    return output_path


def _parse_json_arg(value: str) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Parses a JSON object string; empty string gives {}. Raises ValueError for invalid JSON or non-object values.
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON value: {value}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON value must be an object")
    return parsed


def main() -> None:
    # [FUNCTION DOC]
    # CLI entry point. With --scenario, loads the config and runs the benchmark, writing results to the output dir.
    # Otherwise connects to --base-url with the database id (or a generated one with --auto-db), seeds/resets from --seed-sql
    # (via the seed API first in "api" mode) or plain reset, prints the tool count, and optionally calls --tool with JSON
    # --args and prints the observation.
    # KNOWN ISSUE: in scenario mode the --base-url and --database-id defaults are always non-empty, so they always override
    # the URL and database id from the scenario file.
    parser = argparse.ArgumentParser(description="Calendar environment HTTP client")
    parser.add_argument("--base-url", default="http://localhost:8004")
    parser.add_argument("--database-id", default="default")
    parser.add_argument("--auto-db", action="store_true")
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--context", default="{}")
    parser.add_argument("--seed-sql", default=None)
    parser.add_argument("--seed-mode", choices=["reset", "api"], default=None)
    parser.add_argument("--tool", default=None)
    parser.add_argument("--args", default="{}")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    context = _parse_json_arg(args.context)

    if args.scenario:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        config = _load_scenario_config(
            config_path=args.scenario,
            base_url_override=args.base_url,
            seed_sql_override=args.seed_sql,
            database_id_override=args.database_id,
            auto_db=args.auto_db,
            seed_mode_override=args.seed_mode,
            output_dir=args.output_dir,
            access_token_override=args.access_token,
        )

        async def _run() -> None:
            # [FUNCTION DOC]
            # HELPER: Async closure that runs the scenario benchmark and saves the results file.
            runner = ScenarioRunner(config)
            result = await runner.execute_benchmark()
            output_path = _write_scenario_output(result, config.output_dir)
            logger.info("Scenario results saved to %s", output_path)

        asyncio.run(_run())
        return

    database_id = _generate_database_id() if args.auto_db else args.database_id

    with CalendarEnv(
        base_url=args.base_url,
        database_id=database_id,
        access_token=args.access_token,
        context=context,
    ) as client:
        seed_mode = args.seed_mode or "reset"
        if args.seed_sql:
            if seed_mode == "api":
                client.seed_database_from_file(
                    sql_file_path=args.seed_sql,
                    database_id=database_id,
                )
                client.reset_with_sql_file(
                    sql_file_path=args.seed_sql,
                    database_id=database_id,
                )
            else:
                client.reset_with_sql_file(
                    sql_file_path=args.seed_sql,
                    database_id=database_id,
                )
        else:
            client.reset(database_id=database_id)

        tools = client.list_tools()
        print(f"tools: {len(tools)}")

        if args.tool:
            tool_args = _parse_json_arg(args.args)
            observation = client.call_tool(args.tool, arguments=tool_args)
            print(observation.model_dump())


if __name__ == "__main__":
    main()
