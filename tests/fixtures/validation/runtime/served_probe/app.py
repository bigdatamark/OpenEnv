# [MODULE DOC]
# Fixture server for runtime validation tests: a tiny deterministic counter environment
# served by create_app, wrapped in an ASGI middleware that can inject wire-level faults
# selected by the VALIDATION_FAULT env var.
"""A deterministic subject, with deliberate faults confined to test assets."""

import asyncio
import json
import os
from typing import Any

import uvicorn
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ProbeAction(Action):
    # [CLASS DOC]
    # Action carrying a strict integer increment of 1 or 2.
    increment: int = Field(ge=1, le=2, strict=True)


class ProbeObservation(Observation):
    # [CLASS DOC]
    # Observation carrying a strict integer counter.
    counter: int = Field(strict=True)


class ProbeEnvironment(Environment):
    # [CLASS DOC]
    # Counter environment supporting concurrent sessions. Each step adds the increment;
    # reward is 1.0 once counter >= 2, done after two steps.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        # [FUNCTION DOC]
        # Initialises state (episode "uninitialized", step 0) and counter 0.
        super().__init__()
        self._state = State(episode_id="uninitialized", step_count=0)
        self.counter = 0

    def reset(self, seed=None, episode_id=None, **kwargs):
        # [FUNCTION DOC]
        # Zeroes the counter and step count; episode id is the given one or "probe".
        # Returns: observation with counter 0, reward 0.0, done False.
        self.counter = 0
        self._state = State(episode_id=episode_id or "probe", step_count=0)
        return ProbeObservation(counter=0, reward=0.0, done=False)

    def step(self, action, timeout_s=None, **kwargs):
        # [FUNCTION DOC]
        # Adds action.increment to the counter, increments step_count (mutates state).
        # Returns: observation with counter, reward float(counter >= 2), done step_count >= 2.
        self.counter += action.increment
        self._state.step_count += 1
        return ProbeObservation(
            counter=self.counter,
            reward=float(self.counter >= 2),
            done=self._state.step_count >= 2,
        )

    @property
    def state(self):
        # [FUNCTION DOC]
        # Property returning the current State object.
        return self._state


class WireFault:
    # [CLASS DOC]
    # ASGI middleware wrapping the app; mode selects which fault to inject into websocket traffic.
    """Corrupt real server responses after serialization, preserving raw defects."""

    def __init__(self, application, mode):
        # [FUNCTION DOC]
        # Stores the wrapped ASGI application and the fault mode string.
        self.application = application
        self.mode = mode

    async def __call__(self, scope, receive, send):
        # [FUNCTION DOC]
        # ASGI entry point: forwards to the wrapped app with receive/send wrapped so that
        # faults are injected per connection (step counter is per call).
        steps = 0

        async def fault_receive():
            # [FUNCTION DOC]
            # HELPER: Wraps receive. In hung_step/slow_step modes, on the second incoming "step"
            # message writes OPENENV_VALIDATION_STEP_BLOCKED to fd 1, then sleeps 5.2 s (slow) or
            # waits forever (hung). Returns the original message.
            nonlocal steps
            message = await receive()
            if (
                self.mode in {"hung_step", "slow_step"}
                and message["type"] == "websocket.receive"
                and message.get("text")
                and json.loads(message["text"]).get("type") == "step"
            ):
                steps += 1
                if steps == 2:
                    # Tests signal the CLI only after the collector has completed
                    # a real reset, first step and both corresponding state reads.
                    os.write(1, b"OPENENV_VALIDATION_STEP_BLOCKED\n")
                    if self.mode == "slow_step":
                        await asyncio.sleep(5.2)
                    else:
                        await asyncio.Event().wait()
            return message

        async def fault_send(message: dict[str, Any]):
            # [FUNCTION DOC]
            # HELPER: Wraps send. Rewrites outgoing JSON: observation messages get reward True
            # (bad_reward), a non-integer counter (bad_observation) or no done field (missing_done);
            # state messages get episode_id "wrong-episode" (bad_state). Then forwards.
            if message["type"] == "websocket.send" and message.get("text"):
                data = json.loads(message["text"])
                if data.get("type") == "observation":
                    payload = data["data"]
                    if self.mode == "bad_reward":
                        payload["reward"] = True
                    elif self.mode == "bad_observation":
                        payload["observation"]["counter"] = "not-an-integer"
                    elif self.mode == "missing_done":
                        payload.pop("done", None)
                elif data.get("type") == "state" and self.mode == "bad_state":
                    data["data"]["episode_id"] = "wrong-episode"
                message = {**message, "text": json.dumps(data)}
            await send(message)

        await self.application(scope, fault_receive, fault_send)


def make_app(mode="good"):
    # [FUNCTION DOC]
    # Builds the probe app (max 2 concurrent envs) wrapped in WireFault for the mode.
    # Returns: the ASGI app. Raises: RuntimeError for "startup_failure", ValueError for
    # unknown modes.
    if mode == "startup_failure":
        raise RuntimeError(
            "Deliberate startup failure from the validation test fixture"
        )
    if mode not in {
        "good",
        "bad_reward",
        "bad_observation",
        "missing_done",
        "bad_state",
        "hung_step",
        "slow_step",
    }:
        raise ValueError(f"Unknown fixture mode: {mode}")
    return WireFault(
        create_app(
            ProbeEnvironment,
            ProbeAction,
            ProbeObservation,
            env_name="validation_probe",
            max_concurrent_envs=2,
        ),
        mode,
    )


def main():
    # [FUNCTION DOC]
    # Runs uvicorn on 0.0.0.0:$PORT (default 8000) with the app for $VALIDATION_FAULT
    # (default "good"). Blocks; side effect: network server.
    uvicorn.run(
        make_app(os.environ.get("VALIDATION_FAULT", "good")),
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
