# [MODULE DOC]
# Client for the wildfire environment server plus a text renderer for observations.
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import WildfireAction, WildfireObservation, WildfireState


class WildfireEnv(EnvClient[WildfireAction, WildfireObservation, WildfireState]):
    # [CLASS DOC]
    # EnvClient for the wildfire environment; implements the serialisation hooks.
    """
    Client for the Wildfire Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with WildfireEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(render_grid(result.observation))
        ...
        ...     result = env.step(WildfireAction(action="water", x=5, y=5))
        ...     print(render_grid(result.observation))
    """

    def _step_payload(self, action: WildfireAction) -> dict:
        # [FUNCTION DOC]
        # Returns a mapping with the action name and x/y coordinates (None when unset).
        """Convert WildfireAction to JSON payload for step request."""
        return {"action": action.action, "x": action.x, "y": action.y}

    def _parse_result(self, payload: dict) -> StepResult[WildfireObservation]:
        # [FUNCTION DOC]
        # Builds a WildfireObservation from all keys of payload["observation"] (KeyError if missing, validation
        # error on unexpected shape) and returns a StepResult with envelope reward and done.
        # KNOWN ISSUE: reward/done are not copied onto the observation and envelope metadata is dropped.
        """Parse server response into StepResult[WildfireObservation]."""
        obs = WildfireObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> WildfireState:
        # [FUNCTION DOC]
        # Returns a WildfireState built from all keys of the payload.
        """Parse server response into WildfireState object."""
        return WildfireState(**payload)


def render_grid(obs: WildfireObservation) -> str:
    # [FUNCTION DOC]
    # Returns a text rendering of an observation: one row of emoji per grid row (ash, fuel, burning, firebreak,
    # water; "?" for unknown codes) followed by a line with step, wind, humidity, burning and burned counts.
    legend = {0: "⬛", 1: "🟩", 2: "🟥", 3: "🟫", 4: "🟦"}
    w, h = obs.width, obs.height
    g = obs.grid
    rows = []
    for y in range(h):
        rows.append("".join(legend.get(g[y * w + x], "?") for x in range(w)))
    meta = f"step={obs.step} wind={obs.wind_dir} hum={obs.humidity:.2f} burning={obs.burning_count} burned={obs.burned_count}"
    return "\n".join(rows + [meta])
