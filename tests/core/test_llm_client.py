# [MODULE DOC]
# Tests for openenv.core.llm_client: the LLMClient abstract base (base_url construction and
# endpoint validation), OpenAIClient and AnthropicClient construction and request mapping (SDKs
# mocked or driven through an httpx mock transport), LLMResponse, the create_llm_client factory,
# and helpers converting MCP tool schemas and OpenAI-format messages to provider formats.

# SPDX-License-Identifier: BSD-3-Clause

"""Tests for LLMClient abstraction, OpenAIClient, AnthropicClient, and helpers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncOpenAI
from openenv.core.llm_client import (
    _clean_mcp_schema,
    _mcp_tools_to_anthropic,
    _mcp_tools_to_openai,
    _openai_msgs_to_anthropic,
    AnthropicClient,
    create_llm_client,
    LLMClient,
    LLMResponse,
    OpenAIClient,
    ToolCall,
)


class TestLLMClientABC:
    # [CLASS DOC]
    # Tests the LLMClient base class.
    """Test the abstract base class."""

    def test_cannot_instantiate_directly(self):
        # [FUNCTION DOC]
        # LLMClient is abstract.
        """LLMClient is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            LLMClient("http://localhost", 8000)

    def test_concrete_subclass(self):
        # [FUNCTION DOC]
        # A subclass implementing complete stores endpoint and port.
        """A concrete subclass can be instantiated."""

        class StubClient(LLMClient):
            # [CLASS DOC]
            # HELPER: Minimal concrete LLMClient for this test.
            async def complete(self, prompt: str, **kwargs) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns "stub".
                return "stub"

        client = StubClient("http://localhost", 8000)
        assert client.endpoint == "http://localhost"
        assert client.port == 8000

    def test_base_url_property(self):
        # [FUNCTION DOC]
        # base_url joins endpoint and port.
        """base_url combines endpoint and port."""

        class StubClient(LLMClient):
            # [CLASS DOC]
            # HELPER: Minimal concrete LLMClient for this test.
            async def complete(self, prompt: str, **kwargs) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns "stub".
                return "stub"

        client = StubClient("http://localhost", 8000)
        assert client.base_url == "http://localhost:8000"

    def test_base_url_custom_endpoint(self):
        # [FUNCTION DOC]
        # base_url works for an https endpoint.
        """base_url works with custom endpoints."""

        class StubClient(LLMClient):
            # [CLASS DOC]
            # HELPER: Minimal concrete LLMClient for this test.
            async def complete(self, prompt: str, **kwargs) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns "stub".
                return "stub"

        client = StubClient("https://api.example.com", 443)
        assert client.base_url == "https://api.example.com:443"

    @pytest.mark.parametrize(
        ("endpoint", "port", "expected"),
        [
            ("http://localhost:8000", None, "http://localhost:8000"),
            ("http://localhost:8000", 8000, "http://localhost:8000"),
            ("http://localhost:8000/", 8000, "http://localhost:8000"),
            ("http://proxy/litellm", 4000, "http://proxy:4000/litellm"),
            ("http://[::1]", 8000, "http://[::1]:8000"),
            (
                "https://gw.example.com/openai/v1",
                None,
                "https://gw.example.com/openai/v1",
            ),
            ("http://localhost", None, "http://localhost"),
        ],
    )
    def test_base_url_endpoint_forms(self, endpoint, port, expected):
        # [FUNCTION DOC]
        # Parametrized: the port is appended only when the URL has none, trailing slashes are dropped,
        # paths and IPv6 hosts are preserved.
        """Port is appended only when the URL names none; a path survives."""

        class StubClient(LLMClient):
            # [CLASS DOC]
            # HELPER: Minimal concrete LLMClient for this test.
            async def complete(self, prompt: str, **kwargs) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns "stub".
                return "stub"

        assert StubClient(endpoint, port).base_url == expected

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://localhost:8000:8000",
            "http://localhost:",
            "localhost:8000",
            "http://[::1",
            "ftp://localhost:8000",
            "http:///v1",
            "http://user:token@localhost:8000",
            "http://localhost:8000/v1?api-version=1",
            "http://localhost:8000#frag",
            "http://localhost:0",
            "http://localhost:99999",
        ],
    )
    def test_base_url_malformed_endpoint_raises(self, endpoint):
        # [FUNCTION DOC]
        # Parametrized malformed endpoints (double port, no scheme, bad scheme, credentials, query,
        # fragment, invalid port) raise ValueError "Invalid endpoint URL".
        """Invalid endpoints fail with a clear error instead of a bad URL."""

        class StubClient(LLMClient):
            # [CLASS DOC]
            # HELPER: Minimal concrete LLMClient for this test.
            async def complete(self, prompt: str, **kwargs) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns "stub".
                return "stub"

        with pytest.raises(ValueError, match="Invalid endpoint URL"):
            StubClient(endpoint, 8000).base_url

    @pytest.mark.parametrize("port", [0, -1, 65536, 99999])
    def test_base_url_port_out_of_range_raises(self, port):
        # [FUNCTION DOC]
        # Out-of-range port arguments raise ValueError.
        """The appended port is range-checked like a port inside the URL."""

        class StubClient(LLMClient):
            # [CLASS DOC]
            # HELPER: Minimal concrete LLMClient for this test.
            async def complete(self, prompt: str, **kwargs) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns "stub".
                return "stub"

        with pytest.raises(ValueError, match="out of range 1-65535"):
            StubClient("http://localhost", port).base_url

    def test_base_url_error_does_not_echo_credentials(self):
        # [FUNCTION DOC]
        # A credential-bearing endpoint error masks the secret.
        """A credential-bearing endpoint is rejected without leaking the secret."""

        class StubClient(LLMClient):
            # [CLASS DOC]
            # HELPER: Minimal concrete LLMClient for this test.
            async def complete(self, prompt: str, **kwargs) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns "stub".
                return "stub"

        with pytest.raises(ValueError, match="credentials") as excinfo:
            StubClient("http://user:s3cret@localhost:8000/v1", None).base_url

        assert "s3cret" not in str(excinfo.value)
        assert "***@localhost:8000/v1" in str(excinfo.value)

    def test_base_url_conflicting_port_raises(self):
        # [FUNCTION DOC]
        # A port argument conflicting with the URL's port raises ValueError.
        """An explicit port that differs from the URL's port is rejected."""

        class StubClient(LLMClient):
            # [CLASS DOC]
            # HELPER: Minimal concrete LLMClient for this test.
            async def complete(self, prompt: str, **kwargs) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns "stub".
                return "stub"

        with pytest.raises(ValueError, match="conflicts with port=8000"):
            StubClient("http://localhost:11434", 8000).base_url

    @pytest.mark.asyncio
    async def test_complete_with_tools_not_implemented(self):
        # [FUNCTION DOC]
        # The default complete_with_tools raises NotImplementedError naming the subclass.
        """Default complete_with_tools raises NotImplementedError."""

        class StubClient(LLMClient):
            # [CLASS DOC]
            # HELPER: Minimal concrete LLMClient for this test.
            async def complete(self, prompt: str, **kwargs) -> str:
                # [FUNCTION DOC]
                # HELPER: Returns "stub".
                return "stub"

        client = StubClient("http://localhost", 8000)
        with pytest.raises(NotImplementedError, match="StubClient"):
            await client.complete_with_tools([], [])


class TestOpenAIClientConstruction:
    # [CLASS DOC]
    # Tests OpenAIClient construction (AsyncOpenAI patched).
    """Test OpenAIClient initialization."""

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_basic_construction(self, mock_openai_cls):
        # [FUNCTION DOC]
        # Stores endpoint, port, model and default temperature 0.0, max_tokens 256, no system prompt; the SDK
        # gets base URL <base>/v1 and api key "not-needed".
        """OpenAIClient stores params and creates AsyncOpenAI."""
        client = OpenAIClient("http://localhost", 8000, model="gpt-4")

        assert client.endpoint == "http://localhost"
        assert client.port == 8000
        assert client.model == "gpt-4"
        assert client.temperature == 0.0
        assert client.max_tokens == 256
        assert client.system_prompt is None

        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
        )

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_custom_api_key(self, mock_openai_cls):
        # [FUNCTION DOC]
        # A given api key is passed to the SDK.
        """API key is passed through to AsyncOpenAI."""
        OpenAIClient("http://localhost", 8000, model="gpt-4", api_key="sk-test-123")

        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="sk-test-123",
        )

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_default_api_key_when_none(self, mock_openai_cls):
        # [FUNCTION DOC]
        # api_key None becomes "not-needed".
        """api_key=None defaults to 'not-needed'."""
        OpenAIClient("http://localhost", 8000, model="gpt-4", api_key=None)

        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
        )

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_system_prompt_stored(self, mock_openai_cls):
        # [FUNCTION DOC]
        # The system prompt is stored.
        """System prompt is stored for use in complete()."""
        client = OpenAIClient(
            "http://localhost",
            8000,
            model="gpt-4",
            system_prompt="You are a judge.",
        )
        assert client.system_prompt == "You are a judge."

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_custom_temperature_and_max_tokens(self, mock_openai_cls):
        # [FUNCTION DOC]
        # Custom temperature and max_tokens are stored.
        """Custom temperature and max_tokens are stored."""
        client = OpenAIClient(
            "http://localhost",
            8000,
            model="gpt-4",
            temperature=0.7,
            max_tokens=512,
        )
        assert client.temperature == 0.7
        assert client.max_tokens == 512

    @pytest.mark.parametrize(
        ("endpoint", "port", "expected"),
        [
            ("http://localhost:8000", None, "http://localhost:8000/v1"),
            ("http://localhost:8000", 8000, "http://localhost:8000/v1"),
            ("http://localhost:8000/", None, "http://localhost:8000/v1"),
            ("http://localhost:8000/v1", None, "http://localhost:8000/v1"),
            ("http://localhost:8000/v1/", None, "http://localhost:8000/v1"),
            ("http://localhost", 8001, "http://localhost:8001/v1"),
            ("http://proxy:4000/litellm", None, "http://proxy:4000/litellm"),
            (
                "https://api.groq.com/openai/v1",
                None,
                "https://api.groq.com/openai/v1",
            ),
            (
                "https://generativelanguage.googleapis.com/v1beta/openai",
                None,
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
        ],
    )
    def test_endpoint_url_forms(self, endpoint, port, expected):
        # [FUNCTION DOC]
        # Parametrized: /v1 is appended only when the URL has no path; existing paths are used as-is.
        """/v1 is appended only when the URL has no path; a path is used as-is."""
        with patch("openenv.core.llm_client.AsyncOpenAI") as mock_openai_cls:
            OpenAIClient(endpoint, port, model="gpt-4")

        mock_openai_cls.assert_called_once_with(
            base_url=expected,
            api_key="not-needed",
        )

    def test_endpoint_with_malformed_port_raises(self):
        # [FUNCTION DOC]
        # A double-port endpoint raises ValueError.
        """The double-port form fails before any client is created."""
        with pytest.raises(ValueError, match="Invalid endpoint URL"):
            OpenAIClient("http://localhost:8000:8000", 8000, model="gpt-4")

    def test_endpoint_with_conflicting_port_raises(self):
        # [FUNCTION DOC]
        # A conflicting port raises ValueError.
        """A port argument that contradicts the URL's port is rejected."""
        with pytest.raises(ValueError, match="conflicts with port=8000"):
            OpenAIClient("http://localhost:11434", 8000, model="gpt-4")


class TestOpenAIClientRequestUrl:
    # [CLASS DOC]
    # Tests the URL actually requested by the real OpenAI SDK.
    """The URL the SDK actually requests, not only what is handed to it."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("endpoint", "port", "expected"),
        [
            (
                "http://localhost:8000",
                None,
                "http://localhost:8000/v1/chat/completions",
            ),
            (
                "http://localhost:8000/v1",
                None,
                "http://localhost:8000/v1/chat/completions",
            ),
            ("http://localhost", 8001, "http://localhost:8001/v1/chat/completions"),
            (
                "https://gw.example.com/openai/v1",
                None,
                "https://gw.example.com/openai/v1/chat/completions",
            ),
        ],
    )
    async def test_chat_completion_request_url(self, endpoint, port, expected):
        # [FUNCTION DOC]
        # Parametrized: complete reaches <base>/chat/completions via an httpx mock transport and returns its content.
        """complete() reaches <base>/chat/completions through the real SDK."""
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            # [FUNCTION DOC]
            # HELPER: Mock transport handler recording the URL and returning a minimal chat completion "ok".
            requested.append(str(request.url))
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        def build_sdk_client(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Builds a real AsyncOpenAI with the given kwargs and an httpx client on the mock transport.
            transport = httpx.MockTransport(handler)
            return AsyncOpenAI(
                http_client=httpx.AsyncClient(transport=transport), **kwargs
            )

        with patch("openenv.core.llm_client.AsyncOpenAI", side_effect=build_sdk_client):
            client = OpenAIClient(endpoint, port, model="m")

        assert await client.complete("hi") == "ok"
        assert requested == [expected]


class TestOpenAIClientComplete:
    # [CLASS DOC]
    # Tests OpenAIClient.complete.
    """Test the complete() method."""

    @pytest.mark.asyncio
    async def test_complete_without_system_prompt(self):
        # [FUNCTION DOC]
        # Sends only the user message with default sampling and returns the content.
        """complete() sends user message only when no system prompt."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "42"
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openenv.core.llm_client.AsyncOpenAI", return_value=mock_openai):
            client = OpenAIClient("http://localhost", 8000, model="gpt-4")
            result = await client.complete("What is 2+2?")

        assert result == "42"
        mock_openai.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=0.0,
            max_tokens=256,
        )

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self):
        # [FUNCTION DOC]
        # Prepends the system message when configured.
        """complete() includes system message when system_prompt is set."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.8"
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openenv.core.llm_client.AsyncOpenAI", return_value=mock_openai):
            client = OpenAIClient(
                "http://localhost",
                8000,
                model="gpt-4",
                system_prompt="You are a judge.",
            )
            result = await client.complete("Rate this code.")

        assert result == "0.8"
        mock_openai.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a judge."},
                {"role": "user", "content": "Rate this code."},
            ],
            temperature=0.0,
            max_tokens=256,
        )

    @pytest.mark.asyncio
    async def test_complete_kwargs_override(self):
        # [FUNCTION DOC]
        # Keyword arguments override temperature and max_tokens.
        """Keyword arguments override default temperature and max_tokens."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openenv.core.llm_client.AsyncOpenAI", return_value=mock_openai):
            client = OpenAIClient("http://localhost", 8000, model="gpt-4")
            await client.complete("hi", temperature=0.9, max_tokens=100)

        mock_openai.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.9,
            max_tokens=100,
        )


class TestOpenAIClientCompleteWithTools:
    # [CLASS DOC]
    # Tests OpenAIClient.complete_with_tools.
    """Test complete_with_tools() on OpenAIClient."""

    @pytest.mark.asyncio
    async def test_no_tool_calls(self):
        # [FUNCTION DOC]
        # A response without tool calls yields an LLMResponse with empty tool_calls.
        """Response without tool calls returns empty tool_calls list."""
        mock_openai = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Hello there"
        mock_msg.tool_calls = None
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_msg
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openenv.core.llm_client.AsyncOpenAI", return_value=mock_openai):
            client = OpenAIClient("http://localhost", 8000, model="gpt-4")
            result = await client.complete_with_tools(
                [{"role": "user", "content": "hi"}], []
            )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello there"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_kwargs_override(self):
        # [FUNCTION DOC]
        # Keyword arguments override sampling defaults; no tools parameter is sent for an empty tool list.
        """Keyword arguments override default temperature and max_tokens."""
        mock_openai = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "Hello there"
        mock_msg.tool_calls = None
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_msg
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openenv.core.llm_client.AsyncOpenAI", return_value=mock_openai):
            client = OpenAIClient("http://localhost", 8000, model="gpt-4")
            await client.complete_with_tools(
                [{"role": "user", "content": "hi"}],
                [],
                temperature=0.9,
                max_tokens=100,
            )

        mock_openai.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.9,
            max_tokens=100,
        )

    @pytest.mark.asyncio
    async def test_with_tool_calls(self):
        # [FUNCTION DOC]
        # Tool calls are parsed into ToolCall objects with JSON-decoded arguments.
        """Response with tool calls are parsed into ToolCall objects."""
        mock_openai = MagicMock()
        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"city": "SF"}'

        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.tool_calls = [mock_tc]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = mock_msg
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openenv.core.llm_client.AsyncOpenAI", return_value=mock_openai):
            client = OpenAIClient("http://localhost", 8000, model="gpt-4")
            result = await client.complete_with_tools(
                [{"role": "user", "content": "weather?"}],
                [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == {"city": "SF"}


class TestAnthropicClientConstruction:
    # [CLASS DOC]
    # Tests AnthropicClient construction.
    """Test AnthropicClient initialization."""

    def test_missing_anthropic_package(self):
        # [FUNCTION DOC]
        # Without the anthropic package, construction raises ImportError.
        """Raises ImportError with helpful message when anthropic is missing."""
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                AnthropicClient(
                    "https://api.anthropic.com", 443, model="claude-sonnet-4-20250514"
                )

    @patch("openenv.core.llm_client.AnthropicClient.__init__", return_value=None)
    def test_is_llm_client_subclass(self, mock_init):
        # [FUNCTION DOC]
        # AnthropicClient subclasses LLMClient.
        """AnthropicClient is a proper LLMClient subclass."""
        assert issubclass(AnthropicClient, LLMClient)


class TestAnthropicClientComplete:
    # [CLASS DOC]
    # Tests AnthropicClient.complete.
    """Test the complete() method on AnthropicClient."""

    @pytest.mark.asyncio
    async def test_complete_basic(self):
        # [FUNCTION DOC]
        # complete calls the messages API on a manually-initialized client and returns the text block.
        """complete() calls the Anthropic messages API and returns text."""
        mock_anthropic = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "4"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        with patch(
            "openenv.core.llm_client.AnthropicClient.__init__", return_value=None
        ):
            client = AnthropicClient.__new__(AnthropicClient)
            client.endpoint = "https://api.anthropic.com"
            client.port = 443
            client.model = "claude-sonnet-4-20250514"
            client.system_prompt = None
            client.temperature = 0.0
            client.max_tokens = 256
            client._client = mock_anthropic

            result = await client.complete("What is 2+2?")

        assert result == "4"
        mock_anthropic.messages.create.assert_called_once()


class TestAnthropicClientCompleteWithTools:
    # [CLASS DOC]
    # Tests AnthropicClient.complete_with_tools.
    """Test complete_with_tools() on AnthropicClient."""

    @pytest.mark.asyncio
    async def test_with_tool_use_response(self):
        # [FUNCTION DOC]
        # Text and tool_use blocks are parsed into content and ToolCall objects.
        """Tool use blocks are parsed into ToolCall objects."""
        mock_anthropic = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Let me check"
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "toolu_abc"
        mock_tool_block.name = "get_weather"
        mock_tool_block.input = {"city": "SF"}
        mock_response = MagicMock()
        mock_response.content = [mock_text_block, mock_tool_block]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient.__new__(AnthropicClient)
        client.endpoint = "https://api.anthropic.com"
        client.port = 443
        client.model = "claude-sonnet-4-20250514"
        client.system_prompt = None
        client.temperature = 0.0
        client.max_tokens = 256
        client._client = mock_anthropic

        result = await client.complete_with_tools(
            [{"role": "user", "content": "weather?"}],
            [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        )

        assert result.content == "Let me check"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_abc"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == {"city": "SF"}


# ---------------------------------------------------------------------------
# LLMResponse / ToolCall
# ---------------------------------------------------------------------------


class TestLLMResponse:
    # [CLASS DOC]
    # Tests LLMResponse.to_message_dict.
    """Test LLMResponse dataclass."""

    def test_to_message_dict_no_tools(self):
        # [FUNCTION DOC]
        # Without tool calls the dict is a plain assistant message.
        """to_message_dict without tool calls is a plain assistant message."""
        resp = LLMResponse(content="hello")
        msg = resp.to_message_dict()
        assert msg == {"role": "assistant", "content": "hello"}
        assert "tool_calls" not in msg

    def test_to_message_dict_with_tools(self):
        # [FUNCTION DOC]
        # With tool calls the dict includes OpenAI-format tool_calls.
        """to_message_dict includes tool_calls in OpenAI format."""
        resp = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="c1", name="foo", args={"x": 1})],
        )
        msg = resp.to_message_dict()
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "c1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "foo"
        assert json.loads(tc["function"]["arguments"]) == {"x": 1}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateLLMClient:
    # [CLASS DOC]
    # Tests create_llm_client.
    """Test the create_llm_client factory."""

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_openai_provider(self, mock_openai_cls):
        # [FUNCTION DOC]
        # "openai" creates an OpenAIClient.
        """'openai' creates an OpenAIClient."""
        client = create_llm_client("openai", "gpt-4", "sk-key")
        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-4"

    def test_anthropic_provider(self):
        # [FUNCTION DOC]
        # "anthropic" creates an AnthropicClient.
        """'anthropic' creates an AnthropicClient."""
        mock_async_anthropic = MagicMock()
        mock_module = MagicMock()
        mock_module.AsyncAnthropic = MagicMock(return_value=mock_async_anthropic)
        with patch.dict("sys.modules", {"anthropic": mock_module}):
            client = create_llm_client(
                "anthropic", "claude-sonnet-4-20250514", "sk-ant"
            )
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-sonnet-4-20250514"

    def test_unsupported_provider(self):
        # [FUNCTION DOC]
        # An unknown provider raises ValueError.
        """Unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="google"):
            create_llm_client("google", "gemini-pro", "key")

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_case_insensitive(self, mock_openai_cls):
        # [FUNCTION DOC]
        # Provider names are case-insensitive.
        """Provider name is case-insensitive."""
        client = create_llm_client("OpenAI", "gpt-4", "sk-key")
        assert isinstance(client, OpenAIClient)

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_custom_params(self, mock_openai_cls):
        # [FUNCTION DOC]
        # Temperature and max_tokens are forwarded.
        """Temperature and max_tokens are forwarded."""
        client = create_llm_client(
            "openai", "gpt-4", "sk-key", temperature=0.5, max_tokens=1024
        )
        assert client.temperature == 0.5
        assert client.max_tokens == 1024

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_system_prompt_forwarded(self, mock_openai_cls):
        # [FUNCTION DOC]
        # The system prompt is forwarded.
        """system_prompt is forwarded to the client."""
        client = create_llm_client(
            "openai", "gpt-4", "sk-key", system_prompt="You are a judge."
        )
        assert client.system_prompt == "You are a judge."


# ---------------------------------------------------------------------------
# MCP schema helpers
# ---------------------------------------------------------------------------


class TestCleanMCPSchema:
    # [CLASS DOC]
    # Tests _clean_mcp_schema.
    """Test _clean_mcp_schema helper."""

    def test_non_dict_returns_empty(self):
        # [FUNCTION DOC]
        # A non-dict yields an empty object schema.
        assert _clean_mcp_schema("not a dict") == {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def test_passthrough_simple_object(self):
        # [FUNCTION DOC]
        # A simple object schema keeps its properties.
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = _clean_mcp_schema(schema)
        assert result["properties"]["x"]["type"] == "string"

    def test_oneOf_selects_object(self):
        # [FUNCTION DOC]
        # oneOf resolves to the object alternative.
        schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "object", "properties": {"a": {"type": "int"}}},
            ]
        }
        result = _clean_mcp_schema(schema)
        assert "a" in result["properties"]

    def test_allOf_merges(self):
        # [FUNCTION DOC]
        # allOf merges properties and required lists.
        schema = {
            "allOf": [
                {"properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"properties": {"b": {"type": "int"}}, "required": ["b"]},
            ]
        }
        result = _clean_mcp_schema(schema)
        assert "a" in result["properties"]
        assert "b" in result["properties"]
        assert result["required"] == ["a", "b"]

    def test_anyOf_selects_object(self):
        # [FUNCTION DOC]
        # anyOf resolves to the object alternative.
        schema = {
            "anyOf": [
                {"type": "null"},
                {"type": "object", "properties": {"x": {"type": "string"}}},
            ]
        }
        result = _clean_mcp_schema(schema)
        assert "x" in result["properties"]

    def test_sets_default_type(self):
        # [FUNCTION DOC]
        # A missing type defaults to "object".
        result = _clean_mcp_schema({"properties": {"a": {"type": "string"}}})
        assert result["type"] == "object"

    def test_does_not_mutate_input(self):
        # [FUNCTION DOC]
        # The input dict is not modified.
        """_clean_mcp_schema must not modify the caller's dict."""
        original = {"type": "object"}
        _clean_mcp_schema(original)
        # Should not have added "properties" to the original dict.
        assert "properties" not in original


class TestMCPToolsToOpenAI:
    # [CLASS DOC]
    # Tests _mcp_tools_to_openai.
    """Test _mcp_tools_to_openai conversion."""

    def test_basic_conversion(self):
        # [FUNCTION DOC]
        # An MCP tool becomes an OpenAI function tool with parameters from inputSchema.
        mcp_tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        result = _mcp_tools_to_openai(mcp_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert "city" in result[0]["function"]["parameters"]["properties"]

    def test_empty_list(self):
        # [FUNCTION DOC]
        # An empty list converts to an empty list.
        assert _mcp_tools_to_openai([]) == []


class TestMCPToolsToAnthropic:
    # [CLASS DOC]
    # Tests _mcp_tools_to_anthropic.
    """Test _mcp_tools_to_anthropic conversion."""

    def test_basic_conversion(self):
        # [FUNCTION DOC]
        # An MCP tool becomes an Anthropic tool with name and input_schema.
        mcp_tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        result = _mcp_tools_to_anthropic(mcp_tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert "input_schema" in result[0]

    def test_empty_list(self):
        # [FUNCTION DOC]
        # An empty list converts to an empty list.
        assert _mcp_tools_to_anthropic([]) == []


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------


class TestOpenAIMsgsToAnthropic:
    # [CLASS DOC]
    # Tests _openai_msgs_to_anthropic.
    """Test _openai_msgs_to_anthropic conversion."""

    def test_system_extracted(self):
        # [FUNCTION DOC]
        # System messages are returned separately from the message list.
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, result = _openai_msgs_to_anthropic(msgs)
        assert system == "You are helpful."
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_tool_calls_converted(self):
        # [FUNCTION DOC]
        # Assistant tool calls become text plus tool_use content blocks with decoded input.
        msgs = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            },
        ]
        system, result = _openai_msgs_to_anthropic(msgs)
        assert system == ""
        assert len(result) == 2
        assistant_msg = result[1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        assert assistant_msg["content"][0]["type"] == "text"
        assert assistant_msg["content"][1]["type"] == "tool_use"
        assert assistant_msg["content"][1]["name"] == "get_weather"
        assert assistant_msg["content"][1]["input"] == {"city": "SF"}

    def test_tool_result_becomes_user_turn(self):
        # [FUNCTION DOC]
        # Tool messages become user turns with tool_result blocks referencing the call id.
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "foo",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"result": 42}'},
        ]
        _, result = _openai_msgs_to_anthropic(msgs)
        # tool result should be a user message with tool_result content block
        tool_turn = result[2]
        assert tool_turn["role"] == "user"
        assert isinstance(tool_turn["content"], list)
        assert tool_turn["content"][0]["type"] == "tool_result"
        assert tool_turn["content"][0]["tool_use_id"] == "c1"

    def test_multiple_system_messages_concatenated(self):
        # [FUNCTION DOC]
        # Multiple system messages are joined with blank lines.
        msgs = [
            {"role": "system", "content": "Rule 1."},
            {"role": "system", "content": "Rule 2."},
            {"role": "user", "content": "Hi"},
        ]
        system, _ = _openai_msgs_to_anthropic(msgs)
        assert system == "Rule 1.\n\nRule 2."
