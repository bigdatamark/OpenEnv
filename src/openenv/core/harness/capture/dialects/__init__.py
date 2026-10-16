# [PACKAGE DOC]
# Dialect transformers package; exposes TransformManager, which maps each APIType to
# a singleton transformer instance (Anthropic, OpenAI chat, OpenAI Responses, Google).
"""Transform manager — dispatches to the right transformer by API type."""

from ..detection import APIType
from .anthropic import AnthropicTransformer
from .base import BaseTransformer
from .google import GoogleTransformer
from .openai_chat import OpenAIChatTransformer
from .openai_responses import OpenAIResponsesTransformer


class TransformManager:
    # [CLASS DOC]
    # Holds one transformer instance per APIType and returns the matching one.
    """Route to the correct transformer based on detected API type."""

    def __init__(self):
        # [FUNCTION DOC]
        # Instantiates the four transformers into a mapping keyed by APIType.
        self._transformers: dict[APIType, BaseTransformer] = {
            APIType.ANTHROPIC: AnthropicTransformer(),
            APIType.OPENAI_CHAT: OpenAIChatTransformer(),
            APIType.OPENAI_RESPONSES: OpenAIResponsesTransformer(),
            APIType.GOOGLE: GoogleTransformer(),
        }

    def get(self, api_type: APIType) -> BaseTransformer:
        # [FUNCTION DOC]
        # Returns the transformer for api_type. Raises KeyError for an unknown type.
        return self._transformers[api_type]
