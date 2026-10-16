# [MODULE DOC]
# Tests that the Google dialect transformer emits thoughtSignature as valid
# base64 (JSON bytes) of make_signature(thought text), for buffered and
# streamed responses.
"""Google's bytes fields must survive strict SDK JSON decoding."""

import base64

from openenv.core.harness.capture.dialects.google import (
    _GoogleStreamState,
    GoogleTransformer,
)
from openenv.core.harness.capture.dialects.reasoning import make_signature


def _assert_signature(response):
    # [FUNCTION DOC]
    # HELPER: asserts the first part is a thought whose thoughtSignature strictly
    # base64-decodes to make_signature of its text.
    part = response["candidates"][0]["content"]["parts"][0]
    assert part["thought"] is True
    assert base64.b64decode(
        part["thoughtSignature"], validate=True
    ).decode() == make_signature(part["text"])


def test_google_buffered_thought_signature_is_json_bytes():
    # [FUNCTION DOC]
    # transform_response on reasoning content yields a valid signature.
    response = GoogleTransformer().transform_response(
        {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "Inspect the CSV first.",
                        "content": "Working.",
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        {},
    )
    _assert_signature(response)


def test_google_streamed_thought_signature_is_json_bytes():
    # [FUNCTION DOC]
    # Every streamed chunk response for reasoning content has a valid signature.
    responses = _GoogleStreamState(GoogleTransformer()).process_chunk(
        {"choices": [{"delta": {"reasoning_content": "Inspect the CSV first."}}]}
    )
    assert responses
    for response in responses:
        _assert_signature(response)
