# [MODULE DOC]
# HELPER: Content-conversion helpers between dialects for text and image parts:
# OpenAI chat parts (text, image_url), OpenAI Responses input parts, Anthropic blocks
# (text, image, document) and Google parts (text, inline_data, file_data). Handles
# base64 data URLs. Pure functions; no I/O.
"""Image content conversion helpers for gateway API transformers."""

from __future__ import annotations

import re
from typing import Any

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime_type>[^;,]+);base64,(?P<data>.*)$", re.DOTALL
)


def is_image_mime_type(mime_type: Any) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if mime_type is a string starting with "image/" (case-insensitive).
    return isinstance(mime_type, str) and mime_type.lower().startswith("image/")


def make_data_url(mime_type: str, data: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns data unchanged if it already starts with "data:", else builds
    # "data:<mime>;base64,<data>".
    if data.startswith("data:"):
        return data
    return f"data:{mime_type};base64,{data}"


def parse_data_url(url: str) -> tuple[str, str] | None:
    # [FUNCTION DOC]
    # HELPER: Parses a base64 data URL; returns (mime_type, data) only for image mime
    # types, else None.
    match = _DATA_URL_RE.match(url)
    if not match:
        return None
    mime_type = match.group("mime_type")
    if not is_image_mime_type(mime_type):
        return None
    return mime_type, match.group("data")


# OpenAI's image_url.detail only accepts these values; vLLM rejects anything
# else. Harnesses send their own (e.g. codex's "original"), so drop unknowns
# rather than forward a value that 400s the whole image request.
_VALID_IMAGE_DETAILS = frozenset({"auto", "low", "high"})


def openai_image_url_block(url: str, *, detail: Any = None) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Builds an OpenAI image_url part; includes detail only if it is one of
    # auto/low/high (other values dropped).
    image_url: dict[str, Any] = {"url": url}
    if isinstance(detail, str) and detail in _VALID_IMAGE_DETAILS:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def openai_text_block(text: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Builds an OpenAI text part.
    return {"type": "text", "text": text}


def openai_image_url(block: dict[str, Any]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns the URL of an OpenAI image part whether image_url is a string or a
    # dict with "url"; None if missing/empty.
    image_url = block.get("image_url")
    if isinstance(image_url, str) and image_url:
        return image_url
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str) and url:
            return url
    return None


def openai_image_detail(block: dict[str, Any]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns image_url.detail or the block-level detail when a non-empty string, else None.
    image_url = block.get("image_url")
    if isinstance(image_url, dict):
        detail = image_url.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    detail = block.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    return None


def openai_content_from_text_and_images(
    parts: list[dict[str, Any]],
    *,
    text_separator: str = "\n",
) -> str | list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: If any part is an image_url part returns the parts list unchanged; otherwise
    # joins the text of text parts with text_separator (default newline) into a string.
    has_image = any(part.get("type") == "image_url" for part in parts)
    if has_image:
        return parts
    return text_separator.join(
        part.get("text", "")
        for part in parts
        if part.get("type") == "text" and isinstance(part.get("text"), str)
    )


def openai_responses_input_content_to_chat(content: Any) -> str | list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: Converts OpenAI Responses input content to chat content: strings returned
    # as-is; non-lists stringified ("" if falsy); list blocks of type
    # input_text/output_text/text become text parts, input_image/image_url become image
    # parts (with block-level detail). Returns a string or list via
    # openai_content_from_text_and_images.
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""

    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            parts.append(openai_text_block(block))
            continue
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type in ("input_text", "output_text", "text"):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(openai_text_block(text))
            continue
        if block_type in ("input_image", "image_url"):
            url = _responses_image_url(block)
            if url:
                parts.append(openai_image_url_block(url, detail=block.get("detail")))

    return openai_content_from_text_and_images(parts)


def _responses_image_url(block: dict[str, Any]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns a Responses image block's URL (string image_url or image_url.url), else None.
    image_url = block.get("image_url")
    if isinstance(image_url, str) and image_url:
        return image_url
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str) and url:
            return url
    return None


def anthropic_content_to_openai_chat(content: Any) -> str | list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: Converts Anthropic content to OpenAI chat content: text blocks to text parts,
    # image blocks to image_url parts, document blocks to text parts (via
    # anthropic_document_to_text); other block types dropped. Strings returned as-is,
    # non-lists stringified.
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""

    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            parts.append(openai_text_block(block))
            continue
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(openai_text_block(text))
            continue
        if block_type == "image":
            image = anthropic_image_to_openai_chat(block)
            if image:
                parts.append(image)
            continue
        if block_type == "document":
            text = anthropic_document_to_text(block)
            if text:
                parts.append(openai_text_block(text))

    return openai_content_from_text_and_images(parts)


def anthropic_document_to_text(block: dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns text from an Anthropic document block: source type "text" -> data;
    # "content" -> newline-joined text blocks; anything else (e.g. base64 PDF) -> "".
    """Extract text from an Anthropic `document` block.

    Handles `source.type == "text"` and `source.type == "content"`. Base64
    PDFs are dropped — SGLang can't render binary docs through the chat
    template.
    """
    source = block.get("source")
    if not isinstance(source, dict):
        return ""
    source_type = source.get("type")
    if source_type == "text":
        data = source.get("data")
        return data if isinstance(data, str) else ""
    if source_type == "content":
        inner = source.get("content")
        if isinstance(inner, list):
            pieces: list[str] = []
            for inner_block in inner:
                if isinstance(inner_block, dict) and inner_block.get("type") == "text":
                    text = inner_block.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
            return "\n".join(pieces)
    return ""


def anthropic_image_to_openai_chat(block: dict[str, Any]) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: Converts an Anthropic image block (base64 with image media_type/mediaType,
    # or url source) to an OpenAI image_url part; None otherwise.
    source = block.get("source")
    if not isinstance(source, dict):
        return None

    source_type = source.get("type")
    if source_type == "base64":
        mime_type = source.get("media_type") or source.get("mediaType")
        data = source.get("data")
        if is_image_mime_type(mime_type) and isinstance(data, str) and data:
            return openai_image_url_block(make_data_url(mime_type, data))
    if source_type == "url":
        url = source.get("url")
        if isinstance(url, str) and url:
            return openai_image_url_block(url)
    return None


def google_content_parts_to_openai_chat(parts: Any) -> str | list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: Converts Gemini parts to OpenAI chat content, skipping thought parts; text
    # parts and image parts (inline/file data) kept. Non-list input -> "".
    if not isinstance(parts, list):
        return ""

    openai_parts: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            openai_parts.append(openai_text_block(part))
            continue
        if not isinstance(part, dict):
            continue

        # Thought parts are reasoning_content, not user-visible content.
        if part.get("thought") is True:
            continue

        text = part.get("text")
        if isinstance(text, str):
            openai_parts.append(openai_text_block(text))
            continue

        image = google_part_to_openai_chat(part)
        if image:
            openai_parts.append(image)

    return openai_content_from_text_and_images(openai_parts)


def google_part_to_openai_chat(part: dict[str, Any]) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: Converts a Gemini inline_data/inlineData image (as a data URL) or
    # file_data/fileData image (by URI) to an OpenAI image_url part; None otherwise.
    inline_data = part.get("inline_data") or part.get("inlineData")
    if isinstance(inline_data, dict):
        mime_type = inline_data.get("mime_type") or inline_data.get("mimeType")
        data = inline_data.get("data")
        if is_image_mime_type(mime_type) and isinstance(data, str) and data:
            return openai_image_url_block(make_data_url(mime_type, data))

    file_data = part.get("file_data") or part.get("fileData")
    if isinstance(file_data, dict):
        mime_type = file_data.get("mime_type") or file_data.get("mimeType")
        uri = file_data.get("file_uri") or file_data.get("fileUri")
        if is_image_mime_type(mime_type) and isinstance(uri, str) and uri:
            return openai_image_url_block(uri)

    return None


def openai_chat_content_to_anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: Converts OpenAI chat content to Anthropic blocks (text and image); never
    # returns empty (falls back to one empty text block).
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content) if content else ""}]

    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str):
                blocks.append({"type": "text", "text": text})
            continue
        if part_type == "image_url":
            image = openai_chat_image_to_anthropic(part)
            if image:
                blocks.append(image)

    return blocks or [{"type": "text", "text": ""}]


def openai_chat_image_to_anthropic(part: dict[str, Any]) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: Converts an OpenAI image part to an Anthropic image block: base64 source for
    # image data URLs, otherwise a url source. None if the part has no URL.
    url = openai_image_url(part)
    if not url:
        return None

    parsed = parse_data_url(url)
    if parsed:
        mime_type, data = parsed
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data,
            },
        }

    return {"type": "image", "source": {"type": "url", "url": url}}


def openai_chat_content_to_google_parts(content: Any) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: Converts OpenAI chat content to Gemini parts (text and image); never returns
    # empty (falls back to one empty text part).
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content) if content else ""}]

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str):
                parts.append({"text": text})
            continue
        if part_type == "image_url":
            image = openai_chat_image_to_google(part)
            if image:
                parts.append(image)

    return parts or [{"text": ""}]


def openai_chat_image_to_google(part: dict[str, Any]) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: Converts an OpenAI image part to a Gemini part: inline_data for image data
    # URLs, otherwise file_data with the URL.
    # KNOWN ISSUE: non-data URLs are always labelled mime type "image/jpeg" regardless of
    # the actual image type.
    url = openai_image_url(part)
    if not url:
        return None

    parsed = parse_data_url(url)
    if parsed:
        mime_type, data = parsed
        return {"inline_data": {"mime_type": mime_type, "data": data}}

    return {"file_data": {"mime_type": "image/jpeg", "file_uri": url}}
