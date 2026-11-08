"""Streamlit custom component that reads image data from the clipboard."""

from pathlib import Path
from typing import Any
import base64
import binascii

import streamlit.components.v1 as components

from config import MAX_UPLOAD_BYTES

_COMPONENT = components.declare_component(
    "satellite_clipboard_image",
    path=str(Path(__file__).parent / "components" / "clipboard_image"),
)


def clipboard_image_input(label: str, key: str) -> dict[str, Any] | None:
    """Show a keyboard-paste target and return its clipboard image payload."""
    return _COMPONENT(
        label=label,
        max_bytes=MAX_UPLOAD_BYTES,
        key=key,
        default=None,
    )


def decode_clipboard_payload(
    payload: dict[str, Any] | None, label: str
) -> tuple[bytes, str] | None:
    """Validate a component payload and return image bytes and a display name."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: clipboard returned an invalid image payload.")

    encoded = payload.get("data")
    mime = payload.get("mime", "")
    if not isinstance(encoded, str) or not isinstance(mime, str) or not mime.startswith("image/"):
        raise ValueError(f"{label}: clipboard data is not an image.")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{label}: clipboard image data is invalid.") from exc

    if not image_bytes:
        raise ValueError(f"{label}: clipboard image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"{label}: image is larger than the "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB paste limit."
        )
    return image_bytes, f"{label} ({mime})"
