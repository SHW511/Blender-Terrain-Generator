"""
AI-powered heightmap generation from color map images.

Sends a color D&D battle map to OpenAI (gpt-image-1) or Google Gemini
and receives a grayscale heightmap in return. Uses only Python stdlib
(no pip dependencies) since Blender extensions cannot bundle third-party packages.
"""

import base64
import json
import os
import uuid
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "Convert this color D&D battle map into a grayscale heightmap image. "
    "White = highest elevation, black = lowest. "
    "Water (rivers, lakes, ocean) should be near-black. "
    "Roads, paths, and flat ground should be mid-gray. "
    "Hills, cliffs, and elevated terrain should be light gray to white. "
    "Buildings and walls should be slightly elevated above surrounding ground. "
    "Maintain smooth gradients between elevation zones for natural 3D terrain. "
    "Output a single grayscale image with the same dimensions as the input."
)


def generate_heightmap(provider, api_key, image_path, custom_prompt=""):
    """Convert a color map image to a grayscale heightmap via AI.

    Args:
        provider: "OPENAI" or "GEMINI"
        api_key: API key for the chosen provider
        image_path: Absolute path to the source color map image
        custom_prompt: Optional extra instructions appended to the default prompt

    Returns:
        Path to the saved grayscale heightmap PNG.

    Raises:
        ValueError: Bad inputs (missing key, missing file, unknown provider).
        ConnectionError: Network / API errors.
    """
    if not api_key:
        raise ValueError("API key is empty. Set it in Edit > Preferences > Add-ons > D&D Tile Forge.")
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(f"Source image not found: {image_path}")

    prompt = DEFAULT_PROMPT
    if custom_prompt:
        prompt += "\n\nAdditional instructions: " + custom_prompt

    if provider == "OPENAI":
        img_bytes = _call_openai(api_key, image_path, prompt)
    elif provider == "GEMINI":
        img_bytes = _call_gemini(api_key, image_path, prompt)
    else:
        raise ValueError(f"Unknown AI provider: {provider}")

    # Save result next to source image
    out_dir = os.path.dirname(image_path)
    out_path = os.path.join(out_dir, "tileforge_ai_heightmap.png")
    with open(out_path, "wb") as f:
        f.write(img_bytes)

    return out_path


# ---------------------------------------------------------------------------
# OpenAI gpt-image-1  (images/edits endpoint, multipart/form-data)
# ---------------------------------------------------------------------------

def _call_openai(api_key, image_path, prompt):
    """Call OpenAI image edit endpoint and return PNG bytes."""
    boundary = "----TileForge" + uuid.uuid4().hex

    with open(image_path, "rb") as f:
        image_data = f.read()

    # Detect MIME type from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    filename = os.path.basename(image_path)

    # Build multipart body manually (no `requests` library available)
    parts = []

    # model field
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"gpt-image-1"
    )

    # prompt field
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{prompt}"
    )

    # size field — "auto" lets the API match the input aspect ratio
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="size"\r\n\r\n'
        f"auto"
    )

    # image[] file field
    file_header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image[]"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    )

    # Assemble body as bytes
    body = b""
    for part in parts:
        body += part.encode("utf-8") + b"\r\n"
    body += file_header.encode("utf-8") + image_data + b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/images/edits",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise ConnectionError(
            f"OpenAI API error {e.code}: {body_text[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"Network error connecting to OpenAI: {e.reason}") from e

    # Extract base64 image data
    try:
        b64_data = result["data"][0]["b64_json"]
    except (KeyError, IndexError) as e:
        # Fallback: check for URL response
        try:
            url = result["data"][0]["url"]
            with urllib.request.urlopen(url, timeout=60) as img_resp:
                return img_resp.read()
        except Exception:
            raise ConnectionError(
                f"Unexpected OpenAI response format: {json.dumps(result)[:500]}"
            ) from e

    try:
        return base64.b64decode(b64_data)
    except Exception as e:
        raise ConnectionError("Failed to decode OpenAI image data") from e


# ---------------------------------------------------------------------------
# Google Gemini  (generateContent with image output)
# ---------------------------------------------------------------------------

def _call_gemini(api_key, image_path, prompt):
    """Call Gemini generateContent endpoint and return PNG bytes."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime,
                            "data": base64.b64encode(image_data).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.0-flash:generateContent"
    )

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise ConnectionError(
            f"Gemini API error {e.code}: {body_text[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"Network error connecting to Gemini: {e.reason}") from e

    # Find image data in response parts
    try:
        parts = result["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError) as e:
        raise ConnectionError(
            f"Unexpected Gemini response format: {json.dumps(result)[:500]}"
        ) from e

    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and "data" in inline:
            try:
                return base64.b64decode(inline["data"])
            except Exception as e:
                raise ConnectionError("Failed to decode Gemini image data") from e

    raise ConnectionError(
        "Gemini response contained no image data. "
        "The model may not have generated an image output."
    )
