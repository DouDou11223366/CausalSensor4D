from __future__ import annotations

"""OpenRouter / OpenAI-compatible LLM client for CausalSensor4D public_release.

This module is intentionally small and explicit. It sends verified CausalSensor4D
prompt artifacts to an OpenRouter-compatible chat-completions endpoint and saves
both the raw response and the extracted assistant text.

Important safety rule: never hard-code API keys in the repository. Use the
OPENROUTER_API_KEY environment variable or another environment variable name
passed via config.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import time

import requests


@dataclass
class OpenRouterConfig:
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "deepseek/deepseek-chat-v3.1:free"
    temperature: float = 0.2
    max_tokens: int = 1800
    timeout: int = 120
    site_url: str = "https://local.causalsensor4d"
    app_title: str = "CausalSensor4D"


def get_api_key(api_key_env: str = "OPENROUTER_API_KEY") -> str:
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing OpenRouter API key. Set environment variable {api_key_env}.\n"
            f"Windows PowerShell example: $env:{api_key_env}='<OPENROUTER_API_KEY>'\n"
            "Do not write the key into source code or commit it to Git."
        )
    return key


def _extract_text(response_json: Dict[str, Any]) -> str:
    try:
        choices = response_json.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)
        return str(content)
    except Exception:
        return ""


def call_openrouter_chat(
    prompt: str,
    config: Optional[OpenRouterConfig] = None,
    system_message: Optional[str] = None,
) -> Dict[str, Any]:
    config = config or OpenRouterConfig()
    api_key = get_api_key(config.api_key_env)
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.site_url,
        "X-Title": config.app_title,
    }
    messages: List[Dict[str, str]] = []
    messages.append({
        "role": "system",
        "content": system_message or (
            "You are an autonomous-driving research assistant. Use only the provided "
            "CausalSensor4D evidence. Do not invent numeric values."
        ),
    })
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    started = time.time()
    response = requests.post(url, headers=headers, json=payload, timeout=config.timeout)
    elapsed = time.time() - started
    try:
        response_json = response.json()
    except Exception:
        response_json = {"raw_text": response.text}

    result = {
        "ok": bool(response.ok),
        "status_code": response.status_code,
        "elapsed_seconds": elapsed,
        "request_config": asdict(config) | {"api_key_env": config.api_key_env, "api_key_value": "<hidden>"},
        "response_json": response_json,
        "assistant_text": _extract_text(response_json),
    }
    if not response.ok:
        # Keep the error payload for debugging, but never include API key.
        raise RuntimeError(
            "OpenRouter request failed with status "
            f"{response.status_code}: {json.dumps(response_json, ensure_ascii=False)[:2000]}"
        )
    return result


def run_prompt_file(
    prompt_path: str | Path,
    out_dir: str | Path,
    output_stem: str,
    config: Optional[OpenRouterConfig] = None,
) -> Dict[str, str]:
    prompt_path = Path(prompt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_path.read_text(encoding="utf-8")
    result = call_openrouter_chat(prompt, config=config)

    raw_path = out_dir / f"{output_stem}.raw_response.json"
    text_path = out_dir / f"{output_stem}.md"
    raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(result.get("assistant_text", ""), encoding="utf-8")
    return {"raw_response": str(raw_path), "assistant_text": str(text_path)}
