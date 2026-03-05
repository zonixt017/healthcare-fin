#!/usr/bin/env python3
"""Validate cloud inference credentials (OpenRouter preferred, HuggingFace fallback)."""

import json
import os
import sys
from urllib import request, error

from huggingface_hub import InferenceClient
from dotenv import load_dotenv


def fail(msg: str, code: int = 1):
    print(f"❌ {msg}")
    sys.exit(code)


def validate_openrouter() -> tuple[bool, str]:
    api_key = (
        os.getenv("OPENROUTER_API_KEY", "").strip()
        or os.getenv("OPENROUTER_KEY", "").strip()
        or os.getenv("OR_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not api_key:
        return False, "OpenRouter key is not set (OPENROUTER_API_KEY / OPENROUTER_KEY / OR_API_KEY / OPENAI_API_KEY)."

    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free").strip()
    fallbacks = [
        candidate.strip()
        for candidate in os.getenv("OPENROUTER_FALLBACKS", "google/gemma-2-9b-it:free,mistralai/mistral-7b-instruct:free").split(",")
        if candidate.strip()
    ]
    timeout = float(os.getenv("OPENROUTER_TIMEOUT", "45"))
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

    for candidate in [model, *fallbacks]:
        print(f"🔎 Checking OpenRouter model callability: {candidate}")
        payload = {
            "model": candidate,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        req = request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices") or []
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            print(f"✅ OpenRouter succeeded with {candidate}. Sample output: {content!r}")
            return True, ""
        except Exception as exc:
            print(f"⚠️ OpenRouter failed for {candidate}: {type(exc).__name__}: {exc}")

    return False, "OpenRouter call failed for all configured models."


def validate_hf() -> tuple[bool, str]:
    token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
    )
    if not token:
        return False, "HuggingFace token not set."

    model = os.getenv("HF_INFERENCE_API", "Qwen/Qwen2.5-7B-Instruct").strip()
    fallbacks = [
        candidate.strip()
        for candidate in os.getenv("HF_INFERENCE_FALLBACKS", "HuggingFaceH4/zephyr-7b-beta,microsoft/Phi-3-mini-4k-instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0").split(",")
        if candidate.strip()
    ]
    timeout = float(os.getenv("HF_API_TIMEOUT", "45"))
    task = os.getenv("HF_INFERENCE_TASK", "conversational").strip() or "conversational"
    task_fallbacks = [
        t.strip()
        for t in os.getenv("HF_INFERENCE_TASK_FALLBACKS", "text-generation,conversational").split(",")
        if t.strip()
    ]

    prompt = "Reply with exactly: ok"
    candidates = [model, *fallbacks]
    tasks = [task, *[t for t in task_fallbacks if t != task]]

    for candidate in candidates:
        for task_name in tasks:
            print(f"🔎 Checking HuggingFace model callability: {candidate} (task={task_name})")
            client = InferenceClient(model=candidate, token=token, timeout=timeout)

            try:
                if task_name == "conversational":
                    output = client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=16,
                        temperature=0.0,
                    )
                    sample = output.choices[0].message.content if output.choices else ""
                else:
                    sample = client.text_generation(prompt, max_new_tokens=8, temperature=0.0)

                print(f"✅ HuggingFace succeeded with {candidate} (task={task_name}). Sample output: {sample!r}")
                return True, ""
            except Exception as exc:
                print(f"⚠️ HuggingFace failed for {candidate}|{task_name}: {type(exc).__name__}: {exc}")

    return False, "HuggingFace call failed for all configured model/task combinations."


def main() -> int:
    load_dotenv()

    ok, reason = validate_openrouter()
    if ok:
        print("✅ Cloud inference is ready via OpenRouter.")
        return 0

    print(f"ℹ️ {reason}")
    ok, reason = validate_hf()
    if ok:
        print("✅ Cloud inference is ready via HuggingFace.")
        return 0

    fail(f"No working cloud provider found. {reason}", code=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
