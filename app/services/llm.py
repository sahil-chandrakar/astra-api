from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import LlmProfileConfig, LlmProviderStatus, LlmSettingsResponse, LlmSettingsUpdateRequest


PROVIDER_LABELS = {
    "cerebras": "Cerebras",
    "nvidia": "NVIDIA NIM",
}

CEREBRAS_FAST_MODELS = [
    "llama3.1-8b",
]

CEREBRAS_PRO_MODELS = [
    "zai-glm-4.7",
    "gpt-oss-120b",
    "qwen-3-235b-a22b-instruct-2507",
]

# NVIDIA Build catalog "Free Endpoint" chat models, checked May 21, 2026.
# Only chat-completions compatible models are exposed here; media, embedding,
# rerank, TTS, safety, and smoke-test failing endpoints are intentionally hidden.
NVIDIA_FAST_MODELS = [
    "google/gemma-3n-e2b-it",
    "google/gemma-3n-e4b-it",
    "microsoft/phi-4-multimodal-instruct",
    "abacusai/dracarys-llama-3.1-70b-instruct",
]

NVIDIA_PRO_MODELS = [
    "qwen/qwen3-coder-480b-a35b-instruct",
    "meta/llama-4-maverick-17b-128e-instruct",
]

NVIDIA_FREE_CHAT_MODEL_IDS = frozenset([*NVIDIA_FAST_MODELS, *NVIDIA_PRO_MODELS])


class LlmService:
    def __init__(self, settings: Settings):
        self.settings = settings
        base = Path(settings.data_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        self.settings_dir = base / "settings"
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.llm_settings_path = self.settings_dir / "llm.json"

    async def complete(self, system_prompt: str, user_prompt: str, model: str | None = None) -> tuple[str, list[str]]:
        provider, selected_model = self._resolve_model(model)
        if provider == "nvidia":
            return await self._complete_nvidia(system_prompt, user_prompt, selected_model)
        return await self._complete_cerebras(system_prompt, user_prompt, selected_model)

    def model_for_profile(self, profile: str) -> str:
        profiles = self.current_profiles()
        selected = profiles.get(profile if profile in {"fast", "pro"} else "pro") or profiles["pro"]
        return f"{selected.provider}:{selected.model}"

    def provider_configured(self, provider: str) -> bool:
        return self._provider_configured(provider)

    def provider_for_model(self, model: str | None = None) -> str:
        provider, _ = self._resolve_model(model)
        return provider

    def model_configured(self, model: str | None = None) -> bool:
        provider, _ = self._resolve_model(model)
        return self._provider_configured(provider)

    def missing_setup_for_model(self, model: str | None = None) -> str:
        provider, _ = self._resolve_model(model)
        if provider == "nvidia":
            return "NVIDIA_API_KEY"
        return "CEREBRAS_API_KEY"

    def settings_response(self) -> LlmSettingsResponse:
        return LlmSettingsResponse(profiles=self.current_profiles(), providers=self.provider_statuses())

    def update_settings(self, request: LlmSettingsUpdateRequest) -> LlmSettingsResponse:
        current = self.current_profiles()
        for profile_name, profile in request.profiles.items():
            allowed_models = self._models_for_profile(profile.provider, profile_name)
            if not allowed_models:
                continue
            model = profile.model.strip()
            if model not in allowed_models:
                model = allowed_models[0]
            current[profile_name] = LlmProfileConfig(provider=profile.provider, model=model)
        self._write_profiles(current)
        return self.settings_response()

    def current_profiles(self) -> dict[str, LlmProfileConfig]:
        defaults = self._default_profiles()
        raw = self._read_profiles()
        for profile_name in ("fast", "pro"):
            item = raw.get(profile_name)
            if not isinstance(item, dict):
                continue
            provider = str(item.get("provider") or "").strip().lower()
            model = str(item.get("model") or "").strip()
            allowed_models = self._models_for_profile(provider, profile_name)
            if allowed_models:
                defaults[profile_name] = LlmProfileConfig(
                    provider=provider,  # type: ignore[arg-type]
                    model=model if model in allowed_models else allowed_models[0],
                )
        return defaults

    def provider_statuses(self) -> list[LlmProviderStatus]:
        return [
            LlmProviderStatus(
                id="cerebras",
                label=PROVIDER_LABELS["cerebras"],
                configured=self.settings.has_cerebras,
                models=self._cerebras_models(),
                base_url="https://api.cerebras.ai/v1",
            ),
            LlmProviderStatus(
                id="nvidia",
                label=PROVIDER_LABELS["nvidia"],
                configured=self.settings.has_nvidia,
                models=self._nvidia_models(),
                base_url=self.settings.nvidia_base_url.rstrip("/"),
            ),
        ]

    async def _complete_cerebras(self, system_prompt: str, user_prompt: str, selected_model: str) -> tuple[str, list[str]]:
        if not self.settings.has_cerebras:
            return self._fallback_answer(user_prompt, "cerebras", selected_model), ["CEREBRAS_API_KEY"]

        try:
            from cerebras.cloud.sdk import Cerebras
        except Exception as exc:
            return (
                f"Astra could not load the Cerebras SDK even though CEREBRAS_API_KEY is configured: {exc}",
                ["CEREBRAS_SDK"],
            )

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                client = Cerebras(api_key=self.settings.cerebras_api_key)
                completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=selected_model,
                    max_completion_tokens=4096,
                    temperature=0.2,
                    top_p=1,
                    stream=False,
                )
                return completion.choices[0].message.content, []
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.6 * (attempt + 1))

        return (
            f"Astra reached the Cerebras client but the request failed. Check model, key permissions, quota, or network: {last_error}",
            ["CEREBRAS_API"],
        )

    async def _complete_nvidia(self, system_prompt: str, user_prompt: str, selected_model: str) -> tuple[str, list[str]]:
        if not self.settings.has_nvidia:
            return self._fallback_answer(user_prompt, "nvidia", selected_model), ["NVIDIA_API_KEY"]

        prompt_text = f"{system_prompt}\n{user_prompt}".lower()
        strict_json_task = "strict json" in prompt_text or "json only" in prompt_text
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 1,
            "stream": False,
        }
        if selected_model.startswith("moonshotai/kimi"):
            payload["chat_template_kwargs"] = {"thinking": not strict_json_task}

        headers = {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        url = f"{self.settings.nvidia_base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None
        attempt_count = 1 if strict_json_task else 3
        request_timeout = 45 if strict_json_task else 90
        for attempt in range(attempt_count):
            try:
                import httpx

                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                return self._content_from_chat_completion(data), []
            except Exception as exc:
                last_error = exc
                if attempt < attempt_count - 1:
                    await asyncio.sleep(0.6 * (attempt + 1))

        return (
            f"Astra reached NVIDIA NIM but the request failed. Check model, key permissions, quota, or network: {last_error}",
            ["NVIDIA_API"],
        )

    def _resolve_model(self, model: str | None) -> tuple[str, str]:
        selected = (model or self.model_for_profile("pro")).strip()
        if ":" in selected:
            provider, selected_model = selected.split(":", 1)
            provider = provider.strip().lower()
            selected_model = selected_model.strip()
            if provider in {"cerebras", "nvidia"} and selected_model:
                return provider, selected_model
        return "cerebras", selected or self.settings.resolved_cerebras_pro_model

    def _provider_configured(self, provider: str) -> bool:
        provider = provider.strip().lower()
        if provider == "cerebras":
            return self.settings.has_cerebras
        if provider == "nvidia":
            return self.settings.has_nvidia
        return False

    def _default_profiles(self) -> dict[str, LlmProfileConfig]:
        return {
            "fast": LlmProfileConfig(provider="cerebras", model=self.settings.resolved_cerebras_fast_model),
            "pro": LlmProfileConfig(provider="cerebras", model=self.settings.resolved_cerebras_pro_model),
        }

    def _cerebras_models(self) -> list[str]:
        return self._dedupe_models([*self._cerebras_fast_models(), *self._cerebras_pro_models()])

    def _cerebras_fast_models(self) -> list[str]:
        return self._dedupe_models([self.settings.resolved_cerebras_fast_model, *CEREBRAS_FAST_MODELS])

    def _cerebras_pro_models(self) -> list[str]:
        return self._dedupe_models([self.settings.resolved_cerebras_pro_model, *CEREBRAS_PRO_MODELS])

    def _nvidia_models(self) -> list[str]:
        return self._dedupe_models([*self._nvidia_pro_models(), *self._nvidia_fast_models()])

    def _nvidia_fast_models(self) -> list[str]:
        return self._dedupe_models([*NVIDIA_FAST_MODELS])

    def _nvidia_pro_models(self) -> list[str]:
        configured = self.settings.nvidia_model.strip()
        models = [*NVIDIA_PRO_MODELS]
        if configured in NVIDIA_FREE_CHAT_MODEL_IDS:
            models.insert(0, configured)
        return self._dedupe_models(models)

    def _models_for_profile(self, provider: str, profile_name: str) -> list[str]:
        provider = provider.strip().lower()
        profile_name = profile_name if profile_name in {"fast", "pro"} else "pro"
        if provider == "cerebras":
            return self._cerebras_fast_models() if profile_name == "fast" else self._cerebras_pro_models()
        if provider == "nvidia":
            return self._nvidia_fast_models() if profile_name == "fast" else self._nvidia_pro_models()
        return []

    def _dedupe_models(self, models: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for model in models:
            clean = model.strip()
            if clean and clean not in seen:
                seen.add(clean)
                deduped.append(clean)
        return deduped

    def _read_profiles(self) -> dict[str, Any]:
        if not self.llm_settings_path.exists():
            return {}
        try:
            raw = json.loads(self.llm_settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw.get("profiles", raw) if isinstance(raw, dict) else {}

    def _write_profiles(self, profiles: dict[str, LlmProfileConfig]) -> None:
        payload = {"profiles": {name: profile.model_dump(mode="json") for name, profile in profiles.items()}}
        self.llm_settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _content_from_chat_completion(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"].get("content", "")
        except Exception:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content or "")

    def _fallback_answer(self, user_prompt: str, provider: str, model: str) -> str:
        return (
            f"Astra is running in setup mode. Add the {provider.upper()} API key in astra-backend/.env "
            f"to enable {model} reasoning. I can still show the workflow, route tasks, "
            f"and prepare structured steps for: {user_prompt[:220]}"
        )
