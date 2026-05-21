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

NVIDIA_FAST_MODELS = [
    "openai/gpt-oss-20b",
    "deepseek-ai/deepseek-v4-flash",
    "nvidia/nvidia-nemotron-nano-9b-v2",
    "nvidia/nemotron-3-nano-30b-a3b",
    "microsoft/phi-4-mini-flash-reasoning",
    "microsoft/phi-4-mini-instruct",
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.2-3b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "qwen/qwen2.5-coder-32b-instruct",
    "stepfun-ai/step-3-5-flash",
]

NVIDIA_PRO_MODELS = [
    "moonshotai/kimi-k2.6",
    "z-ai/glm5.1",
    "z-ai/glm4.7",
    "deepseek-ai/deepseek-v4-pro",
    "openai/gpt-oss-120b",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen3-5-122b-a10b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "minimaxai/minimax-m2.7",
    "moonshotai/kimi-k2-thinking",
    "moonshotai/kimi-k2-instruct",
    "mistralai/mistral-nemotron",
    "mistralai/mixtral-8x22b-instruct",
]


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
            if model in self._models_for_profile(provider, profile_name):
                defaults[profile_name] = LlmProfileConfig(provider=provider, model=model)  # type: ignore[arg-type]
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
            payload["chat_template_kwargs"] = {"thinking": True}

        headers = {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        url = f"{self.settings.nvidia_base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                import httpx

                async with httpx.AsyncClient(timeout=90) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                return self._content_from_chat_completion(data), []
            except Exception as exc:
                last_error = exc
                if attempt < 2:
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
        return self._dedupe_models([self.settings.nvidia_model, *NVIDIA_PRO_MODELS])

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
