import asyncio

from app.config import Settings


class LlmService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def complete(self, system_prompt: str, user_prompt: str, model: str | None = None) -> tuple[str, list[str]]:
        selected_model = (model or self.settings.resolved_cerebras_pro_model).strip() or self.settings.resolved_cerebras_pro_model
        if not self.settings.has_cerebras:
            return self._fallback_answer(user_prompt, selected_model), ["CEREBRAS_API_KEY"]

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

    def _fallback_answer(self, user_prompt: str, model: str) -> str:
        return (
            "Astra is running in setup mode. Add CEREBRAS_API_KEY in astra-backend/.env "
            f"to enable {model} reasoning. I can still show the workflow, route tasks, "
            f"and prepare structured steps for: {user_prompt[:220]}"
        )
