from app.config import Settings


class LlmService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, list[str]]:
        if not self.settings.has_cerebras:
            return self._fallback_answer(user_prompt), ["CEREBRAS_API_KEY"]

        try:
            from cerebras.cloud.sdk import Cerebras

            client = Cerebras(api_key=self.settings.cerebras_api_key)
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.settings.cerebras_model,
                max_completion_tokens=1024,
                temperature=0.2,
                top_p=1,
                stream=False,
            )
            return completion.choices[0].message.content, []
        except Exception as exc:
            return (
                f"Astra could not reach Cerebras yet. Setup or network issue: {exc}",
                ["CEREBRAS_API_KEY"],
            )

    def _fallback_answer(self, user_prompt: str) -> str:
        return (
            "Astra is running in setup mode. Add CEREBRAS_API_KEY in astra-backend/.env "
            "to enable llama3.1-8b reasoning. I can still show the workflow, route tasks, "
            f"and prepare structured steps for: {user_prompt[:220]}"
        )
