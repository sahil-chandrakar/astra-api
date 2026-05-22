from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.models import AgentCommandRisk, AutomationRecipeCreateRequest, AutomationRecipeStatus
from app.services.agent_runtime import ToolRegistry


FUTURE_DESKTOP_TOOLS = {
    "desktop.observe",
}

RISKY_ACTION_WORDS = re.compile(
    r"\b(send|submit|post|publish|email|message|pay|purchase|buy|delete|remove|move|rename)\b",
    re.IGNORECASE,
)


@dataclass
class RecipeBuildResult:
    request: AutomationRecipeCreateRequest
    used_goal: str
    missing_tools: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


class AutomationBuilderService:
    """Build reusable automation recipes from goals and available tools.

    The builder is deliberately stricter than the runner: it can save a draft
    that names future desktop tools, but only marks recipes executable when
    every step is available and validated.
    """

    def __init__(self, registry: ToolRegistry, executable_tool_ids: set[str]):
        self.registry = registry
        self.executable_tool_ids = set(executable_tool_ids)
        self.known_builder_tool_ids = {tool.id for tool in registry.definitions()} | self.executable_tool_ids | FUTURE_DESKTOP_TOOLS | {"automation.save_recipe"}

    def build(self, goal: str, planned_steps: list[dict[str, Any]] | None = None, source_prompt: str = "") -> RecipeBuildResult:
        clean_goal = self._clean_goal(goal)
        steps = self._message_steps(clean_goal) or self._planned_steps(planned_steps or [])
        inputs = self._infer_inputs(clean_goal, steps)
        risk = self._infer_risk(clean_goal, steps)
        steps = self._ensure_safety_gate(clean_goal, steps, risk)
        missing_tools, validation_errors = self._validate_steps(steps)
        status = self._status_for(steps, missing_tools, validation_errors)
        if status == "needs_tools" and not missing_tools:
            validation_errors.append("Recipe uses unsupported or incomplete automation steps.")
        name = self._recipe_name(clean_goal)
        request = AutomationRecipeCreateRequest(
            name=name,
            prompt=clean_goal,
            steps=steps,
            inputs=inputs,
            risk=risk,
            status=status,
            missing_tools=missing_tools,
            validation_errors=validation_errors,
            built_from=source_prompt or clean_goal,
        )
        return RecipeBuildResult(request=request, used_goal=clean_goal, missing_tools=missing_tools, validation_errors=validation_errors)

    def message_steps_for_goal(self, goal: str) -> list[dict[str, Any]]:
        return self._message_steps(self._clean_goal(goal))

    def needs_goal(self, prompt: str) -> bool:
        goal = self.extract_goal(prompt)
        return not goal or self._is_generic_builder_prompt(goal)

    def extract_goal(self, prompt: str, clarification: str = "") -> str:
        note = self._clean_goal(clarification)
        if note and not self._is_generic_builder_prompt(note):
            return note

        text = self._clean_goal(prompt)
        goal_match = re.search(r"\bgoal\s*:\s*(.+?)(?:\btools?\s+allowed\b|\bsave\b|$)", text, re.IGNORECASE)
        if goal_match:
            goal = self._clean_goal(goal_match.group(1).strip(" .:-"))
            if goal:
                return goal

        if not re.search(r"\b(create|make|build|save)\b.*\b(automation|workflow|flow|recipe)\b|\bnew workflow\b", text, re.IGNORECASE):
            return text

        text = re.sub(r"\b(create|make|build|save)\s+(a\s+)?(new\s+)?(reusable\s+)?automation\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(workflow|flow|recipe)\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(to|that|which|where)\b\s+", " ", text, flags=re.IGNORECASE).strip(" .:-")
        return self._clean_goal(text)

    def _message_steps(self, goal: str) -> list[dict[str, Any]]:
        parsed = self._parse_message_goal(goal)
        if not parsed:
            return []
        app_name = parsed["app_name"]
        contact = parsed["contact"]
        message = parsed["message"]
        if app_name.strip().lower() in {"whatsapp", "whats app", "whatsapp desktop"}:
            return [
                {"tool": "app.resolve", "description": "Resolve WhatsApp.", "args": {"app_name": "WhatsApp"}},
                {"tool": "app.open", "description": "Open WhatsApp.", "args": {"app_name": "WhatsApp"}},
                {
                    "tool": "windows.prepare_whatsapp_message",
                    "description": f"Open chat with {contact} and type the message.",
                    "args": {"contact": contact, "message": message},
                },
                {
                    "tool": "windows.send_prepared_whatsapp_message",
                    "description": "Send the prepared WhatsApp message after approval.",
                    "args": {"contact": contact, "expected_message": message},
                },
                {
                    "tool": "desktop.verify_text",
                    "description": "Verify the sent message is visible.",
                    "args": {"text": message, "app_name": "WhatsApp", "exclude_control_types": ["Edit"], "timeout": 10},
                },
            ]
        return [
            {"tool": "app.resolve", "description": f"Resolve {app_name}.", "args": {"app_name": app_name}},
            {"tool": "app.open", "description": f"Open {app_name}.", "args": {"app_name": app_name}},
            {
                "tool": "desktop.find_text",
                "description": "Find the app search box.",
                "args": {"text": "Search", "control_types": ["Edit", "Text"], "app_name": app_name},
            },
            {"tool": "desktop.click", "description": "Focus the app search box.", "args": {"target": "$found_text"}},
            {"tool": "desktop.type_text", "description": f"Search for contact {contact}.", "args": {"text": contact, "replace": True}},
            {
                "tool": "desktop.find_text",
                "description": f"Find contact {contact}.",
                "args": {"text": contact, "exclude_control_types": ["Edit"], "app_name": app_name},
            },
            {"tool": "desktop.click", "description": f"Open chat with {contact}.", "args": {"target": "$found_text"}},
            {"tool": "desktop.type_text", "description": "Type the message.", "args": {"text": message}},
            {"tool": "desktop.press_key", "description": "Send the message after approval.", "args": {"key": "Enter"}},
            {"tool": "desktop.verify_text", "description": "Verify the sent message is visible.", "args": {"text": message}},
        ]

    def _parse_message_goal(self, goal: str) -> dict[str, str] | None:
        normalized = goal.lower()
        if not re.search(r"\b(send|message|dm|text)\b", normalized):
            return None
        if not re.search(r"\b(whatsapp|telegram|discord|slack|sms|message)\b", normalized):
            return None

        app_name = "WhatsApp" if "whatsapp" in normalized else "Messages"
        app_match = re.search(r"\bopen\s+([a-zA-Z0-9 ._-]{2,40}?)(?:\s+app)?\s+and\s+send\b", goal, re.IGNORECASE)
        if app_match:
            app_name = re.sub(r"\b(app|application)\b", " ", app_match.group(1), flags=re.IGNORECASE).strip() or app_name

        message = ""
        quoted = re.search(r"['\"]([^'\"]{1,500})['\"]", goal)
        if quoted:
            message = quoted.group(1).strip()
        else:
            msg_match = re.search(r"\bsend(?:\s+message)?\s+(.+?)\s+\bto\b", goal, re.IGNORECASE)
            if msg_match:
                message = msg_match.group(1).strip(" .")
                message = re.sub(r"^(?:whats\s*app|whatsapp|telegram|discord|slack|sms|messages?)\s+", "", message, flags=re.IGNORECASE).strip(" .")
                message = re.sub(r"^(?:message|text)\s+", "", message, flags=re.IGNORECASE).strip(" .")

        contact = ""
        contact_match = re.search(r"\bto\s+(.+)$", goal, re.IGNORECASE)
        if contact_match:
            contact = re.sub(r"\b(on|in|using)\s+(whatsapp|telegram|discord|slack|messages?)\b", " ", contact_match.group(1), flags=re.IGNORECASE)
            contact = contact.strip(" .")

        if not message or not contact:
            return None
        return {"app_name": app_name, "message": message[:500], "contact": contact[:120]}

    def _planned_steps(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clean: list[dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or "").strip()
            if tool == "automation.save_recipe":
                continue
            clean.append(
                {
                    "tool": tool,
                    "description": str(step.get("description") or tool).strip()[:240],
                    "args": step.get("args") if isinstance(step.get("args"), dict) else {},
                }
            )
        return clean

    def _validate_steps(self, steps: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        missing_tools: list[str] = []
        validation_errors: list[str] = []
        if not steps:
            return [], ["Recipe needs at least one automation step."]

        for index, step in enumerate(steps, start=1):
            tool = str(step.get("tool") or "").strip()
            if tool not in self.known_builder_tool_ids:
                validation_errors.append(f"Step {index} uses unknown tool: {tool or '<missing>'}.")
                missing_tools.append(tool or "<missing>")
            elif tool not in self.executable_tool_ids:
                missing_tools.append(tool)
            if not isinstance(step.get("args"), dict):
                validation_errors.append(f"Step {index} args must be an object.")

        return sorted(set(missing_tools)), validation_errors

    def _ensure_safety_gate(self, goal: str, steps: list[dict[str, Any]], risk: AgentCommandRisk) -> list[dict[str, Any]]:
        if risk != "safe_confirm":
            return steps
        if any(step.get("tool") in {"desktop.press_key", "windows.send_prepared_whatsapp_message", "python.run_safe", "browser.download"} for step in steps):
            return steps
        has_gate = any(step.get("tool") == "system.ask_user" for step in steps)
        if has_gate:
            return steps
        final_index = max(0, len(steps) - 1)
        gate = {
            "tool": "system.ask_user",
            "description": "Ask for approval before the external action.",
            "args": {"message": "Ready to perform the external action. Approve?"},
        }
        return [*steps[:final_index], gate, *steps[final_index:]]

    def _infer_inputs(self, goal: str, steps: list[dict[str, Any]]) -> list[str]:
        inputs: list[str] = []
        text = goal.lower()
        if re.search(r"\b(send|message|email|post)\b", text):
            inputs.extend(["recipient", "message"])
        for step in steps:
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            for value in args.values():
                if isinstance(value, str) and value.startswith("$"):
                    inputs.append(value.lstrip("$"))
        return list(dict.fromkeys(inputs))

    def _infer_risk(self, goal: str, steps: list[dict[str, Any]]) -> AgentCommandRisk:
        if RISKY_ACTION_WORDS.search(goal):
            return "safe_confirm"
        if any(str(step.get("tool") or "") in {"desktop.press_key", "browser.click"} for step in steps):
            return "safe_confirm"
        return "safe_auto"

    def _status_for(self, steps: list[dict[str, Any]], missing_tools: list[str], validation_errors: list[str]) -> AutomationRecipeStatus:
        if missing_tools:
            return "needs_tools"
        if validation_errors or not steps:
            return "draft"
        return "executable"

    def _recipe_name(self, goal: str) -> str:
        if "whatsapp" in goal.lower() and re.search(r"\b(send|message)\b", goal, re.IGNORECASE):
            return "Send WhatsApp message"
        words = re.sub(r"[^a-zA-Z0-9 ]+", " ", goal).split()
        return (" ".join(words[:6]).strip() or "Automation")[:64]

    def _is_generic_builder_prompt(self, text: str) -> bool:
        compact = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        return compact in {
            "create a new reusable automation",
            "create new reusable automation",
            "new reusable automation",
            "create automation",
            "save flow",
            "new workflow",
        }

    def _clean_goal(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()
