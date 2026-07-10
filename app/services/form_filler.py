from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

from app.config import Settings
from app.services.llm import LlmService


SENSITIVE_FIELD_RE = re.compile(
    r"\b("
    r"password|passcode|pin|otp|one[- ]?time|captcha|security\s*code|cvv|cvc|"
    r"credit\s*card|debit\s*card|card\s*number|expiry|payment|bank|account\s*number|"
    r"routing|ifsc|upi|token|api\s*key|secret|recovery\s*code|private\s*key|"
    r"ssn|social\s*security|passport|aadhaar|aadhar|pan\s*card|tax\s*id|"
    r"signature|attestation|declaration"
    r")\b",
    re.IGNORECASE,
)

SUBMIT_ACTION_RE = re.compile(
    r"\b(submit|send|pay|purchase|buy|delete|remove|final|confirm\s+order|place\s+order|agree)\b",
    re.IGNORECASE,
)

KNOWN_SITE_PATTERNS: tuple[tuple[re.Pattern[str], str, str, str], ...] = (
    (
        re.compile(r"\b(youtube|you\s*tube|yt)\b", re.IGNORECASE),
        "https://accounts.google.com/ServiceLogin?service=youtube&continue=https%3A%2F%2Fwww.youtube.com%2F",
        "https://accounts.google.com/signup/v2/createaccount?service=youtube&continue=https%3A%2F%2Fwww.youtube.com%2F",
        "https://www.youtube.com/",
    ),
    (
        re.compile(r"\b(gmail|google\s*mail)\b", re.IGNORECASE),
        "https://accounts.google.com/ServiceLogin?service=mail&continue=https%3A%2F%2Fmail.google.com%2F",
        "https://accounts.google.com/signup/v2/createaccount?service=mail&continue=https%3A%2F%2Fmail.google.com%2F",
        "https://mail.google.com/",
    ),
    (
        re.compile(r"\b(google)\b", re.IGNORECASE),
        "https://accounts.google.com/",
        "https://accounts.google.com/signup/v2/createaccount",
        "https://www.google.com/",
    ),
    (
        re.compile(r"\b(github)\b", re.IGNORECASE),
        "https://github.com/login",
        "https://github.com/signup",
        "https://github.com/",
    ),
    (
        re.compile(r"\b(linkedin)\b", re.IGNORECASE),
        "https://www.linkedin.com/login",
        "https://www.linkedin.com/signup",
        "https://www.linkedin.com/",
    ),
    (
        re.compile(r"\b(gitlab)\b", re.IGNORECASE),
        "https://gitlab.com/users/sign_in",
        "https://gitlab.com/users/sign_up",
        "https://gitlab.com/",
    ),
    (
        re.compile(r"\b(stack\s*overflow|stackoverflow)\b", re.IGNORECASE),
        "https://stackoverflow.com/users/login",
        "https://stackoverflow.com/users/signup",
        "https://stackoverflow.com/",
    ),
    (
        re.compile(r"\b(reddit)\b", re.IGNORECASE),
        "https://www.reddit.com/login/",
        "https://www.reddit.com/register/",
        "https://www.reddit.com/",
    ),
    (
        re.compile(r"\b(twitter|x\.com|\bx\b)\b", re.IGNORECASE),
        "https://x.com/i/flow/login",
        "https://x.com/i/flow/signup",
        "https://x.com/",
    ),
    (
        re.compile(r"\b(instagram|insta)\b", re.IGNORECASE),
        "https://www.instagram.com/accounts/login/",
        "https://www.instagram.com/accounts/emailsignup/",
        "https://www.instagram.com/",
    ),
    (
        re.compile(r"\b(facebook|fb)\b", re.IGNORECASE),
        "https://mbasic.facebook.com/login/",
        "https://mbasic.facebook.com/reg/",
        "https://mbasic.facebook.com/",
    ),
    (
        re.compile(r"\b(microsoft|outlook|hotmail|live\.com)\b", re.IGNORECASE),
        "https://login.live.com/",
        "https://signup.live.com/",
        "https://www.microsoft.com/",
    ),
    (
        re.compile(r"\b(amazon)\b", re.IGNORECASE),
        "https://www.amazon.in/ap/signin",
        "https://www.amazon.in/ap/register",
        "https://www.amazon.in/",
    ),
    (
        re.compile(r"\b(flipkart)\b", re.IGNORECASE),
        "https://www.flipkart.com/account/login",
        "https://www.flipkart.com/account/login?signup=true",
        "https://www.flipkart.com/",
    ),
    (
        re.compile(r"\b(naukri)\b", re.IGNORECASE),
        "https://www.naukri.com/nlogin/login",
        "https://www.naukri.com/registration/createAccount",
        "https://www.naukri.com/",
    ),
    (
        re.compile(r"\b(indeed)\b", re.IGNORECASE),
        "https://secure.indeed.com/auth",
        "https://secure.indeed.com/account/register",
        "https://www.indeed.com/",
    ),
    (
        re.compile(r"\b(coursera)\b", re.IGNORECASE),
        "https://www.coursera.org/?authMode=login",
        "https://www.coursera.org/?authMode=signup",
        "https://www.coursera.org/",
    ),
    (
        re.compile(r"\b(udemy)\b", re.IGNORECASE),
        "https://www.udemy.com/join/login-popup/",
        "https://www.udemy.com/join/signup-popup/",
        "https://www.udemy.com/",
    ),
)


class FormFillerService:
    """Generic browser form-filler for Agent Mode.

    The service intentionally splits work into preview and fill phases. Preview
    may open/search a page and inspect visible fields; fill writes values only
    after Agent Mode confirmation. It never submits forms.
    """

    def __init__(self, settings: Settings, llm: LlmService):
        self.settings = settings
        self.llm = llm
        base = Path(settings.data_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        self.form_dir = base / "agent" / "form-filler"
        self.browser_profile_dir = self.form_dir / "browser-profile"
        self.browser_profile_dir.mkdir(parents=True, exist_ok=True)
        self._browser_lock = asyncio.Lock()
        self._playwright: Any = None
        self._browser_context: Any = None
        self._page: Any = None
        self._last_session: dict[str, Any] = {}

    async def preview(self, params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        task = self._clean_text(str(params.get("task") or params.get("input_text") or ""))
        continue_current = bool(params.get("continue_current")) or self._is_continue_request(task)
        direct_url = self._clean_url(str(params.get("url") or ""))
        site_query = self._clean_text(str(params.get("site_query") or ""))
        if continue_current and not direct_url and not site_query:
            direct_url = self._clean_url(str(self._last_session.get("url") or ""))
            if not task:
                task = self._clean_text(str(self._last_session.get("task") or "continue form fill"))
        if not continue_current and not direct_url and not site_query:
            site_query = self._search_query_from_task(task)
        if not direct_url and not site_query:
            return False, "Give Astra a form link or the site/page name to search.", {}

        flow_type = self._infer_flow_type(task)
        navigation: list[dict[str, Any]] = []
        page = await self._ensure_page()
        try:
            if continue_current and not direct_url:
                navigation.append({"action": "resume_current_page", "url": page.url})
            elif direct_url:
                await self._goto(page, direct_url)
                navigation.append({"action": "open_url", "url": page.url})
            else:
                await self._search_and_open(page, site_query, flow_type=flow_type)
                navigation.append({"action": "open_site", "query": site_query, "url": page.url})
            await self._handle_cookie_prompt(page)
        except Exception as exc:
            return False, f"Could not open the requested form page: {self._format_exception(exc)}", {}

        fields = await self.extract_fields()
        if not fields:
            discovered = await self._discover_and_open_form(page, task or site_query, flow_type, navigation)
            if discovered:
                fields = await self.extract_fields()
        if not fields:
            manual_state = await self._manual_state(page, [], assume_site_gate=flow_type in {"login", "create_account"})
            if manual_state["manual_required"]:
                preview = self._preview_payload(
                    page=page,
                    title=await self._safe_title(page),
                    fields=[],
                    mapped_values={},
                    blocked_fields=manual_state["blocked_fields"],
                    missing_required=[],
                    warnings=manual_state["warnings"],
                    flow_type=flow_type,
                    navigation=navigation,
                    manual_state=manual_state,
                )
                self._last_session = {
                    "url": page.url,
                    "task": task,
                    "site_query": site_query,
                    "flow_type": flow_type,
                    "field_values": {},
                }
                return (
                    True,
                    "Astra reached the page, but a human-only step is needed before safe fields can be filled.",
                    {"preview": preview, "params": {"url": page.url, "site_query": site_query, "task": task, "field_values": {}}},
                )
            return (
                False,
                "I opened the page, but I could not find a visible fillable form.",
                {"url": page.url, "title": await self._safe_title(page), "navigation": navigation},
            )

        mapped_values = self._coerce_field_values(params.get("field_values") or params.get("provided_values"))
        if not mapped_values:
            mapped_values = await self._map_values(task, fields)
        if self._wants_dummy_data(task):
            mapped_values = self._apply_dummy_values(task, fields, mapped_values)
        mapped_values = self._normalize_mapping(fields, mapped_values)

        blocked_fields = [self._field_summary(field) for field in fields if field.get("sensitive")]
        fillable_fields = [field for field in fields if not field.get("sensitive")]
        missing_required = [
            self._field_summary(field)
            for field in fillable_fields
            if field.get("required") and not self._has_value(mapped_values.get(str(field.get("key") or "")))
        ]
        manual_state = await self._manual_state(page, fields)
        warnings: list[str] = []
        if re.search(r"\b(submit|send|apply|register|sign\s*up|pay|purchase)\b", task, re.IGNORECASE):
            warnings.append("Astra will fill only. It will not submit or click final action buttons.")
        if blocked_fields:
            warnings.append("Sensitive/manual fields were detected. Astra will wait for the human there.")
        if self._wants_dummy_data(task):
            warnings.append("Dummy data is used only for safe non-sensitive fields.")
        warnings.extend(warning for warning in manual_state["warnings"] if warning not in warnings)

        title = await self._safe_title(page)
        preview = self._preview_payload(
            page=page,
            title=title,
            fields=fields,
            mapped_values=mapped_values,
            blocked_fields=blocked_fields,
            missing_required=missing_required,
            warnings=warnings,
            flow_type=flow_type,
            navigation=navigation,
            manual_state=manual_state,
        )
        fill_params = {
            "url": page.url,
            "site_query": site_query,
            "task": task,
            "field_values": mapped_values,
            "flow_type": flow_type,
        }
        self._last_session = dict(fill_params)
        return True, "Review the detected form values before Astra fills the page.", {"preview": preview, "params": fill_params}

    async def fill(self, params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        url = self._clean_url(str(params.get("url") or ""))
        if not url:
            return False, "Form fill needs the reviewed page URL.", {}
        field_values = self._coerce_field_values(params.get("field_values"))
        if not field_values:
            return False, "No field values were approved for filling.", {"url": url}

        page = await self._ensure_page()
        try:
            if self._canonical_url(page.url) != self._canonical_url(url):
                await self._goto(page, url)
        except Exception as exc:
            return False, f"Could not reopen the form page: {self._format_exception(exc)}", {"url": url}

        fields = await self.extract_fields()
        field_by_key = {str(field.get("key") or ""): field for field in fields}
        filled: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        for key, value in field_values.items():
            field = field_by_key.get(str(key))
            if not field:
                skipped.append({"key": key, "label": key, "reason": "field no longer visible"})
                continue
            if field.get("sensitive"):
                skipped.append({**self._field_summary(field), "reason": field.get("blocked_reason") or "sensitive field"})
                continue
            if not self._has_value(value):
                skipped.append({**self._field_summary(field), "reason": "empty value"})
                continue
            try:
                await self._fill_field(page, field, value)
                filled.append(self._field_summary(field))
            except Exception as exc:
                failed.append({**self._field_summary(field), "reason": self._format_exception(exc)})

        await page.wait_for_timeout(350)
        verified_fields = await self.extract_fields()
        verified = self._verify_values(verified_fields, field_values)
        manual_state = await self._manual_state(page, verified_fields)
        result = {
            "url": page.url,
            "title": await self._safe_title(page),
            "filled": filled,
            "skipped": skipped,
            "failed": failed,
            "verified": verified,
            "submit_blocked": True,
            "manual_required": manual_state["manual_required"],
            "waiting_for": manual_state["waiting_for"],
            "blocked_fields": manual_state["blocked_fields"],
            "warnings": manual_state["warnings"],
        }
        self._last_session = {
            "url": page.url,
            "task": str(params.get("task") or self._last_session.get("task") or ""),
            "site_query": str(params.get("site_query") or self._last_session.get("site_query") or ""),
            "flow_type": str(params.get("flow_type") or self._last_session.get("flow_type") or ""),
            "field_values": field_values,
        }
        if failed:
            return False, f"Filled {len(filled)} field(s), but {len(failed)} field(s) failed.", {"form_fill_result": result}
        if manual_state["manual_required"]:
            wait_text = ", ".join(manual_state["waiting_for"]) or "manual review"
            return (
                True,
                f"Filled {len(filled)} safe field(s). Waiting for human step: {wait_text}. Astra did not submit the form.",
                {"form_fill_result": result},
            )
        return (
            True,
            f"Filled {len(filled)} field(s). Review the browser page and submit manually if everything looks right.",
            {"form_fill_result": result},
        )

    def _preview_payload(
        self,
        *,
        page: Any,
        title: str,
        fields: list[dict[str, Any]],
        mapped_values: dict[str, Any],
        blocked_fields: list[dict[str, Any]],
        missing_required: list[dict[str, Any]],
        warnings: list[str],
        flow_type: str,
        navigation: list[dict[str, Any]],
        manual_state: dict[str, Any],
    ) -> dict[str, Any]:
        combined_blocked: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for item in [*blocked_fields, *manual_state["blocked_fields"]]:
            key = str(item.get("key") or item.get("label") or "")
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            combined_blocked.append(item)
        return {
            "url": page.url,
            "title": title,
            "fields": [self._public_field(field) for field in fields],
            "field_values": mapped_values,
            "missing_required": missing_required,
            "blocked_fields": combined_blocked,
            "warnings": warnings,
            "detected_count": len(fields),
            "mapped_count": len([value for value in mapped_values.values() if self._has_value(value)]),
            "flow_type": flow_type,
            "flow_label": self._flow_label(flow_type),
            "manual_required": bool(manual_state["manual_required"]),
            "waiting_for": manual_state["waiting_for"],
            "navigation": navigation,
        }

    async def extract_fields(self) -> list[dict[str, Any]]:
        page = await self._ensure_page()
        try:
            await page.wait_for_selector("input, textarea, select", timeout=5000)
        except Exception:
            pass
        fields: list[dict[str, Any]] = []
        frames = list(getattr(page, "frames", []) or [page])
        if not frames:
            frames = [page]
        for frame_index, frame in enumerate(frames):
            try:
                raw_fields = await frame.evaluate(FORM_FIELD_EXTRACTION_SCRIPT)
            except Exception:
                continue
            for field in raw_fields if isinstance(raw_fields, list) else []:
                if not isinstance(field, dict):
                    continue
                field["frame_index"] = frame_index
                field["frame_url"] = str(getattr(frame, "url", "") or "")
                fields.append(field)
        return self._postprocess_fields(fields)

    def _postprocess_fields(self, fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: dict[str, int] = {}
        processed: list[dict[str, Any]] = []
        for index, field in enumerate(fields):
            label = self._clean_text(str(field.get("label") or field.get("placeholder") or field.get("name") or f"Field {index + 1}"))
            key_base = self._slug(label or str(field.get("name") or f"field_{index + 1}"))
            if not key_base:
                key_base = f"field_{index + 1}"
            duplicate_count = seen.get(key_base, 0)
            seen[key_base] = duplicate_count + 1
            key = key_base if duplicate_count == 0 else f"{key_base}_{duplicate_count + 1}"
            haystack = " ".join(
                str(field.get(name) or "")
                for name in ("label", "name", "id", "placeholder", "type", "autocomplete", "aria_label")
            )
            sensitive = bool(SENSITIVE_FIELD_RE.search(haystack)) or str(field.get("type") or "").lower() in {"password", "file"}
            blocked_reason = "sensitive field" if sensitive else ""
            if str(field.get("type") or "").lower() == "file":
                blocked_reason = "file upload needs manual selection"
            processed.append(
                {
                    **field,
                    "key": key,
                    "label": label,
                    "sensitive": sensitive,
                    "blocked_reason": blocked_reason,
                    "required": bool(field.get("required")),
                }
            )
        return processed

    async def _manual_state(self, page: Any, fields: list[dict[str, Any]], assume_site_gate: bool = False) -> dict[str, Any]:
        try:
            raw = await page.evaluate(MANUAL_STATE_SCRIPT)
        except Exception:
            raw = {}
        signals = raw if isinstance(raw, dict) else {}
        waiting_for: list[str] = []
        blocked_fields = [self._field_summary(field) for field in fields if field.get("sensitive")]

        def add(kind: str, label: str, reason: str) -> None:
            if label not in waiting_for:
                waiting_for.append(label)
            if not any(item.get("key") == kind for item in blocked_fields):
                blocked_fields.append({"key": kind, "label": label, "required": False, "control_type": "manual", "reason": reason})

        if any(self._normalize(str(field.get("label") or field.get("name") or field.get("type") or "")).find("password") >= 0 for field in fields if field.get("sensitive")):
            add("password", "Password", "password entry must be completed by the human")
        if bool(signals.get("hasCaptcha")):
            add("captcha", "CAPTCHA or bot check", "human verification must be completed by the human")
        if bool(signals.get("hasOtp")):
            add("otp", "OTP or verification code", "one-time code must be completed by the human")
        if bool(signals.get("hasPayment")):
            add("payment", "Payment or banking field", "payment and banking details are blocked")
        if bool(signals.get("hasFileUpload")):
            add("file_upload", "File upload", "file selection must be completed by the human")
        if bool(signals.get("hasLegalAttestation")):
            add("legal_attestation", "Legal attestation", "legal declarations must be completed by the human")
        if bool(signals.get("hasAccessBlock")):
            add("site_gate", "Site security or compatibility gate", "the site is blocking automation or needs a human browser step")
        if assume_site_gate and not blocked_fields:
            add("site_gate", "Site security or compatibility gate", "the site did not expose fillable fields to automation")
        submit_actions = signals.get("submitActions") if isinstance(signals.get("submitActions"), list) else []
        if submit_actions:
            add("final_submit", "Final submit", "Astra fills only; final submission needs human approval")

        warnings: list[str] = []
        if blocked_fields:
            warnings.append("Human-only fields or final actions are present. Astra will pause there.")
        if submit_actions:
            warnings.append("Final submission is intentionally blocked.")
        return {
            "manual_required": bool(blocked_fields),
            "waiting_for": waiting_for,
            "blocked_fields": blocked_fields,
            "warnings": warnings,
            "signals": signals,
        }

    async def _map_values(self, task: str, fields: list[dict[str, Any]]) -> dict[str, Any]:
        local_values = self._map_values_locally(task, fields)
        if not self.llm.model_configured():
            return local_values
        safe_fields = [self._public_field(field) for field in fields if not field.get("sensitive")]
        if not safe_fields:
            return {}
        system_prompt = (
            "You map user-provided text to web form fields. Return strict JSON only. "
            "Never invent personal data. Use only values explicitly present in the prompt. "
            "Skip passwords, OTP, CAPTCHA, payment, bank, token, legal attestation, and file upload fields. "
            "Use field keys exactly as provided."
        )
        user_prompt = json.dumps(
            {
                "prompt": task,
                "fields": safe_fields,
                "local_guess": local_values,
                "output_schema": {
                    "field_values": {"field_key": "value from prompt"},
                    "missing_required": ["field_key"],
                    "warnings": ["short string"],
                },
            },
            ensure_ascii=True,
        )
        try:
            raw, setup = await self.llm.complete(system_prompt, user_prompt, model="fast")
        except Exception:
            return local_values
        if setup:
            return local_values
        payload = self._parse_json(raw)
        if not payload:
            return local_values
        llm_values = self._coerce_field_values(payload.get("field_values"))
        return {**local_values, **llm_values}

    def _map_values_locally(self, task: str, fields: list[dict[str, Any]]) -> dict[str, Any]:
        explicit_values = self._extract_explicit_values(task)
        mapped: dict[str, Any] = {}
        for field in fields:
            if field.get("sensitive"):
                continue
            key = str(field.get("key") or "")
            haystack = self._normalize(" ".join(str(field.get(name) or "") for name in ("label", "name", "placeholder", "id", "autocomplete")))
            control_type = str(field.get("control_type") or field.get("type") or "").lower()

            value = self._value_for_field(haystack, explicit_values)
            if value is None and control_type in {"select", "radio"}:
                value = self._option_from_prompt(task, field)
            if value is None and control_type == "checkbox":
                value = self._checkbox_value_from_prompt(task, field)
            if value is not None:
                mapped[key] = value
        return mapped

    def _extract_explicit_values(self, text: str) -> dict[str, str]:
        values: dict[str, str] = {}
        normalized_labels = {
            "name": ("name", "full name", "your name"),
            "first_name": ("first name", "firstname", "given name"),
            "last_name": ("last name", "lastname", "surname"),
            "email": ("email", "e-mail", "mail"),
            "phone": ("phone", "mobile", "contact number", "phone number", "number"),
            "address": ("address", "location"),
            "city": ("city",),
            "state": ("state", "province"),
            "country": ("country",),
            "message": ("message", "comment", "comments", "note", "description"),
            "company": ("company", "organization", "organisation"),
            "college": ("college", "school", "university", "institute"),
            "course": ("course", "program", "programme"),
        }
        for canonical, aliases in normalized_labels.items():
            for alias in aliases:
                pattern = re.compile(
                    rf"\b{re.escape(alias)}\b\s*(?:is|=|:|-)\s*(?P<value>[^,;\n]+)",
                    re.IGNORECASE,
                )
                match = pattern.search(text)
                if match:
                    values[canonical] = self._clean_value(match.group("value"))
                    break
        name_match = re.search(r"\bmy\s+name\s+is\s+(?P<value>[^,;\n]+)", text, re.IGNORECASE)
        if name_match and "name" not in values:
            values["name"] = self._clean_value(name_match.group("value"))
        email_match = re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", text)
        if email_match and "email" not in values:
            values["email"] = email_match.group(0)
        phone_match = re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", text)
        if phone_match and "phone" not in values:
            values["phone"] = self._clean_value(phone_match.group(0))
        return {key: value for key, value in values.items() if value}

    def _value_for_field(self, haystack: str, values: dict[str, str]) -> str | None:
        field_aliases = {
            "first_name": ("first name", "firstname", "given name"),
            "last_name": ("last name", "lastname", "surname"),
            "email": ("email", "e-mail", "mail"),
            "phone": ("phone", "mobile", "contact number", "phone number", "number"),
            "address": ("address", "street", "location"),
            "city": ("city",),
            "state": ("state", "province"),
            "country": ("country",),
            "message": ("message", "comment", "comments", "note", "description", "query"),
            "company": ("company", "organization", "organisation"),
            "college": ("college", "school", "university", "institute"),
            "course": ("course", "program", "programme"),
            "name": ("name", "full name", "your name"),
        }
        for canonical, aliases in field_aliases.items():
            if canonical not in values:
                continue
            if any(alias in haystack for alias in aliases):
                return values[canonical]
        return None

    def _option_from_prompt(self, task: str, field: dict[str, Any]) -> str | None:
        normalized_task = self._normalize(task)
        for option in field.get("options") or []:
            label = self._normalize(str(option.get("label") or option.get("value") or ""))
            value = str(option.get("value") or option.get("label") or "").strip()
            if label and re.search(rf"\b{re.escape(label)}\b", normalized_task):
                return value
        return None

    def _checkbox_value_from_prompt(self, task: str, field: dict[str, Any]) -> bool | None:
        label = self._normalize(str(field.get("label") or ""))
        if not label:
            return None
        normalized_task = self._normalize(task)
        if label in normalized_task:
            if re.search(r"\b(no|not|do not|don't|uncheck|false)\b", normalized_task):
                return False
            if re.search(r"\b(yes|check|agree|subscribe|true|enable)\b", normalized_task):
                return True
        return None

    def _wants_dummy_data(self, task: str) -> bool:
        return bool(re.search(r"\b(dummy|fake|sample|test\s+data|random)\b", task, re.IGNORECASE))

    def _apply_dummy_values(self, task: str, fields: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, Any]:
        values = dict(current)
        for field in fields:
            key = str(field.get("key") or "")
            if not key or field.get("sensitive") or self._has_value(values.get(key)):
                continue
            dummy = self._dummy_value_for_field(task, field)
            if self._has_value(dummy):
                values[key] = dummy
        return values

    def _dummy_value_for_field(self, task: str, field: dict[str, Any]) -> Any:
        haystack = self._normalize(
            " ".join(str(field.get(name) or "") for name in ("label", "name", "placeholder", "id", "autocomplete", "type"))
        )
        control_type = str(field.get("control_type") or field.get("type") or "").lower()
        if control_type == "checkbox":
            if re.search(r"\b(terms|privacy|agree|consent|declaration|attestation)\b", haystack):
                return None
            return False
        if control_type in {"select", "radio"}:
            options = [
                option
                for option in field.get("options") or []
                if self._normalize(str(option.get("label") or option.get("value") or "")) not in {"", "choose one", "select", "select one", "none"}
            ]
            if options:
                first = options[0]
                return str(first.get("value") or first.get("label") or "")
            return None
        if "email" in haystack or "e mail" in haystack:
            return "dummy@example.com"
        if "phone" in haystack or "mobile" in haystack or "contact number" in haystack:
            return "9876543210"
        if "first name" in haystack or "given name" in haystack:
            return "Demo"
        if "last name" in haystack or "surname" in haystack:
            return "User"
        if "full name" in haystack or re.search(r"\bname\b", haystack):
            return "Demo User"
        if "username" in haystack or "user id" in haystack or "identifier" in haystack:
            return "demo.user"
        if "city" in haystack:
            return "New Delhi"
        if "state" in haystack or "province" in haystack:
            return "Delhi"
        if "country" in haystack:
            return "India"
        if "address" in haystack or "location" in haystack:
            return "Demo address"
        if "company" in haystack or "organization" in haystack or "organisation" in haystack:
            return "Demo Company"
        if "message" in haystack or "comment" in haystack or "description" in haystack or "query" in haystack:
            return "This is dummy data for testing the form filler."
        if field.get("required"):
            return "Dummy"
        return None

    def _normalize_mapping(self, fields: list[dict[str, Any]], values: dict[str, Any]) -> dict[str, Any]:
        allowed = {str(field.get("key") or ""): field for field in fields}
        clean: dict[str, Any] = {}
        for key, value in values.items():
            field = allowed.get(str(key))
            if not field or field.get("sensitive"):
                continue
            if isinstance(value, bool):
                clean[str(key)] = value
            else:
                clean[str(key)] = self._clean_value(str(value))[:500]
        return clean

    async def _fill_field(self, page: Any, field: dict[str, Any], value: Any) -> None:
        control_type = str(field.get("control_type") or field.get("type") or "text").lower()
        selector = str(field.get("selector") or "")
        context = self._field_context(page, field)
        if control_type == "select":
            locator = context.locator(selector).first
            option_value = self._resolve_option_value(field, str(value))
            try:
                await locator.select_option(value=option_value, timeout=5000)
            except Exception:
                await locator.select_option(label=str(value), timeout=5000)
            return
        if control_type == "radio":
            option_selector = self._radio_option_selector(field, str(value))
            if option_selector:
                await context.locator(option_selector).first.check(timeout=5000)
                return
            await context.get_by_label(re.compile(re.escape(str(value)), re.IGNORECASE)).first.check(timeout=5000)
            return
        if control_type == "checkbox":
            await context.locator(selector).first.set_checked(bool(value), timeout=5000)
            return
        locator = context.locator(selector).first
        try:
            await locator.fill(str(value), timeout=8000)
        except Exception as exc:
            message = str(exc).lower()
            if "not editable" not in message and "element is not editable" not in message:
                raise
            await locator.click(timeout=5000, force=True)
            await page.keyboard.press("Control+A")
            await page.keyboard.type(str(value), delay=15)

    def _field_context(self, page: Any, field: dict[str, Any]) -> Any:
        try:
            frame_index = int(field.get("frame_index", 0))
        except (TypeError, ValueError):
            frame_index = 0
        frames = list(getattr(page, "frames", []) or [])
        if 0 <= frame_index < len(frames):
            return frames[frame_index]
        return page

    def _resolve_option_value(self, field: dict[str, Any], value: str) -> str:
        normalized = self._normalize(value)
        for option in field.get("options") or []:
            label = self._normalize(str(option.get("label") or ""))
            option_value = self._normalize(str(option.get("value") or ""))
            if normalized in {label, option_value}:
                return str(option.get("value") or option.get("label") or value)
        return value

    def _radio_option_selector(self, field: dict[str, Any], value: str) -> str:
        normalized = self._normalize(value)
        for option in field.get("options") or []:
            label = self._normalize(str(option.get("label") or ""))
            option_value = self._normalize(str(option.get("value") or ""))
            if normalized in {label, option_value}:
                return str(option.get("selector") or "")
        return ""

    def _verify_values(self, fields: list[dict[str, Any]], expected_values: dict[str, Any]) -> list[dict[str, Any]]:
        by_key = {str(field.get("key") or ""): field for field in fields}
        verified: list[dict[str, Any]] = []
        for key, expected in expected_values.items():
            field = by_key.get(str(key))
            if not field or field.get("sensitive") or not self._has_value(expected):
                continue
            actual = field.get("value")
            control_type = str(field.get("control_type") or "").lower()
            if control_type == "checkbox":
                ok = bool(actual) == bool(expected)
            else:
                ok = self._normalize(str(expected)) in self._normalize(str(actual))
            verified.append({**self._field_summary(field), "ok": ok, "expected": expected, "actual": actual})
        return verified

    async def _goto(self, page: Any, url: str) -> None:
        self._assert_safe_url(url)
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        except Exception as exc:
            if "ERR_ABORTED" not in str(exc):
                raise
            await page.wait_for_timeout(1500)
            current_url = str(getattr(page, "url", "") or "")
            if not current_url.startswith("http"):
                raise
        await page.wait_for_timeout(600)

    async def _search_and_open(self, page: Any, query: str, flow_type: str = "") -> None:
        known_url = self._known_site_url(query, flow_type=flow_type)
        if known_url:
            await self._goto(page, known_url)
            return
        search_query = query
        if "form" not in self._normalize(search_query):
            search_query = f"{search_query} form"
        await page.goto(f"https://www.google.com/search?q={quote_plus(search_query)}", wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(1200)
        candidates = await page.evaluate(SEARCH_RESULT_SCRIPT)
        for candidate in candidates if isinstance(candidates, list) else []:
            url = str(candidate.get("url") or "")
            if not url or not self._is_safe_public_result(url):
                continue
            await self._goto(page, url)
            return
        raise ValueError("No safe search result was found for the requested form page.")

    async def _discover_and_open_form(self, page: Any, task: str, flow_type: str, navigation: list[dict[str, Any]]) -> bool:
        for depth in range(2):
            await self._handle_cookie_prompt(page)
            try:
                raw_candidates = await page.evaluate(FORM_LINK_DISCOVERY_SCRIPT)
            except Exception:
                raw_candidates = []
            candidates = raw_candidates if isinstance(raw_candidates, list) else []
            ranked = sorted(
                (
                    {**candidate, "score": self._score_form_candidate(candidate, task, flow_type)}
                    for candidate in candidates
                    if isinstance(candidate, dict)
                ),
                key=lambda item: int(item.get("score") or 0),
                reverse=True,
            )
            ranked = [candidate for candidate in ranked if int(candidate.get("score") or 0) >= 25][:5]
            if not ranked:
                return False
            for candidate in ranked:
                before_url = page.url
                try:
                    url = self._clean_url(str(candidate.get("url") or ""))
                    selector = str(candidate.get("selector") or "")
                    if url and self._canonical_url(url) != self._canonical_url(before_url):
                        await self._goto(page, url)
                    elif selector:
                        await page.locator(selector).first.click(timeout=6000)
                        try:
                            await page.wait_for_load_state("domcontentloaded", timeout=12000)
                        except Exception:
                            pass
                        await page.wait_for_timeout(900)
                    else:
                        continue
                    await self._handle_cookie_prompt(page)
                    navigation.append(
                        {
                            "action": "discover_form",
                            "depth": depth + 1,
                            "label": self._clean_text(str(candidate.get("text") or ""))[:80],
                            "url": page.url,
                            "score": int(candidate.get("score") or 0),
                        }
                    )
                    fields = await self.extract_fields()
                    if fields:
                        return True
                    manual_state = await self._manual_state(page, [], assume_site_gate=flow_type in {"login", "create_account"})
                    if manual_state["manual_required"]:
                        return True
                except Exception as exc:
                    navigation.append(
                        {
                            "action": "discover_form_failed",
                            "label": self._clean_text(str(candidate.get("text") or ""))[:80],
                            "error": self._format_exception(exc),
                        }
                    )
                    try:
                        if page.url != before_url:
                            await self._goto(page, before_url)
                    except Exception:
                        pass
                    continue
        return False

    async def _handle_cookie_prompt(self, page: Any) -> None:
        try:
            raw_candidates = await page.evaluate(COOKIE_PROMPT_SCRIPT)
        except Exception:
            return
        candidates = raw_candidates if isinstance(raw_candidates, list) else []
        for candidate in candidates[:3]:
            if not isinstance(candidate, dict):
                continue
            selector = str(candidate.get("selector") or "")
            if not selector:
                continue
            try:
                await page.locator(selector).first.click(timeout=2500)
                await page.wait_for_timeout(450)
                return
            except Exception:
                continue

    def _score_form_candidate(self, candidate: dict[str, Any], task: str, flow_type: str) -> int:
        text = self._normalize(" ".join(str(candidate.get(name) or "") for name in ("text", "url", "aria_label", "title")))
        if not text:
            return 0
        score = 0
        intent_keywords = {
            "login": ("login", "log in", "sign in", "signin", "account", "continue with email"),
            "create_account": ("sign up", "signup", "create account", "register", "registration", "join", "get started", "start free"),
            "contact_form": ("contact", "support", "request demo", "talk to sales", "inquiry", "enquiry", "help"),
            "job_application": ("apply", "careers", "jobs", "positions", "openings", "candidate", "resume"),
            "admission_form": ("admission", "enroll", "enrol", "application", "course", "program"),
            "survey": ("survey", "feedback", "questionnaire", "review"),
            "generic_form": ("form", "apply", "register", "contact", "sign up", "login", "start"),
        }
        for keyword in intent_keywords.get(flow_type, intent_keywords["generic_form"]):
            if keyword in text:
                score += 45
        for keyword in intent_keywords["generic_form"]:
            if keyword in text:
                score += 12
        normalized_task = self._normalize(task)
        for token in normalized_task.split():
            if len(token) >= 5 and token in text:
                score += 3
        if re.search(r"\b(pricing|privacy|terms|policy|blog|news|docs|documentation|download|learn|about|investors)\b", text):
            score -= 35
        if re.search(r"\b(logout|sign out|delete|remove|unsubscribe|purchase|buy now|checkout|cart)\b", text):
            score -= 60
        if str(candidate.get("tag") or "").lower() == "button":
            score += 4
        return max(0, min(score, 100))

    def _known_site_url(self, query: str, flow_type: str = "") -> str:
        wants_login = flow_type == "login" or bool(re.search(r"\b(login|log\s*in|sign\s*in|signin|account)\b", query, re.IGNORECASE))
        wants_signup = flow_type == "create_account" or bool(
            re.search(r"\b(sign\s*up|signup|create\s+(?:an\s+)?(?:\w+\s+){0,4}account|new\s+account|register|registration)\b", query, re.IGNORECASE)
        )
        for pattern, login_url, signup_url, home_url in KNOWN_SITE_PATTERNS:
            if pattern.search(query):
                if wants_signup:
                    return signup_url
                return login_url if wants_login else home_url
        return ""

    async def _ensure_page(self):
        async with self._browser_lock:
            try:
                if self._page and not self._page.is_closed():
                    return self._page
            except Exception:
                self._page = None
            try:
                from playwright.async_api import async_playwright
            except Exception as exc:
                raise RuntimeError(
                    "Playwright is not installed. Install it with `pip install playwright` and `python -m playwright install chromium`."
                ) from exc
            if not self._playwright:
                self._playwright = await async_playwright().start()
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    if not self._browser_context:
                        self._browser_context = await self._playwright.chromium.launch_persistent_context(
                            user_data_dir=str(self.browser_profile_dir),
                            headless=False,
                        )
                    pages = [page for page in self._browser_context.pages if not page.is_closed()]
                    self._page = pages[0] if pages else await self._browser_context.new_page()
                    return self._page
                except Exception as exc:
                    last_error = exc
                    await self._reset_browser_handles()
                    if attempt == 0:
                        continue
                    break
            raise RuntimeError(f"Could not open the form filler browser: {self._format_exception(last_error)}") from last_error

    async def _reset_browser_handles(self) -> None:
        context = self._browser_context
        self._page = None
        self._browser_context = None
        if context:
            try:
                await context.close()
            except Exception:
                pass

    def _coerce_field_values(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        return {str(key): value for key, value in raw.items()}

    def _is_continue_request(self, task: str) -> bool:
        return bool(re.search(r"\b(continue|resume|carry\s+on|go\s+on|next\s+step|i\s+handled|done\s+manually)\b", task, re.IGNORECASE))

    def _infer_flow_type(self, task: str) -> str:
        normalized = self._normalize(task)
        if re.search(r"\b(login|log in|sign in|signin)\b", normalized):
            return "login"
        if re.search(r"\b(sign up|signup|create (?:an )?(?:[a-z0-9]+ ){0,4}account|new account|register|registration|join)\b", normalized):
            return "create_account"
        if re.search(r"\b(job|career|careers|apply|application|resume|cv|opening|position)\b", normalized):
            return "job_application"
        if re.search(r"\b(admission|enroll|enrol|course application|college|university|school)\b", normalized):
            return "admission_form"
        if re.search(r"\b(contact|support|inquiry|enquiry|request demo|talk to sales|message)\b", normalized):
            return "contact_form"
        if re.search(r"\b(survey|feedback|questionnaire|review form)\b", normalized):
            return "survey"
        return "generic_form"

    def _flow_label(self, flow_type: str) -> str:
        labels = {
            "login": "Login",
            "create_account": "Create account",
            "job_application": "Job application",
            "admission_form": "Admission form",
            "contact_form": "Contact form",
            "survey": "Survey",
            "generic_form": "Generic form",
        }
        return labels.get(flow_type, "Generic form")

    def _public_field(self, field: dict[str, Any]) -> dict[str, Any]:
        return {
            "key": field.get("key"),
            "label": field.get("label"),
            "control_type": field.get("control_type"),
            "type": field.get("type"),
            "required": bool(field.get("required")),
            "options": field.get("options") or [],
            "sensitive": bool(field.get("sensitive")),
            "blocked_reason": field.get("blocked_reason") or "",
            "value": field.get("value"),
            "frame_url": field.get("frame_url") or "",
        }

    def _field_summary(self, field: dict[str, Any]) -> dict[str, Any]:
        return {
            "key": field.get("key"),
            "label": field.get("label"),
            "required": bool(field.get("required")),
            "control_type": field.get("control_type"),
            "reason": field.get("blocked_reason") or "",
        }

    def _clean_url(self, url: str) -> str:
        cleaned = url.strip().rstrip(".,)")
        return cleaned if cleaned else ""

    def _assert_safe_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Only http(s) URLs can be opened for form filling.")
        host = parsed.hostname or ""
        if host in {"localhost", "127.0.0.1", "::1"}:
            return
        if host.startswith("10.") or host.startswith("192.168.") or re.match(r"^172\.(1[6-9]|2\d|3[0-1])\.", host):
            raise ValueError("Private network form pages are not enabled for generic form filling.")

    def _is_safe_public_result(self, url: str) -> bool:
        try:
            self._assert_safe_url(url)
        except Exception:
            return False
        return True

    def _search_query_from_task(self, task: str) -> str:
        cleaned = re.sub(
            r"\b(please|open|go to|visit|fill|complete|populate|form|application|registration|sign up|signup|register|with|using|my)\b",
            " ",
            task,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"https?://\S+", " ", cleaned)
        cleaned = re.sub(r"\b(name|email|phone|mobile|message|address|city|state|country)\s*(?:is|=|:|-)\s*[^,;\n]+", " ", cleaned, flags=re.IGNORECASE)
        return self._clean_text(cleaned)[:180]

    def _clean_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip(" \t\r\n")

    def _clean_value(self, value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip(" .;\t\r\n")
        value = re.sub(r"\b(and then|then click|then submit|submit it|send it)\b.*$", "", value, flags=re.IGNORECASE).strip()
        return value

    def _normalize(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _slug(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:48]

    def _has_value(self, value: Any) -> bool:
        if isinstance(value, bool):
            return True
        return bool(str(value or "").strip())

    def _canonical_url(self, url: str) -> str:
        parsed = urlparse(url or "")
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")

    async def _safe_title(self, page: Any) -> str:
        try:
            return self._clean_text(await page.title())
        except Exception:
            return ""

    def _parse_json(self, raw: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except Exception:
                return None
        return payload if isinstance(payload, dict) else None

    def _format_exception(self, exc: BaseException | None) -> str:
        if exc is None:
            return "Unknown error."
        message = re.sub(r"\s+", " ", str(exc)).strip()
        return (message or type(exc).__name__)[-600:]


FORM_FIELD_EXTRACTION_SCRIPT = r"""
() => {
  const clean = (value) => (value || "").replace(/\s+/g, " ").trim();
  const cssEscape = (value) => {
    if (window.CSS && CSS.escape) return CSS.escape(value);
    return String(value).replace(/["\\]/g, "\\$&");
  };
  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
  };
  const nearbyText = (el) => {
    const pieces = [];
    const parent = el.closest("label, .form-group, .field, .input, p, div, li, td, tr");
    if (parent) pieces.push(parent.innerText || "");
    let previous = el.previousElementSibling;
    let guard = 0;
    while (previous && guard < 3) {
      pieces.push(previous.innerText || previous.textContent || "");
      previous = previous.previousElementSibling;
      guard += 1;
    }
    return clean(pieces.join(" ")).slice(0, 180);
  };
  const labelOwnText = (label, control) => {
    const clone = label.cloneNode(true);
    clone.querySelectorAll("input, textarea, select, option, button, datalist").forEach((node) => node.remove());
    return clean(clone.textContent || "");
  };
  const labelFor = (el) => {
    const id = el.getAttribute("id") || "";
    const ariaLabel = el.getAttribute("aria-label") || "";
    const labelledBy = el.getAttribute("aria-labelledby") || "";
    const labels = [];
    if (id) {
      const label = document.querySelector(`label[for="${cssEscape(id)}"]`);
      if (label) labels.push(labelOwnText(label, el) || label.innerText || label.textContent || "");
    }
    const closestLabel = el.closest("label");
    if (closestLabel) labels.push(labelOwnText(closestLabel, el) || closestLabel.innerText || closestLabel.textContent || "");
    if (labelledBy) {
      for (const part of labelledBy.split(/\s+/)) {
        const node = document.getElementById(part);
        if (node) labels.push(node.innerText || node.textContent || "");
      }
    }
    labels.push(ariaLabel, el.getAttribute("placeholder") || "", el.getAttribute("title") || "");
    if (!labels.some((value) => clean(value))) labels.push(nearbyText(el));
    const unique = [];
    const seen = new Set();
    for (const value of labels) {
      const item = clean(value);
      const key = item.toLowerCase();
      if (!item || seen.has(key)) continue;
      seen.add(key);
      unique.push(item);
    }
    const label = clean(unique.join(" "));
    return label || clean(el.getAttribute("name") || el.getAttribute("id") || "");
  };
  const baseSelector = (el, index) => {
    const tag = el.tagName.toLowerCase();
    const id = el.getAttribute("id");
    const name = el.getAttribute("name");
    if (id) return `${tag}#${cssEscape(id)}`;
    if (name && el.type !== "radio") return `${tag}[name="${cssEscape(name)}"]`;
    const marker = `astra-form-field-${index}`;
    el.setAttribute("data-astra-form-field", marker);
    return `[data-astra-form-field="${marker}"]`;
  };
  const controls = Array.from(document.querySelectorAll("input, textarea, select"))
    .filter((el) => !el.disabled && isVisible(el))
    .filter((el) => !["hidden", "submit", "button", "reset", "image"].includes((el.getAttribute("type") || "").toLowerCase()));
  const out = [];
  const radioGroups = new Map();
  controls.forEach((el, index) => {
    const tag = el.tagName.toLowerCase();
    const type = tag === "textarea" ? "textarea" : tag === "select" ? "select" : (el.getAttribute("type") || "text").toLowerCase();
    if (type === "radio") {
      const name = el.getAttribute("name") || `radio-${index}`;
      if (!radioGroups.has(name)) radioGroups.set(name, []);
      radioGroups.get(name).push({ el, index });
      return;
    }
    const selector = baseSelector(el, index);
    const field = {
      id: el.getAttribute("id") || "",
      name: el.getAttribute("name") || "",
      label: labelFor(el),
      placeholder: el.getAttribute("placeholder") || "",
      aria_label: el.getAttribute("aria-label") || "",
      autocomplete: el.getAttribute("autocomplete") || "",
      type,
      control_type: tag === "select" ? "select" : type === "checkbox" ? "checkbox" : tag === "textarea" ? "textarea" : "text",
      required: Boolean(el.required || el.getAttribute("aria-required") === "true"),
      selector,
      value: type === "checkbox" ? Boolean(el.checked) : el.value || "",
      options: [],
    };
    if (tag === "select") {
      field.options = Array.from(el.options).map((option) => ({ label: clean(option.textContent || ""), value: option.value || clean(option.textContent || "") })).filter((option) => option.label || option.value);
    }
    out.push(field);
  });
  for (const [name, items] of radioGroups.entries()) {
    const first = items[0].el;
    const fieldset = first.closest("fieldset");
    const groupLabel = clean((fieldset && fieldset.querySelector("legend")?.innerText) || labelFor(first) || name);
    out.push({
      id: first.getAttribute("id") || "",
      name,
      label: groupLabel,
      placeholder: "",
      aria_label: "",
      autocomplete: "",
      type: "radio",
      control_type: "radio",
      required: items.some((item) => item.el.required || item.el.getAttribute("aria-required") === "true"),
      selector: `input[type="radio"][name="${cssEscape(name)}"]`,
      value: (items.find((item) => item.el.checked)?.el.value || ""),
      options: items.map((item) => ({
        label: labelFor(item.el),
        value: item.el.value || labelFor(item.el),
        selector: baseSelector(item.el, item.index),
      })),
    });
  }
  return out;
}
"""


MANUAL_STATE_SCRIPT = r"""
() => {
  const clean = (value) => (value || "").replace(/\s+/g, " ").trim();
  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
  };
  const bodyText = clean(`${document.title || ""} ${document.body?.innerText || ""}`).slice(0, 12000);
  const lower = bodyText.toLowerCase();
  const inputs = Array.from(document.querySelectorAll("input, textarea, select")).filter((el) => !el.disabled && isVisible(el));
  const hasCaptcha = /captcha|recaptcha|hcaptcha|i am not a robot|verify you are human|security check|bot check/i.test(bodyText)
    || Array.from(document.querySelectorAll("iframe[src], div[class], div[id]")).some((el) => /recaptcha|hcaptcha|captcha|turnstile/i.test((el.getAttribute("src") || "") + " " + (el.getAttribute("class") || "") + " " + (el.getAttribute("id") || "")));
  const hasOtp = /otp|one[-\s]?time|verification code|security code|two[-\s]?factor|2fa|authenticator/i.test(bodyText);
  const hasPayment = /payment|credit card|debit card|card number|cvv|cvc|upi|bank account|billing/i.test(bodyText)
    || inputs.some((el) => /cc-|credit|card|cvv|cvc|upi|bank/i.test((el.getAttribute("name") || "") + " " + (el.getAttribute("id") || "") + " " + (el.getAttribute("autocomplete") || "")));
  const hasFileUpload = inputs.some((el) => (el.getAttribute("type") || "").toLowerCase() === "file");
  const hasLegalAttestation = /declaration|attestation|signature|i certify|i agree under penalty|terms and conditions/i.test(bodyText);
  const hasAccessBlock = /just a moment|access denied|temporarily blocked|enable javascript|unsupported browser|browser is not supported|something went wrong|try again later|robot check|checking if the site connection is secure|unusual traffic/i.test(bodyText);
  const actions = [];
  for (const el of Array.from(document.querySelectorAll("button, input[type='submit'], input[type='button'], a[role='button']"))) {
    if (!isVisible(el)) continue;
    const text = clean(el.innerText || el.textContent || el.value || el.getAttribute("aria-label") || el.getAttribute("title") || "");
    if (!text) continue;
    if (/^(submit|send|apply|register|sign up|create account|continue|next|save|finish|place order|pay|confirm)$/i.test(text)) {
      actions.push(text.slice(0, 80));
    }
    if (actions.length >= 6) break;
  }
  return { hasCaptcha, hasOtp, hasPayment, hasFileUpload, hasLegalAttestation, hasAccessBlock, submitActions: actions, textSample: lower.slice(0, 400) };
}
"""


FORM_LINK_DISCOVERY_SCRIPT = r"""
() => {
  const clean = (value) => (value || "").replace(/\s+/g, " ").trim();
  const cssEscape = (value) => {
    if (window.CSS && CSS.escape) return CSS.escape(value);
    return String(value).replace(/["\\]/g, "\\$&");
  };
  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
  };
  const selectorFor = (el, index) => {
    const tag = el.tagName.toLowerCase();
    const id = el.getAttribute("id");
    const name = el.getAttribute("name");
    if (id) return `${tag}#${cssEscape(id)}`;
    if (name) return `${tag}[name="${cssEscape(name)}"]`;
    const marker = `astra-form-candidate-${index}`;
    el.setAttribute("data-astra-form-candidate", marker);
    return `[data-astra-form-candidate="${marker}"]`;
  };
  const candidates = [];
  const nodes = Array.from(document.querySelectorAll("a[href], button, [role='button'], input[type='button'], input[type='submit']"));
  nodes.forEach((el, index) => {
    if (!isVisible(el)) return;
    const text = clean(el.innerText || el.textContent || el.value || el.getAttribute("aria-label") || el.getAttribute("title") || "");
    const href = el.href || el.getAttribute("href") || "";
    const url = href && /^https?:\/\//i.test(href) ? href : "";
    if (!text && !url) return;
    if (!/(login|log in|sign in|signin|sign up|signup|register|join|account|get started|start|contact|support|request demo|talk to sales|apply|application|admission|enroll|career|job|survey|feedback|form)/i.test(`${text} ${url}`)) return;
    candidates.push({
      text,
      url,
      selector: selectorFor(el, index),
      tag: el.tagName.toLowerCase(),
      aria_label: el.getAttribute("aria-label") || "",
      title: el.getAttribute("title") || "",
    });
  });
  return candidates.slice(0, 40);
}
"""


COOKIE_PROMPT_SCRIPT = r"""
() => {
  const clean = (value) => (value || "").replace(/\s+/g, " ").trim();
  const cssEscape = (value) => {
    if (window.CSS && CSS.escape) return CSS.escape(value);
    return String(value).replace(/["\\]/g, "\\$&");
  };
  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
  };
  const selectorFor = (el, index) => {
    const tag = el.tagName.toLowerCase();
    const id = el.getAttribute("id");
    if (id) return `${tag}#${cssEscape(id)}`;
    const marker = `astra-cookie-action-${index}`;
    el.setAttribute("data-astra-cookie-action", marker);
    return `[data-astra-cookie-action="${marker}"]`;
  };
  const buttons = Array.from(document.querySelectorAll("button, [role='button'], input[type='button'], input[type='submit'], a"));
  const scored = [];
  buttons.forEach((el, index) => {
    if (!isVisible(el)) return;
    const text = clean(el.innerText || el.textContent || el.value || el.getAttribute("aria-label") || el.getAttribute("title") || "");
    if (!text || !/(cookie|accept|agree|reject|decline|necessary|essential|close|got it|ok)/i.test(text)) return;
    let score = 0;
    if (/reject|decline|necessary|essential/i.test(text)) score += 80;
    if (/accept|agree|ok|got it/i.test(text)) score += 55;
    if (/close|dismiss/i.test(text)) score += 40;
    if (/subscribe|sign up|login|continue$/i.test(text)) score -= 60;
    scored.push({ text, selector: selectorFor(el, index), score });
  });
  return scored.sort((a, b) => b.score - a.score).slice(0, 5);
}
"""


SEARCH_RESULT_SCRIPT = r"""
() => {
  const clean = (value) => (value || "").replace(/\s+/g, " ").trim();
  const results = [];
  for (const anchor of Array.from(document.querySelectorAll("a[href]"))) {
    const url = anchor.href || "";
    if (!/^https?:\/\//i.test(url)) continue;
    const text = clean(anchor.innerText || anchor.textContent || "");
    if (!text || text.length < 3) continue;
    if (url.includes("/search?") || url.includes("/preferences") || url.includes("/policies")) continue;
    results.push({ title: text, url });
    if (results.length >= 10) break;
  }
  return results;
}
"""
