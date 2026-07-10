from app.config import Settings
from app.services.form_filler import FormFillerService
from app.services.llm import LlmService


def build_form_service(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), cerebras_api_key="", nvidia_api_key="")
    return FormFillerService(settings, LlmService(settings))


def test_form_filler_marks_sensitive_fields(tmp_path):
    service = build_form_service(tmp_path)

    fields = service._postprocess_fields(
        [
            {"label": "Full name", "name": "name", "type": "text", "control_type": "text", "selector": "#name"},
            {"label": "Password", "name": "password", "type": "password", "control_type": "text", "selector": "#password"},
            {"label": "Credit card number", "name": "card", "type": "text", "control_type": "text", "selector": "#card"},
            {"label": "Resume", "name": "resume", "type": "file", "control_type": "text", "selector": "#resume"},
        ]
    )

    assert fields[0]["sensitive"] is False
    assert fields[1]["sensitive"] is True
    assert fields[2]["sensitive"] is True
    assert fields[3]["sensitive"] is True
    assert fields[3]["blocked_reason"] == "file upload needs manual selection"


def test_form_filler_local_mapping_uses_prompt_only_values(tmp_path):
    service = build_form_service(tmp_path)
    fields = service._postprocess_fields(
        [
            {"label": "Full name", "name": "full_name", "type": "text", "control_type": "text", "selector": "#name", "required": True},
            {"label": "Email address", "name": "email", "type": "email", "control_type": "text", "selector": "#email", "required": True},
            {"label": "Message", "name": "message", "type": "textarea", "control_type": "textarea", "selector": "#message"},
            {"label": "Password", "name": "password", "type": "password", "control_type": "text", "selector": "#password"},
        ]
    )

    values = service._map_values_locally(
        "fill this form name is Sahil, email is sahil@example.com, message is interested in demo",
        fields,
    )

    assert values["full_name"] == "Sahil"
    assert values["email_address"] == "sahil@example.com"
    assert values["message"] == "interested in demo"
    assert "password" not in values


def test_form_filler_blocks_private_urls_but_allows_localhost(tmp_path):
    service = build_form_service(tmp_path)

    service._assert_safe_url("http://127.0.0.1:3000/form")
    service._assert_safe_url("http://localhost:3000/form")

    try:
        service._assert_safe_url("http://192.168.1.1/admin")
    except ValueError as exc:
        assert "Private network" in str(exc)
    else:
        raise AssertionError("private network URL should be blocked")


def test_form_filler_known_site_login_routes_to_real_login(tmp_path):
    service = build_form_service(tmp_path)

    url = service._known_site_url("open youtube and fill form of login with dummy data")

    assert url.startswith("https://accounts.google.com/ServiceLogin")
    assert "youtube" in url


def test_form_filler_known_site_create_account_routes_to_signup(tmp_path):
    service = build_form_service(tmp_path)

    url = service._known_site_url("create youtube account with dummy data")

    assert url.startswith("https://accounts.google.com/signup")
    assert "youtube" in url


def test_form_filler_known_sites_cover_common_login_and_signup_pages(tmp_path):
    service = build_form_service(tmp_path)

    assert service._known_site_url("open stack overflow login with dummy data").endswith("/users/login")
    assert service._known_site_url("create github account with dummy data") == "https://github.com/signup"
    assert "amazon.in/ap/register" in service._known_site_url("create amazon account with dummy data")
    assert "naukri.com/registration" in service._known_site_url("naukri signup with dummy data")


def test_form_filler_infers_flow_type_from_general_prompts(tmp_path):
    service = build_form_service(tmp_path)

    assert service._infer_flow_type("login to a website with dummy data") == "login"
    assert service._infer_flow_type("create account on a new site") == "create_account"
    assert service._infer_flow_type("fill the careers application form") == "job_application"
    assert service._infer_flow_type("fill admission form for college") == "admission_form"
    assert service._infer_flow_type("fill contact form with message") == "contact_form"
    assert service._infer_flow_type("complete feedback survey") == "survey"


def test_form_filler_scores_discovered_links_by_flow_type(tmp_path):
    service = build_form_service(tmp_path)

    login_score = service._score_form_candidate({"text": "Sign in", "url": "https://example.com/login", "tag": "a"}, "login with dummy data", "login")
    blog_score = service._score_form_candidate({"text": "Read our blog", "url": "https://example.com/blog", "tag": "a"}, "login with dummy data", "login")
    apply_score = service._score_form_candidate({"text": "Apply now", "url": "https://example.com/careers/apply", "tag": "a"}, "fill job application", "job_application")

    assert login_score > blog_score
    assert apply_score >= 45


def test_form_filler_dummy_values_fill_only_safe_fields(tmp_path):
    service = build_form_service(tmp_path)
    fields = service._postprocess_fields(
        [
            {"label": "Email or phone", "name": "identifier", "type": "email", "control_type": "text", "selector": "#identifier", "required": True},
            {"label": "Password", "name": "password", "type": "password", "control_type": "text", "selector": "#password", "required": True},
            {"label": "Message", "name": "message", "type": "textarea", "control_type": "textarea", "selector": "#message"},
        ]
    )

    values = service._apply_dummy_values("fill login form with dummy data", fields, {})
    normalized = service._normalize_mapping(fields, values)

    assert normalized["email_or_phone"] == "dummy@example.com"
    assert normalized["message"] == "This is dummy data for testing the form filler"
    assert "password" not in normalized
