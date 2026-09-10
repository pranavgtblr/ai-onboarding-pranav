"""Unit and integration tests for Task 4.2 Provider-Agnostic Setup."""

from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel

from phase_4_agents.agent_cli import build_tool_agent, run_agent_query
from phase_4_agents.config import MockToolChatModel, Settings, get_chat_model
from phase_4_agents.tools import ALL_TOOLS


def test_provider_switch_via_settings_object() -> None:
    """Verify switching providers via Settings requires zero code changes."""
    # 1. Google Gemini Config
    google_settings = Settings(
        model_provider="google_genai",
        model_name="gemini-3.5-flash-lite",
        google_api_key="fake-google-key",
    )
    google_model = get_chat_model(settings=google_settings)
    assert isinstance(google_model, BaseChatModel)
    assert "Google" in google_model.__class__.__name__ or "google" in getattr(
        google_model, "_llm_type", ""
    )

    # 2. OpenAI Config (with zero code changes in tools/agent logic)
    openai_settings = Settings(
        model_provider="openai",
        model_name="gpt-4o-mini",
        openai_api_key="sk-fake-openai-key",
    )
    openai_model = get_chat_model(settings=openai_settings)
    assert isinstance(openai_model, BaseChatModel)
    assert "OpenAI" in openai_model.__class__.__name__ or "openai" in getattr(
        openai_model, "_llm_type", ""
    )

    # 3. Deterministic Mock Config
    mock_settings = Settings(
        model_provider="mock",
        model_name="mock-model",
    )
    mock_model = get_chat_model(settings=mock_settings)
    assert isinstance(mock_model, MockToolChatModel)


def test_provider_switch_via_env_variables() -> None:
    """Verify provider switching via environment variables."""
    with patch.dict(
        "os.environ",
        {"MODEL_PROVIDER": "mock", "MODEL_NAME": "test-mock"},
    ):
        test_settings = Settings()
        assert test_settings.model_provider == "mock"
        model = get_chat_model(settings=test_settings)
        assert isinstance(model, MockToolChatModel)

    # Test switching to OpenAI via environment
    with patch.dict(
        "os.environ",
        {
            "MODEL_PROVIDER": "openai",
            "MODEL_NAME": "gpt-4o-mini",
            "OPENAI_API_KEY": "sk-test-key-12345",
        },
    ):
        test_settings = Settings()
        assert test_settings.model_provider == "openai"
        assert test_settings.model_name == "gpt-4o-mini"
        model = get_chat_model(settings=test_settings)
        assert "OpenAI" in model.__class__.__name__ or "openai" in getattr(
            model, "_llm_type", ""
        )


def test_tool_binding_identical_across_providers() -> None:
    """Verify tool schemas bind cleanly across both Google and OpenAI providers."""
    google_settings = Settings(
        model_provider="google_genai",
        model_name="gemini-3.5-flash-lite",
        google_api_key="fake-key",
    )
    google_model = get_chat_model(settings=google_settings)
    bound_google = google_model.bind_tools(ALL_TOOLS)
    assert bound_google is not None

    openai_settings = Settings(
        model_provider="openai",
        model_name="gpt-4o-mini",
        openai_api_key="sk-fake-key",
    )
    openai_model = get_chat_model(settings=openai_settings)
    bound_openai = openai_model.bind_tools(ALL_TOOLS)
    assert bound_openai is not None


def test_agent_execution_with_mock_provider() -> None:
    """Verify complete end-to-end agent loop executes offline using mock provider."""
    mock_agent = build_tool_agent(provider="mock")
    result = run_agent_query(mock_agent, "Calculate 25 * 4")

    assert result.total_steps >= 3
    assert any(step.actor == "Tool Execution" for step in result.steps)
    assert any(step.actor == "Model (Final Answer)" for step in result.steps)
    assert "Mock Model" in result.final_answer
