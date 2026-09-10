"""Unit and integration tests for Task 4.1 Tool-Calling Agent."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from phase_4_agents.agent_cli import (
    AgentExecutionResult,
    build_tool_agent,
    extract_text_from_content,
    run_agent_query,
)
from phase_4_agents.tools import calculator, get_weather


def test_calculator_operations() -> None:
    """Verify all calculator operations and edge cases."""
    assert calculator.invoke({"operation": "add", "a": 10.0, "b": 5.0}) == "15"
    assert calculator.invoke({"operation": "subtract", "a": 18.0, "b": 11.0}) == "7"
    assert calculator.invoke({"operation": "multiply", "a": 6.0, "b": 7.0}) == "42"
    assert calculator.invoke({"operation": "divide", "a": 100.0, "b": 4.0}) == "25"
    assert "Division by zero" in calculator.invoke(
        {"operation": "divide", "a": 10.0, "b": 0.0}
    )
    assert "Unknown operation" in calculator.invoke(
        {"operation": "exponent", "a": 2.0, "b": 3.0}
    )


def test_get_weather_mock_database() -> None:
    """Verify mock weather lookups and unknown city fallback."""
    tokyo = get_weather.invoke({"city": "Tokyo"})
    assert "Tokyo" in tokyo
    assert "18°C" in tokyo
    assert "Clear and Sunny" in tokyo

    london = get_weather.invoke({"city": "london"})
    assert "11°C" in london

    unknown = get_weather.invoke({"city": "Atlantis"})
    assert "Atlantis" in unknown
    assert "20°C" in unknown


def test_extract_text_from_content() -> None:
    """Test text extractor handling string, list, and dict formats."""
    assert extract_text_from_content("simple text") == "simple text"
    assert (
        extract_text_from_content([{"type": "text", "text": "hello world"}])
        == "hello world"
    )
    assert extract_text_from_content(["part1", "part2"]) == "part1 part2"


def test_run_agent_query_trace_mapping_mocked() -> None:
    """Verify run_agent_query produces detailed trace steps mapped to Task 2.6."""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [
            HumanMessage(content="What is 5 + 3?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"operation": "add", "a": 5, "b": 3},
                        "id": "call_123",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="8", tool_call_id="call_123"),
            AIMessage(content="5 + 3 equals 8."),
        ]
    }

    result = run_agent_query(mock_agent, "What is 5 + 3?")

    assert isinstance(result, AgentExecutionResult)
    assert result.total_steps == 4
    assert result.final_answer == "5 + 3 equals 8."

    # Check step 1 (User Input)
    step1 = result.steps[0]
    assert step1.actor == "User"
    assert step1.message_type == "HumanMessage"
    assert "Initial conversation state" in step1.phase_2_6_mapping

    # Check step 2 (Model Tool Decision)
    step2 = result.steps[1]
    assert step2.actor == "Model (Tool Decision)"
    assert len(step2.tool_calls) == 1
    assert step2.tool_calls[0]["name"] == "calculator"
    assert "functionCall" in step2.phase_2_6_mapping

    # Check step 3 (Tool Execution)
    step3 = result.steps[2]
    assert step3.actor == "Tool Execution"
    assert step3.content == "8"
    assert "Local function executed" in step3.phase_2_6_mapping

    # Check step 4 (Final Answer)
    step4 = result.steps[3]
    assert step4.actor == "Model (Final Answer)"
    assert step4.content == "5 + 3 equals 8."
    assert "loop terminated successfully" in step4.phase_2_6_mapping


def test_build_tool_agent_graph() -> None:
    """Verify build_tool_agent instantiates a CompiledStateGraph."""
    agent = build_tool_agent(api_key="test-api-key")
    assert agent is not None
    # CompiledStateGraph exposes invoke and stream methods
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "stream")
