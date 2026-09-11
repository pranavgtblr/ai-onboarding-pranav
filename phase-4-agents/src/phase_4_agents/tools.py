"""Tool definitions for Task 4.1 Agent (Calculator & Weather).

Re-implements the tools hand-crafted in Task 2.6 as LangChain/LangGraph tools.
"""

from typing import Any

from langchain_core.tools import tool

from phase_4_agents.schemas import CalculatorInput, WeatherInput

# Mock weather database identical to Task 2.6
MOCK_WEATHER_DATABASE: dict[str, dict[str, Any]] = {
    "tokyo": {
        "city": "Tokyo",
        "temperature_c": 18,
        "condition": "Clear and Sunny",
        "humidity": "55%",
        "wind": "12 km/h",
    },
    "london": {
        "city": "London",
        "temperature_c": 11,
        "condition": "Overcast with light drizzle",
        "humidity": "82%",
        "wind": "20 km/h",
    },
    "new york": {
        "city": "New York",
        "temperature_c": 15,
        "condition": "Partly Cloudy",
        "humidity": "60%",
        "wind": "14 km/h",
    },
    "paris": {
        "city": "Paris",
        "temperature_c": 14,
        "condition": "Mild and Breezy",
        "humidity": "68%",
        "wind": "16 km/h",
    },
    "san francisco": {
        "city": "San Francisco",
        "temperature_c": 13,
        "condition": "Foggy morning clearing to sun",
        "humidity": "75%",
        "wind": "18 km/h",
    },
}


@tool(args_schema=CalculatorInput)
def calculator(operation: str, a: float, b: float) -> str:
    """Execute basic arithmetic operations: add, subtract, multiply, or divide.

    Use this tool ONLY for arithmetic computations. DO NOT use for date math,
    string manipulation, or general reasoning.

    Args:
        operation: One of 'add', 'subtract', 'multiply', or 'divide'.
        a: First numeric operand.
        b: Second numeric operand.
    """
    op = operation.lower().strip()
    if op in ("add", "+"):
        res = a + b
    elif op in ("subtract", "-"):
        res = a - b
    elif op in ("multiply", "*"):
        res = a * b
    elif op in ("divide", "/"):
        if b == 0:
            return (
                "Calculator Error: Division by zero is mathematically undefined. "
                "Action for model: Check the divisor operand 'b' or formulate an "
                "alternative mathematical expression."
            )
        res = a / b
    else:
        return (
            f"Calculator Error: Unknown operation '{operation}'. "
            "Action for model: Call calculator with one of "
            "('add', 'subtract', 'multiply', 'divide')."
        )

    if isinstance(res, float) and res.is_integer():
        formatted = int(res)
    else:
        formatted = round(res, 4)
    return str(formatted)


@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Retrieve current weather conditions and temperature for a given city.

    Use this tool ONLY for current meteorological inquiries. DO NOT use for historical
    weather or general city information (use web_search or site_search instead).

    Args:
        city: Name of the city (e.g., 'Tokyo', 'London', 'Paris', 'New York').
    """
    normalized = city.strip().lower()
    if normalized in MOCK_WEATHER_DATABASE:
        data = MOCK_WEATHER_DATABASE[normalized]
        return (
            f"Weather in {data['city']}: {data['condition']}, "
            f"Temperature: {data['temperature_c']}°C, "
            f"Humidity: {data['humidity']}, Wind: {data['wind']}."
        )

    # Fallback for unmocked cities
    return (
        f"Weather in {city.title()}: Partly Cloudy, "
        "Temperature: 20°C, Humidity: 50%, Wind: 10 km/h."
    )


ALL_TOOLS = [calculator, get_weather]
