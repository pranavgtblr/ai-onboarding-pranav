"""Tool definitions for Task 4.1 Agent (Calculator & Weather).

Re-implements the tools hand-crafted in Task 2.6 as LangChain/LangGraph tools.
"""

from typing import Any

from langchain_core.tools import tool

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


@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Execute basic arithmetic operations: add, subtract, multiply, or divide.

    Args:
        operation: One of 'add', 'subtract', 'multiply', or 'divide'.
        a: First operand.
        b: Second operand.
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
            return "Error: Division by zero is undefined."
        res = a / b
    else:
        return f"Error: Unknown operation '{operation}'."

    if isinstance(res, float) and res.is_integer():
        formatted = int(res)
    else:
        formatted = round(res, 4)
    return str(formatted)


@tool
def get_weather(city: str) -> str:
    """Retrieve current weather conditions and temperature for a given city.

    Args:
        city: Name of the city (e.g., 'Tokyo', 'London', 'Paris').
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
