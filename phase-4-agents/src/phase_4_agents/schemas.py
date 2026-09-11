"""Pydantic schemas for agent tools (Task 4.10 Tool Design Pass).

Provides tight, validated schemas with explicit bounds, unambiguous field names,
and actionable descriptions to minimize LLM hallucinations and argument mismatch.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PDFSearchInput(BaseModel):
    """Input schema for searching Mars Odyssey PDF manuals & ECLSS engineering specs."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description=(
            "Keywords or specific question regarding Project Odyssey Mars Base "
            "engineering specifications, ECLSS cabin atmospheric pressure, "
            "oxygen levels, power bus voltage limits, MDAS hypergolic propulsion, "
            "or habitat thermal control loops."
        ),
    )


class SiteSearchInput(BaseModel):
    """Input schema for searching crawled website documentation and capabilities."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description=(
            "Keywords or topic regarding Toobler digital innovation capabilities, "
            "technical services, cloud-native architecture, IoT edge telemetry, "
            "and enterprise RAG systems."
        ),
    )


class DatabaseQueryInput(BaseModel):
    """Input schema for executing structured SQL or natural language data queries."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description=(
            "A read-only SQL SELECT statement OR a natural language query targeting "
            "transactional business records. Valid tables: 'customers', 'orders', "
            "'products', 'appointments'. Mutations are strictly forbidden."
        ),
    )


class AddToCartInput(BaseModel):
    """Input schema for adding items to shopping cart (WRITE ACTION)."""

    model_config = ConfigDict(extra="forbid")

    product_name: str = Field(
        ...,
        min_length=2,
        max_length=150,
        description=(
            "Exact or partial name of the product to purchase (e.g. 'Titanium "
            "Drill Bit', 'Oxygen Scrubber Cartridge', 'Mars Rover Sensor')."
        ),
    )
    quantity: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of units to purchase (integer between 1 and 100).",
    )
    customer_id: int = Field(
        default=1,
        ge=1,
        description="Numeric ID of the authenticated customer.",
    )


class WebSearchInput(BaseModel):
    """Input schema for querying live internet sources and external real-time facts."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description=(
            "Search keywords for real-time 2026 events, breaking news, "
            "NASA Artemis roadmap updates, or external library releases not "
            "present in internal documentation."
        ),
    )


class CalculatorInput(BaseModel):
    """Input schema for basic arithmetic operations."""

    model_config = ConfigDict(extra="forbid")

    operation: str = Field(
        ...,
        min_length=1,
        description=(
            "Arithmetic operation to perform: 'add' (+), 'subtract' (-), "
            "'multiply' (*), or 'divide' (/)."
        ),
    )
    a: float = Field(..., description="First operand (number).")
    b: float = Field(..., description="Second operand (number).")


class WeatherInput(BaseModel):
    """Input schema for querying city weather conditions."""

    model_config = ConfigDict(extra="forbid")

    city: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description=(
            "Name of the city (e.g. 'Tokyo', 'London', 'Paris', 'New York', "
            "'San Francisco')."
        ),
    )
