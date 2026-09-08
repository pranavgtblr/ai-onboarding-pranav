"""Generates test benchmark PDFs exhibiting real-world parsing challenges:
1. Multi-column layouts (text interleaving risk).
2. Headers and footers polluting chunks (repetition and noise).
3. Complex tables (risk of cell flattening into gibberish).
"""

from pathlib import Path

import pymupdf

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


def create_multi_column_pdf(filename: str = "multi_column_sample.pdf") -> Path:
    """Create a 2-column layout PDF where horizontal extraction causes interleaving."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4

    # Document Title
    page.insert_text(
        (50, 50),
        "Odyssey Mission Technical Digest: Dual-System Summary",
        fontsize=14,
        fontname="helv",
        color=(0, 0.2, 0.5),
    )
    page.draw_line(
        pymupdf.Point(50, 65), pymupdf.Point(545, 65), color=(0.7, 0.7, 0.7), width=1
    )

    # Column 1 (Left: X = 50 to 280)
    col1_lines = [
        "COLUMN 1: ROVER EXPLORATION",
        "The Martian rover Odyssey traversed 14",
        "kilometers across the crater floor.",
        "Soil samples revealed magnesium sulfate",
        "deposits and hydrated silica beds.",
        "Wheels exhibited minor abrasive wear",
        "on the titanium grousers, but torque",
        "margins remained above 85 percent.",
        "Autonomous hazard avoidance engaged",
        "twelve times to navigate steep dunes.",
    ]

    # Column 2 (Right: X = 315 to 545)
    col2_lines = [
        "COLUMN 2: NUCLEAR REACTOR",
        "Primary Stirling reactor unit Beta",
        "generated 9.8 kilowatts electrical.",
        "Sodium coolant core temperature",
        "stabilized at 680 Kelvin nominal.",
        "Radiation perimeter sensors recorded",
        "under 0.04 millisieverts per sol.",
        "Backup control rods demonstrated",
        "full insertion capability in 1.2s.",
        "Continuous power delivery was verified.",
    ]

    # Draw vertical divider line between columns
    page.draw_line(
        pymupdf.Point(297, 85),
        pymupdf.Point(297, 280),
        color=(0.8, 0.8, 0.8),
        width=0.5,
    )

    # Insert text lines at matching Y coordinates to simulate horizontal layout
    for idx, (l1, l2) in enumerate(zip(col1_lines, col2_lines)):
        y = 95 + (idx * 18)
        # Left column
        page.insert_text(
            (50, y),
            l1,
            fontsize=9.5,
            fontname="helv",
            color=(0.1, 0.1, 0.1) if idx == 0 else (0.25, 0.25, 0.25),
        )
        # Right column at EXACT same Y coordinate
        page.insert_text(
            (315, y),
            l2,
            fontsize=9.5,
            fontname="helv",
            color=(0.1, 0.1, 0.1) if idx == 0 else (0.25, 0.25, 0.25),
        )

    out_path = BENCHMARK_DIR / filename
    doc.save(out_path)
    doc.close()
    return out_path


def create_polluted_header_footer_pdf(
    filename: str = "polluted_header_footer_sample.pdf",
) -> Path:
    """Create a 3-page document with running headers and footers."""
    doc = pymupdf.open()

    pages_content = [
        (
            "Section 1: Atmospheric Scrubbing Protocols",
            "The habitat life support system relies on dual regenerative amine beds "
            "for carbon dioxide adsorption. Every 30 minutes, the desorbing bed is "
            "heated to 120 degrees Celsius under partial vacuum, releasing gaseous "
            "CO2 into the Sabatier reactor feed line. Relative humidity is kept at "
            "45 percent to avoid condensation in ventilation ducting.",
        ),
        (
            "Section 2: Emergency Depressurization Valves",
            "In the event of rapid hull puncture, automatic isolation valves seal "
            "the affected module within 450 milliseconds. Crew members must deploy "
            "emergency mask quick-don units providing 100 percent oxygen at positive "
            "pressure. Airlock secondary bulkheads withstand differential pressure "
            "up to 120 kPa without structural deformation.",
        ),
        (
            "Section 3: Cryogenic Boil-Off Zero-Vent Management",
            "Liquid methane and liquid oxygen tanks utilize 40 layers of multi-layer "
            "insulation and active pulse-tube cryocoolers operating at 20 Hz. Boil-off "
            "vapors are re-liquefied and returned to storage reservoirs, maintaining "
            "zero venting loss across the entire 500-sol expedition dwell period.",
        ),
    ]

    for idx, (sec_title, sec_body) in enumerate(pages_content, start=1):
        page = doc.new_page(width=595, height=842)

        # Running Header (Pollution Risk)
        page.insert_text(
            (50, 35),
            "=== RESTRICTED // ODYSSEY BASE OPERATIONS // DO NOT DISTRIBUTE ===",
            fontsize=8.0,
            fontname="helv",
            color=(0.5, 0.2, 0.2),
        )
        page.draw_line(
            pymupdf.Point(50, 42),
            pymupdf.Point(545, 42),
            color=(0.8, 0.5, 0.5),
            width=0.5,
        )

        # Body Content
        page.insert_text(
            (50, 80), sec_title, fontsize=13, fontname="helv", color=(0.1, 0.1, 0.1)
        )

        y = 110
        words = sec_body.split()
        curr_line = []
        for w in words:
            curr_line.append(w)
            if len(" ".join(curr_line)) > 75:
                page.insert_text(
                    (50, y),
                    " ".join(curr_line),
                    fontsize=10,
                    fontname="helv",
                    color=(0.25, 0.25, 0.25),
                )
                y += 18
                curr_line = []
        if curr_line:
            page.insert_text(
                (50, y),
                " ".join(curr_line),
                fontsize=10,
                fontname="helv",
                color=(0.25, 0.25, 0.25),
            )

        # Running Footer (Pollution Risk)
        page.draw_line(
            pymupdf.Point(50, 805),
            pymupdf.Point(545, 805),
            color=(0.8, 0.8, 0.8),
            width=0.5,
        )
        footer_str = (
            f"Project Odyssey Field Manual | Document Ref: ODY-MAN-2026 | "
            f"Page {idx} of 3"
        )
        page.insert_text(
            (50, 820),
            footer_str,
            fontsize=8.0,
            fontname="helv",
            color=(0.5, 0.5, 0.5),
        )

    out_path = BENCHMARK_DIR / filename
    doc.save(out_path)
    doc.close()
    return out_path


def create_table_benchmark_pdf(filename: str = "table_sample.pdf") -> Path:
    """Create a test document with a table to evaluate cell flattening."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text(
        (50, 45),
        "Critical Subsystem Operating Limits Table",
        fontsize=14,
        fontname="helv",
        color=(0, 0.2, 0.5),
    )
    page.insert_text(
        (50, 65),
        "Engineering specification matrix testing relational extraction fidelity.",
        fontsize=9,
        fontname="helv",
        color=(0.4, 0.4, 0.4),
    )

    # Draw grid table
    start_x = 50.0
    start_y = 90.0
    col_widths = [120.0, 110.0, 90.0, 80.0, 90.0]
    total_width = sum(col_widths)
    row_height = 22.0

    headers = [
        "Subsystem Category",
        "Sensor Metric",
        "Nominal Value",
        "Unit",
        "Critical Limit",
    ]
    rows = [
        ["Life Support", "Habitat Pressure", "101.3", "kPa", "< 95.0 kPa"],
        ["Life Support", "O2 Partial Pressure", "21.0", "kPa", "< 19.5 kPa"],
        ["Power Systems", "Stirling Bus Voltage", "120.0", "VDC", "< 110.0 VDC"],
        ["Power Systems", "Sodium Loop Temp", "680.0", "K", "> 750.0 K"],
        ["Propulsion", "Liquid Methane Temp", "111.5", "K", "> 115.0 K"],
        ["Propulsion", "Liquid Oxygen Temp", "90.1", "K", "> 95.0 K"],
    ]

    total_rows = len(rows) + 1
    total_height = total_rows * row_height

    # Draw horizontal grid lines
    for i in range(total_rows + 1):
        y = start_y + (i * row_height)
        page.draw_line(
            pymupdf.Point(start_x, y),
            pymupdf.Point(start_x + total_width, y),
            color=(0.2, 0.2, 0.2),
            width=1.0,
        )

    # Draw vertical grid lines
    curr_x = start_x
    page.draw_line(
        pymupdf.Point(curr_x, start_y),
        pymupdf.Point(curr_x, start_y + total_height),
        color=(0.2, 0.2, 0.2),
        width=1.0,
    )
    for w in col_widths:
        curr_x += w
        page.draw_line(
            pymupdf.Point(curr_x, start_y),
            pymupdf.Point(curr_x, start_y + total_height),
            color=(0.2, 0.2, 0.2),
            width=1.0,
        )

    # Fill header row
    page.draw_rect(
        pymupdf.Rect(start_x, start_y, start_x + total_width, start_y + row_height),
        color=(0.85, 0.88, 0.92),
        fill=(0.85, 0.88, 0.92),
    )

    # Write headers
    curr_x = start_x
    for i, h in enumerate(headers):
        page.insert_text(
            (curr_x + 5, start_y + 15),
            h,
            fontsize=8.5,
            fontname="helv",
            color=(0.1, 0.1, 0.1),
        )
        curr_x += col_widths[i]

    # Write rows
    for r_idx, row in enumerate(rows):
        curr_y = start_y + ((r_idx + 1) * row_height)
        curr_x = start_x
        for c_idx, val in enumerate(row):
            page.insert_text(
                (curr_x + 5, curr_y + 15),
                val,
                fontsize=8.5,
                fontname="helv",
                color=(0.15, 0.15, 0.15),
            )
            curr_x += col_widths[c_idx]

    out_path = BENCHMARK_DIR / filename
    doc.save(out_path)
    doc.close()
    return out_path


def generate_all_parser_benchmarks() -> list[Path]:
    """Generate all three targeted benchmark PDFs."""
    p1 = create_multi_column_pdf()
    p2 = create_polluted_header_footer_pdf()
    p3 = create_table_benchmark_pdf()
    print(f"Generated parser benchmark PDFs in {BENCHMARK_DIR}:")
    print(f"  1. {p1.name} (Multi-column layout)")
    print(f"  2. {p2.name} (Header/Footer pollution)")
    print(f"  3. {p3.name} (Table flattening)")
    return [p1, p2, p3]


if __name__ == "__main__":
    generate_all_parser_benchmarks()
