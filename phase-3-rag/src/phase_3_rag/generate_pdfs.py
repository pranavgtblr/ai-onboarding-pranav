"""Generates 20 real PDF documents for Task 3.7 benchmarking.

Includes:
- 1 pure image-only scanned document (pdf_04) requiring OCR
- 1 table-heavy document (pdf_07) containing multi-column tabular data
- 18 digital technical documentation PDFs for Project Odyssey Mars Base
- manifest.json registering all documents and properties
"""

import json
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "pdfs"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def draw_table(
    page: pymupdf.Page,
    headers: list[str],
    rows: list[list[str]],
    start_x: float,
    start_y: float,
    col_widths: list[float],
    row_height: float = 24.0,
) -> float:
    """Draw a table with visible grid lines and text cells."""
    total_width = sum(col_widths)
    total_rows = len(rows) + 1
    total_height = total_rows * row_height

    # Horizontal grid lines
    for i in range(total_rows + 1):
        y = start_y + (i * row_height)
        page.draw_line(
            pymupdf.Point(start_x, y),
            pymupdf.Point(start_x + total_width, y),
            color=(0.2, 0.2, 0.2),
            width=1.0,
        )

    # Vertical grid lines
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

    # Fill header row background
    page.draw_rect(
        pymupdf.Rect(start_x, start_y, start_x + total_width, start_y + row_height),
        color=(0.85, 0.88, 0.92),
        fill=(0.85, 0.88, 0.92),
    )

    # Write headers
    curr_x = start_x
    for i, h in enumerate(headers):
        page.insert_text(
            (curr_x + 5, start_y + 16),
            str(h),
            fontsize=9.0,
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
                (curr_x + 5, curr_y + 16),
                str(val),
                fontsize=8.5,
                fontname="helv",
                color=(0.15, 0.15, 0.15),
            )
            curr_x += col_widths[c_idx]

    return start_y + total_height + 15.0


def create_digital_pdf(
    filename: str,
    title: str,
    sections: list[tuple[str, str]],
) -> None:
    """Create a digital text PDF with multiple pages, headings, and paragraphs."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4

    # Header title
    page.insert_text((50, 60), title, fontsize=16, fontname="helv", color=(0, 0.2, 0.5))
    page.draw_line(
        pymupdf.Point(50, 75),
        pymupdf.Point(545, 75),
        color=(0.7, 0.7, 0.7),
        width=1,
    )

    y = 100.0
    for sec_title, sec_body in sections:
        if y > 720:
            page = doc.new_page(width=595, height=842)
            y = 60.0

        page.insert_text(
            (50, y), sec_title, fontsize=12, fontname="helv", color=(0.1, 0.1, 0.1)
        )
        y += 18

        # Wrap text lines
        words = sec_body.split()
        current_line: list[str] = []
        for word in words:
            current_line.append(word)
            line_str = " ".join(current_line)
            if len(line_str) > 75:
                page.insert_text(
                    (50, y),
                    line_str,
                    fontsize=9.5,
                    fontname="helv",
                    color=(0.25, 0.25, 0.25),
                )
                y += 14
                current_line = []
                if y > 780:
                    page = doc.new_page(width=595, height=842)
                    y = 60.0

        if current_line:
            page.insert_text(
                (50, y),
                " ".join(current_line),
                fontsize=9.5,
                fontname="helv",
                color=(0.25, 0.25, 0.25),
            )
            y += 24

    doc.save(DATA_DIR / filename)
    doc.close()


def create_scanned_pdf(filename: str) -> None:
    """Create a 100% image-only scanned document with 0 digital characters."""
    img = Image.new("RGB", (1000, 1400), color=(248, 246, 238))
    d = ImageDraw.Draw(img)

    # Document border and simulated archival header
    d.rectangle([(40, 40), (960, 1360)], outline=(120, 110, 100), width=3)
    d.rectangle(
        [(50, 50), (950, 150)],
        fill=(235, 230, 215),
        outline=(150, 140, 130),
        width=1,
    )

    d.text(
        (70, 70),
        "RESTRICTED ARCHIVAL RECORD - MARS EXPEDITION VANGUARD",
        fill=(60, 40, 30),
    )
    d.text(
        (70, 105),
        "DECLASSIFIED TECHNICAL FIELD LOG: SOL 01 TO SOL 10",
        fill=(80, 60, 50),
    )

    # Stamp box
    d.rectangle([(720, 60), (930, 135)], outline=(180, 50, 50), width=2)
    d.text((740, 85), "[OFFICIAL OCR AUDIT]", fill=(180, 50, 50))

    # Historical typewriter notes
    lines = [
        "COMMANDER MISSION LOG: SOL 04 EXPEDITION UPDATE",
        "-" * 80,
        "1. Atmospheric sampling reveals surface barometric pressure at 6.1 mbar.",
        "2. Regolith core drill completed at coordinates 18.4 N, 77.2 E.",
        "3. Water ice sub-surface deposits detected at 1.4 meters depth.",
        "4. Primary Stirling engine Kilopower reactor achieved criticality at 0400.",
        "5. Coolant sodium-potassium loop flow rate stabilized at 1.82 kg/sec.",
        "6. Habitat dome environmental life support pressure sealed at 101.3 kPa.",
        "7. Oxygen recovery via MOXIE prototype yielded 12.4 grams per hour.",
        "8. Crew physical conditioning: all 6 astronauts within nominal bounds.",
        "9. Emergency rover EV-1 battery pack recharge cycle verified via solar B.",
        "10. Communication latency to Earth DSN measured at 14.2 minutes.",
        "",
        "FIELD NOTE SIGN-OFF:",
        "Dust deposition on exterior radiator panels exceeded baseline predictions",
        "by 14 percent following local convective dust devil.",
        "Mechanical wiper sweeps scheduled every 48 hours to preserve cooling.",
    ]

    curr_y = 200
    for line in lines:
        d.text((70, curr_y), line, fill=(40, 40, 40))
        curr_y += 36

    img_path = DATA_DIR / "temp_scanned_page.png"
    img.save(img_path, format="PNG")

    # Put image into PDF with NO text layer
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(20, 20, 575, 822), filename=str(img_path))
    doc.save(DATA_DIR / filename)
    doc.close()

    if img_path.exists():
        img_path.unlink()


def create_table_heavy_pdf(filename: str) -> None:
    """Create a PDF dominated by structured multi-column tables."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text(
        (50, 45),
        "Master Telemetry Matrix & Engineering Tolerances",
        fontsize=15,
        fontname="helv",
        color=(0, 0.2, 0.5),
    )
    page.insert_text(
        (50, 65),
        "Document ID: SPEC-TEL-07 | Systems Engineering Reference Table Matrix",
        fontsize=9,
        fontname="helv",
        color=(0.4, 0.4, 0.4),
    )

    # Table 1: Atmospheric Environmental Telemetry
    page.insert_text(
        (50, 95),
        "1. Atmospheric Sensors & Life Support Parameters",
        fontsize=11,
        fontname="helv",
        color=(0.1, 0.1, 0.1),
    )
    t1_headers = ["Subsystem Sensor", "Nominal", "Min Alarm", "Max Alarm", "Unit"]
    t1_rows = [
        ["Habitat Pressure", "101.3", "95.0", "105.0", "kPa"],
        ["Oxygen Partial (pO2)", "21.0", "19.5", "23.5", "kPa"],
        ["Carbon Dioxide (pCO2)", "0.35", "0.0", "0.70", "kPa"],
        ["Nitrogen Partial (pN2)", "78.0", "74.0", "81.0", "kPa"],
        ["Relative Humidity", "45.0", "30.0", "60.0", "%"],
        ["Cabin Ambient Temp", "21.5", "18.0", "25.0", "degC"],
    ]
    next_y = draw_table(page, t1_headers, t1_rows, 50, 110, [140, 80, 80, 80, 90], 20.0)

    # Table 2: Cryogenic Propellant Storage Telemetry
    page.insert_text(
        (50, next_y + 10),
        "2. Cryogenic Propellant Storage Telemetry",
        fontsize=11,
        fontname="helv",
        color=(0.1, 0.1, 0.1),
    )
    t2_headers = [
        "Tank Identifier",
        "Cryogen",
        "Quantity (kg)",
        "Storage Temp (K)",
        "Boil-off Rate",
    ]
    t2_rows = [
        ["TK-LOX-01", "Liquid Oxygen", "18500", "90.2", "0.05% / sol"],
        ["TK-LOX-02", "Liquid Oxygen", "18500", "90.1", "0.04% / sol"],
        ["TK-CH4-01", "Liquid Methane", "6200", "111.5", "0.08% / sol"],
        ["TK-CH4-02", "Liquid Methane", "6200", "111.4", "0.07% / sol"],
        ["TK-LN2-01", "Liquid Nitrogen", "4500", "77.3", "0.02% / sol"],
    ]
    next_y = draw_table(
        page, t2_headers, t2_rows, 50, next_y + 25, [110, 110, 90, 95, 75], 20.0
    )

    # Table 3: Electrical Power Generation Bus
    page.insert_text(
        (50, next_y + 10),
        "3. Nuclear & Solar Power Bus Voltage & Amperage",
        fontsize=11,
        fontname="helv",
        color=(0.1, 0.1, 0.1),
    )
    t3_headers = [
        "Power Bus ID",
        "Source Type",
        "Voltage (VDC)",
        "Max Current (A)",
        "Efficiency",
    ]
    t3_rows = [
        ["BUS-NUC-A", "Kilopower Stirling", "120.0", "83.3", "94.2%"],
        ["BUS-NUC-B", "Kilopower Stirling", "120.0", "83.3", "94.0%"],
        ["BUS-SOL-01", "Photovoltaic Wing 1", "240.0", "41.6", "29.5%"],
        ["BUS-SOL-02", "Photovoltaic Wing 2", "240.0", "41.6", "29.8%"],
        ["BUS-BAT-PRI", "Lithium-Sulfur Pack", "120.0", "150.0", "98.1%"],
    ]
    draw_table(page, t3_headers, t3_rows, 50, next_y + 25, [100, 130, 90, 90, 70], 20.0)

    doc.save(DATA_DIR / filename)
    doc.close()


def generate_all_20_pdfs() -> None:
    """Generate the full 20-PDF dataset."""
    corpus_specs = [
        (
            "pdf_01_atmospheric_entry.pdf",
            "Mars Atmospheric Entry, Descent and Landing (EDL) Manual",
            "Aerospace Engineering",
            [
                (
                    "1. Entry Interface Conditions",
                    "Entry trajectory begins at 125 km altitude with velocity of "
                    "6.1 km/sec. Heat shield thermal protection system utilizes "
                    "PICA-X ablative tiles designed to endure heat fluxes "
                    "exceeding 450 W/cm2.",
                ),
                (
                    "2. Supersonic Parachute Deployment",
                    "At Mach 2.2 and 10.5 km altitude, a 21.5-meter disk-gap-band "
                    "parachute deploys under mortar ejection, shedding 92 percent "
                    "of kinetic energy before radar ground acquisition.",
                ),
                (
                    "3. Terminal Sky Crane Retrorockets",
                    "Terminal touchdown utilizes four throttleable hydrazine "
                    "rocket clusters producing 3.1 kN each, descending at a steady "
                    "0.75 m/sec until laser contact sensors trigger engine cutoff.",
                ),
            ],
        ),
        (
            "pdf_02_nuclear_reactor_specs.pdf",
            "Kilopower 10 kWe Stirling Nuclear Reactor Operating Limits",
            "Power Systems",
            [
                (
                    "1. Reactor Core Architecture",
                    "The nuclear power assembly utilizes an enriched uranium-235 "
                    "molybdenum alloy core casting enclosed in a beryllium oxide "
                    "reflector ring, operating at nominal core temperature of "
                    "800 degrees Celsius.",
                ),
                (
                    "2. Stirling Power Conversion",
                    "Sodium heat pipes transfer thermal energy to dual balanced "
                    "Stirling engines, producing 10 kilowatts continuous "
                    "electrical power with design operational lifespan of 10 "
                    "Martian years.",
                ),
                (
                    "3. Radiation Shielding Zone",
                    "A lithium hydride and depleted tungsten shadow shield "
                    "attenuates gamma and neutron radiation to under 5 rem/year "
                    "at the habitat perimeter boundary 100 meters away.",
                ),
            ],
        ),
        (
            "pdf_03_moxie_electrolysis.pdf",
            "Solid Oxide Electrolysis Oxygen Extraction Manual",
            "Life Support & ISRU",
            [
                (
                    "1. Solid Oxide Electrolyzer Principles",
                    "Martian atmospheric carbon dioxide is compressed to 1.0 bar "
                    "and heated to 800 degrees Celsius across yttria-stabilized "
                    "zirconia (YSZ) catalytic membranes.",
                ),
                (
                    "2. Catalytic Cleavage & Yield",
                    "Electrolysis splits CO2 into oxygen ions and carbon monoxide "
                    "byproduct: 2 CO2 -> 2 CO + O2. Net breathable oxygen output "
                    "reaches 22.0 grams per hour per stack module.",
                ),
                (
                    "3. Gas Purity & Scrubbing",
                    "Produced oxygen passes through ceramic particulate filters "
                    "and silver-zeolite scrubbers, certifying 99.8 percent "
                    "medical-grade breathable purity prior to storage liquefaction.",
                ),
            ],
        ),
        (
            "pdf_05_water_recycling_closed_loop.pdf",
            "ECLSS Closed-Loop Water Recycling & Condensate Distillation",
            "Life Support",
            [
                (
                    "1. Rotary Vacuum Distillation",
                    "Urine processor assemblies utilize vapor compression "
                    "distillation within a spinning centrifugal drum to extract "
                    "pure water vapor from wastewater brine at 93 percent "
                    "efficiency.",
                ),
                (
                    "2. Catalytic Post-Treatment",
                    "Condensate flows through high-temperature catalytic "
                    "oxidizers operating at 120 degrees Celsius with zero-g "
                    "compatible peristaltic pumps, eliminating volatile organic "
                    "contaminants.",
                ),
                (
                    "3. Mineralization & Biocide",
                    "Recycled water is re-mineralized with magnesium and calcium "
                    "salts, dosed with ionic silver biocide at 400 ppb, and stored "
                    "in dual 1500-liter titanium tanks.",
                ),
            ],
        ),
        (
            "pdf_06_eva_spacesuit_maintenance.pdf",
            "Mark IV Planetary Exploration Spacesuit Pressure Garment Maintenance",
            "Crew Equipment",
            [
                (
                    "1. Pressure Garment Assembly",
                    "The Mark IV suit operates at 29.6 kPa (4.3 psi) pure oxygen "
                    "atmosphere, featuring rear-entry composite upper torso and "
                    "articulating titanium rotary bearing joints.",
                ),
                (
                    "2. Portable Life Support System (PLSS)",
                    "PLSS backpack integrates dual lithium hydroxide CO2 "
                    "scrubbers, sublimator ice cooling plates, and dual "
                    "high-pressure oxygen tanks pressurized to 240 bar.",
                ),
                (
                    "3. Micrometeorite & Dust Layer",
                    "Outer orthofabric cover combines Nomex, Kevlar, and "
                    "aluminized Mylar to defend against abrasive basalt dust "
                    "particles and micrometeorite impacts up to 5 km/sec.",
                ),
            ],
        ),
        (
            "pdf_08_cryogenic_methane_propulsion.pdf",
            "Liquid Methane (CH4) & Liquid Oxygen (LOX) Densification Protocol",
            "ISRU & Propulsion",
            [
                (
                    "1. Propellant Subcooling",
                    "Liquid methane is densified to 90 K and liquid oxygen to "
                    "66 K using closed-cycle neon cryocoolers, increasing bulk "
                    "propellant loading density by 8.4 percent inside the Earth "
                    "Return Vehicle tanks.",
                ),
                (
                    "2. Boil-Off Zero-Vent Storage",
                    "Cryogenic tanks incorporate 40-layer double-aluminized "
                    "Mylar multi-layer insulation (MLI) and pulse-tube active "
                    "re-liquefaction to eliminate venting losses over 500-sol "
                    "dwell periods.",
                ),
                (
                    "3. Transfer Line Purge",
                    "All cryogenic propellant transfer lines are vacuum-jacketed "
                    "and pre-chilled with gaseous helium before liquid propellant "
                    "transit to prevent cavitation shockwaves in turbopumps.",
                ),
            ],
        ),
        (
            "pdf_09_rover_autonomous_navigation.pdf",
            "Autonomous Rover Navigation & Stereo Visual Odometry",
            "Mobility",
            [
                (
                    "1. Terrain Hazard Mapping",
                    "Navigational mast cameras execute stereo disparity matching "
                    "at 10 Hz, generating 3D digital elevation meshes with "
                    "2-centimeter resolution up to 30 meters ahead.",
                ),
                (
                    "2. Rocker-Bogie Chassis Control",
                    "Six independent hub-motor driven titanium wheels traverse "
                    "boulders up to 45 centimeters in height with active "
                    "wheel-walking torque vectoring on steep basalt sand dunes.",
                ),
                (
                    "3. Path Planning Algorithmic Suite",
                    "Real-time Field D* pathfinding recalculates optimal traverse "
                    "trajectories every 250 milliseconds, bypassing sand traps "
                    "and maintaining maximum slip ratios below 18 percent.",
                ),
            ],
        ),
        (
            "pdf_10_deep_space_optical_comm.pdf",
            "Deep Space Optical Communications (DSOC) 1550 nm Laser Uplink",
            "Communications",
            [
                (
                    "1. Laser Transmitter Architecture",
                    "The optical communications terminal utilizes a 5-watt "
                    "erbium-doped fiber laser operating at 1550 nanometers "
                    "wavelength, coupled to a 22-centimeter aperture telescope.",
                ),
                (
                    "2. Ground Station Acquisition",
                    "Uplink beacons from Earth Palomar 5-meter telescope are "
                    "acquired using photon-counting superconducting nanowire "
                    "detector arrays, achieving 100 Mbps data bandwidth across "
                    "2.2 AU.",
                ),
                (
                    "3. Pointing & Jitter Isolation",
                    "Piezoelectric fast-steering mirrors compensate for "
                    "spacecraft micro-vibrations with 0.8 microradian pointing "
                    "accuracy, counteracting Martian planetary motion and orbit "
                    "drift.",
                ),
            ],
        ),
        (
            "pdf_11_solar_flare_shelter_protocol.pdf",
            "SPE Class X-Ray Solar Proton Storm Shelter Protocol",
            "Safety & Survival",
            [
                (
                    "1. Early Warning Triggers",
                    "Deep space heliospheric sensors trigger alarm code BRAVO "
                    "upon detecting Class X flare coronal mass ejections with "
                    "proton flux exceeding 1000 pfu at >10 MeV.",
                ),
                (
                    "2. Crew Relocation Timeline",
                    "All extravehicular activities are aborted immediately. "
                    "Crew must complete ingress into the sub-surface habitat "
                    "central storm shelter within 45 minutes of primary alert.",
                ),
                (
                    "3. Storm Shelter Shielding",
                    "The shelter is surrounded by 50 cm polyethylene water "
                    "storage bladders and 1.2 meters of compacted Martian "
                    "regolith, reducing cumulative dose rates to under "
                    "0.15 mSv/day.",
                ),
            ],
        ),
        (
            "pdf_12_hydroponic_crop_production.pdf",
            "Hydroponic & Aeroponic Biomass Production Guidelines",
            "Life Support",
            [
                (
                    "1. Automated Nutrient Film Technique",
                    "Continuous flow nutrient channels support dwarf wheat, "
                    "sweet potato, and leafy greens under dual-spectrum red/blue "
                    "LED arrays operating at 350 micromoles/m2/s.",
                ),
                (
                    "2. Atmosphere Composition & Enrichment",
                    "Greenhouse modules maintain carbon dioxide enrichment at "
                    "1200 ppm and 65 percent relative humidity, accelerating "
                    "crop vegetative growth cycles by 38 percent compared to "
                    "terrestrial baselines.",
                ),
                (
                    "3. Biological Waste Recirculation",
                    "Composted inedible biomass feeds anaerobic digestors, "
                    "yielding bio-available nitrogen and methane while spirulina "
                    "algae vats capture secondary carbon dioxide.",
                ),
            ],
        ),
        (
            "pdf_13_regolith_brick_construction.pdf",
            "Sintered Martian Basalt Regolith Structural Construction",
            "Infrastructure",
            [
                (
                    "1. Microwave Sintering Kinetics",
                    "Martian regolith simulant is heated to 1150 degrees Celsius "
                    "using 2.45 GHz industrial magnetrons, fusing basalt grains "
                    "without requiring external cement or water binders.",
                ),
                (
                    "2. Compressive Strength Metrics",
                    "Sintered basalt structural blocks demonstrate compressive "
                    "strength of 68 MPa, surpassing standard high-strength "
                    "terrestrial concrete and resisting abrasive sand scouring.",
                ),
                (
                    "3. Vaulted Arch Radiation Overburden",
                    "Robotic gantry printers deposit interlocking sintered arches "
                    "over habitat inflatable pressure vessels, supporting a "
                    "3-meter regolith overburden to block galactic cosmic rays.",
                ),
            ],
        ),
        (
            "pdf_14_dust_storm_abrasion_mitigation.pdf",
            "Martian Atmospheric Dust Abrasion & Electrostatic Repulsion",
            "Habitat Operations",
            [
                (
                    "1. Triboelectric Charge Characteristics",
                    "Atmospheric suspension imparts strong negative triboelectric "
                    "charges to iron oxide dust grains, causing high adhesive "
                    "electrostatic affinity to photovoltaic glass and suit fabrics.",
                ),
                (
                    "2. Electrostatic Dust Shielding (EDS)",
                    "Solar arrays feature embedded indium tin oxide electrode "
                    "grids pulsing high-voltage three-phase AC electric fields "
                    "to levitate and sweep dust grains off panel surfaces.",
                ),
                (
                    "3. Rotary Joint Labyrinth Seals",
                    "Air lock hatches and rover wheel bearing assemblies utilize "
                    "triple magnetic fluid ferrofluidic seals and pressurized "
                    "nitrogen purge barriers to prevent particulate ingress.",
                ),
            ],
        ),
        (
            "pdf_15_medical_trauma_guidelines.pdf",
            "Extraterrestrial Medical Emergency & Hypobaric Decompression Guidelines",
            "Medical & Health",
            [
                (
                    "1. Acute Decompression Sickness (DCS)",
                    "DCS protocols require immediate crew placement into the "
                    "2.8 ATA hyperbaric treatment lock, administering 100 percent "
                    "humidified normothermic oxygen for a 120-minute cycle.",
                ),
                (
                    "2. Microgravity Fracture Stabilization",
                    "Long-bone fractures are stabilized with lightweight "
                    "carbon-fiber external fixator splints and ultrasound "
                    "bone-growth stimulation to counter microgravity osteopenia.",
                ),
                (
                    "3. Telemedicine Diagnostic Ultrasound",
                    "Autonomous ultrasound scanning with real-time Earth "
                    "surgical teleconsultation guides thoracic cavity needle "
                    "decompressions in suspected pneumothorax incidents.",
                ),
            ],
        ),
        (
            "pdf_16_sabatier_reactor_kinetics.pdf",
            "Sabatier Carbon Dioxide Methanation Catalytic Kinetics",
            "ISRU & Propulsion",
            [
                (
                    "1. Reaction Stoichiometry",
                    "CO2 and hydrogen react exothermically: CO2 + 4 H2 -> CH4 + "
                    "2 H2O (delta H = -165 kJ/mol). Ruthenium-on-alumina catalyst "
                    "beads operate at 380 degrees Celsius.",
                ),
                (
                    "2. Water Condensation & Electrolysis",
                    "Produced water vapor is condensed in finned heat exchangers, "
                    "and routed to PEM water electrolyzers to regenerate hydrogen "
                    "and deliver pure oxygen to the outpost.",
                ),
                (
                    "3. Methane Liquefaction Train",
                    "Pure methane gas is compressed through a three-stage piston "
                    "compressor to 35 bar, desiccated through 3A molecular "
                    "sieves, and liquefied into cryogenic storage tanks.",
                ),
            ],
        ),
        (
            "pdf_17_waste_heat_radiator_loops.pdf",
            "Active Thermal Control & Ammonia Waste Heat Radiator Loops",
            "Environmental Control",
            [
                (
                    "1. Dual-Loop Thermal Architecture",
                    "Internal water coolant loops absorb electronics and human "
                    "metabolic heat inside the habitat, transferring thermal "
                    "energy via plate-and-frame heat exchangers to an external "
                    "anhydrous ammonia loop.",
                ),
                (
                    "2. Deployable Radiator Panels",
                    "Composite aluminum-honeycomb radiator wings radiate heat "
                    "into deep space at night (effective sink temperature 180 K) "
                    "with emissivity of 0.91 and solar absorptivity of 0.08.",
                ),
                (
                    "3. Passive Freeze Protection",
                    "Proportional bypass valves throttle ammonia flow during "
                    "extreme Martian winter nights (-120 C) to prevent coolant "
                    "freezing, maintaining core loop return temperatures above "
                    "-35 C.",
                ),
            ],
        ),
        (
            "pdf_18_sample_return_containment.pdf",
            "Planetary Protection Biosafety Level 4 Geological Sample Isolation",
            "Science & Planetary Protection",
            [
                (
                    "1. Pristine Geological Containment",
                    "Sub-surface Martian soil and rock cores are sealed within "
                    "double-walled brazed titanium canisters under negative "
                    "pressure vacuum seals at the drill head.",
                ),
                (
                    "2. Dry Heat Microbial Sterilization",
                    "Sample canister exteriors undergo dry heat microbial "
                    "reduction (DHMR) at 135 degrees Celsius for 42 hours before "
                    "transit through cleanroom airlocks to eliminate "
                    "cross-contamination.",
                ),
                (
                    "3. Glovebox Raman Spectrometry",
                    "Non-destructive Raman, XRF, and infrared spectroscopy "
                    "inspect sample mineralogy within a glovebox filled with "
                    "ultra-pure argon atmosphere at positive pressure.",
                ),
            ],
        ),
        (
            "pdf_19_geological_drilling_telemetry.pdf",
            "Deep Subsurface Core Drilling Telemetry & Borehole Stratigraphy",
            "Science & Geology",
            [
                (
                    "1. Rotary Percussive Drill Mechanics",
                    "The 10-meter coring rig operates at 400 RPM with 15 Hz "
                    "axial percussion, utilizing polycrystalline diamond compact "
                    "(PDC) drill bits cooled by pressurized gaseous nitrogen.",
                ),
                (
                    "2. Subsurface Ice Layer Detection",
                    "At borehole depth 4.8 meters, drill telemetry registered "
                    "torque drop from 45 Nm to 12 Nm, confirming penetration into "
                    "massive glacial ice sheet with 97 percent water purity.",
                ),
                (
                    "3. Core Extraction Logging",
                    "Extracted 25 mm diameter stratigraphy cores are cataloged "
                    "into automated cryogenic sample cassettes preserved at "
                    "210 K to avoid thermal alteration.",
                ),
            ],
        ),
        (
            "pdf_20_outpost_charter_governance.pdf",
            "Odyssey Expedition Inter-Agency Command Authority Protocol",
            "Governance & Administration",
            [
                (
                    "1. Chain of Command Hierarchy",
                    "The Expedition Commander holds absolute operational "
                    "authority during red and yellow alert conditions. Science "
                    "payload directors report through the Deputy Commander.",
                ),
                (
                    "2. Off-Nominal Life Support Allocation",
                    "In scenario DELTA (loss of primary oxygen generation), "
                    "non-essential activity is suspended, habitat zones 3 and 4 "
                    "are isolated, and basal metabolic rations are enforced.",
                ),
                (
                    "3. Earth Communications Blackout Rules",
                    "During solar conjunction (2-week blackout when Sun is "
                    "between Mars and Earth), the outpost operates under full "
                    "autonomous local governance without real-time mission control "
                    "oversight.",
                ),
            ],
        ),
    ]

    manifest = []

    # 1. Generate the 18 digital PDFs
    for filename, title, category, sections in corpus_specs:
        create_digital_pdf(filename, title, sections)
        manifest.append(
            {
                "id": filename.replace(".pdf", ""),
                "filename": filename,
                "title": title,
                "category": category,
                "is_scanned": False,
                "mostly_tables": False,
            }
        )

    # 2. Generate the 1 scanned document (pdf_04)
    create_scanned_pdf("pdf_04_archival_log_scanned.pdf")
    manifest.append(
        {
            "id": "pdf_04_archival_log_scanned",
            "filename": "pdf_04_archival_log_scanned.pdf",
            "title": (
                "Restricted Archival Record: Declassified Technical Field Log "
                "(Sol 01-10)"
            ),
            "category": "Historical Records",
            "is_scanned": True,
            "mostly_tables": False,
        }
    )

    # 3. Generate the 1 table-heavy document (pdf_07)
    create_table_heavy_pdf("pdf_07_telemetry_matrix_tables.pdf")
    manifest.append(
        {
            "id": "pdf_07_telemetry_matrix_tables",
            "filename": "pdf_07_telemetry_matrix_tables.pdf",
            "title": "Master Telemetry Matrix & Engineering Tolerances",
            "category": "Telemetry & Systems",
            "is_scanned": False,
            "mostly_tables": True,
        }
    )

    # Sort manifest by filename
    manifest.sort(key=lambda x: x["filename"])

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} PDFs and wrote manifest to {manifest_path}")


if __name__ == "__main__":
    generate_all_20_pdfs()
