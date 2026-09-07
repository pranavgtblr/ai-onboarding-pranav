# Deep Space Optical & RF Communications System

## 1. Transmission Bands & Laser Comm (DSOC)
Odyssey Base maintains two primary communication links with Earth:
- Primary Link: Deep Space Optical Communications (DSOC) 1550 nm near-infrared laser transceiver (30 W laser power).
- Backup Link: High-gain 3.5-meter steerable parabolic reflector operating in the Ka-band (32 GHz RF).
The laser link achieves downlink throughput of up to 100 Mbps at closest orbital approach (0.5 AU) and 15 Mbps at conjunction (2.5 AU).

## 2. Propagation Delay & Routing
One-way light time latency between Earth and Mars ranges from 3.1 minutes (closest distance, 55 million km) to 22.4 minutes (farthest distance, 401 million km). All network communications use Delay-Tolerant Networking (DTN) protocols with Bundle Protocol (RFC 5050) over Licklider Transmission Protocol (LTP).

## 3. Solar Conjunction Blackout Period
Every 26 months, Mars and Earth are on opposite sides of the Sun (solar conjunction). When the Sun-Earth-Mars angle drops below 2 degrees, solar coronal plasma causes severe radio scintillation and laser diffraction. All non-critical transmissions are suspended for approximately 14 days.

## 4. Relay Orbiters
Surface communication to Earth is augmented by two orbital relays: the Mars Reconnaissance Orbiter II (MRO-II) and the Odyssey Relay Satellite in a 12-hour areostationary orbit.