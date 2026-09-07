# Ares-IV Pressurized Exploration Rover Operations

## 1. Vehicle Dimensions & Mobility Chassis
The Ares-IV is a 6-wheeled pressurized crew rover designed for extended scientific sorties up to 500 km from base. The rover weighs 4,200 kg empty and 6,800 kg fully loaded with two astronauts, science payloads, and 14 sols of consumables. The chassis uses a rocker-bogie active leveling suspension capable of scaling 40 cm obstacles and traversing 32-degree slopes.

## 2. Power & Drivetrain
Each of the six titanium mesh wheels is driven by an independent 5 kW brushless DC hub motor with 100:1 planetary reduction gearing. Power is supplied by a 120 kWh Li-S battery pack supplemented by a 1.2 kW rear-mounted Radioisotope Thermoelectric Generator (RTG) fueled with Plutonium-238 (Pu-238). Maximum cruising speed on packed sand is 22 km/h.

## 3. Autonomous Obstacle Avoidance
Due to communication delays with base, the rover runs autonomous path planning using forward-looking stereo cameras, dual 64-beam solid-state lidar units, and inertial measurement units (IMUs). Real-time SLAM algorithms map terrain at 30 frames per second, detecting soft regolith trenches and boulders larger than 25 cm.

## 4. Emergency Tether & Towing
Rovers operate strictly in pairs on sorties exceeding 50 km. In the event of catastrophic motor drive failure, a secondary rover can establish a high-tensile Spectra towline and share life support umbilicals.