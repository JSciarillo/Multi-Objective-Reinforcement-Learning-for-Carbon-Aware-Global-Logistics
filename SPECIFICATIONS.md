Project Specifications: Carbon-Aware Logistics Routing Engine
Executive Summary
This document outlines the architecture, data engineering pipeline, and artificial intelligence roadmap for a Multi-Objective Reinforcement Learning (MORL) routing engine. The system is designed to navigate the Pareto frontier between delivery speed and carbon emissions ($CO_2e$), replacing traditional single-objective (fastest route) navigation.

Phase 1: Data Engineering & Pipeline Architecture
The foundational environment is constructed via a sequential pipeline of four modular Python scripts, converting physical urban infrastructure into a mathematically interactive directed graph.

1. Network Topology Extraction (src/extract_network.py)
Purpose: To acquire and structure the geometric "game board" for the routing agent.
Mechanism: Utilizes the osmnx library to query OpenStreetMap (OSM) APIs.
Functionality:
Targets a designated central coordinate (Downtown Washington D.C.).
Extracts a strictly drive directed network, preserving one-way street legality.
Automatically simplifies complex raw intersections into clean, singular nodes to prevent state-space explosion.
Prunes the network mathematically by radial distance to constrain the environment to a highly dense, computational-friendly ~1,000 node subgraph.
2. Topographical Elevation Integration (src/add_elevation.py)
Purpose: To integrate the Z-axis (verticality) into the routing logic, acknowledging that heavy vehicles consume exponentially more fuel on steep inclines.
Mechanism: Interfaces with the OpenTopoData API (querying NASA SRTM GL1 30-meter resolution datasets).
Functionality:
Extracts coordinates for all ~830 nodes and batches HTTP requests to bypass API rate limits.
Assigns absolute sea-level elevation to every intersection.
Computes the physical directed gradient (rise over run) for every connecting edge.
3. Traffic & Congestion Simulation (src/add_speeds.py)
Purpose: To mathematically model the realities of urban gridlock via Time-of-Day Decay Curves.
Mechanism: Heuristic speed imputation and temporal profiling.
Functionality:
Imputes missing free-flow speed limits based on OSM highway classification algorithms.
Applies a dynamic congestion_profiles dictionary to generate four distinct traffic states per edge:
free_flow_midnight (1.0x baseline speed)
lunch_traffic (0.85x speed degradation)
morning_rush (0.6x speed degradation)
evening_rush (0.5x speed degradation)
Calculates the corresponding travel time (in seconds) for each of these four distinct states.
4. Hybrid Carbon Calculation (src/calculate_carbon.py)
Purpose: To unify spatial distance, topographical hills, and temporal congestion into scalar Carbon Emission weights ($g CO_2e$) for multiple vehicle classes.
Mechanism: A hybrid integration of EPA baseline scaling and the MOVES5 (Motor Vehicle Emission Simulator) interpolation curve.
Functionality:
The MOVES5 Physics Engine: Interpolates a scientifically validated "U-shaped" curve mapping vehicle speed to fuel burn.
Temporal Integration: Extracts the 4 specific time-of-day speeds computed in add_speeds.py. By passing a slow evening_rush speed (e.g., 15 kph) to the MOVES5 curve, it intrinsically calculates the massive idling emission penalty. Passing a free_flow speed (e.g., 50 kph) yields optimal, low emissions.
Vehicle Scaling: Applies scaling factors and specific hill-climbing penalties to three distinct classes: HeavyTruck, DeliveryVan, and GasCar.
Output: Generates 12 distinct carbon weights per edge (3 vehicles × 4 times of day), allowing dynamic, multi-scenario routing.
5. Spatial Visualization (notebooks/Phase1_Visualization.ipynb)
Purpose: To visually validate the mathematical outputs of the pipeline.
Functionality: Uses matplotlib to render the directed graph as a high-contrast heatmap. Brighter/thicker corridors immediately highlight routes with severe carbon intensity (due to steep grades or heavy congestion), proving the pipeline's efficacy to human operators.
Phase 2: Model Building & Environment Formalization (Next Steps)
With the static graph constructed, Phase 2 focuses on building the "Physics Engine" that allows an Artificial Intelligence to interact with the map.

1. Formalizing the Markov Decision Process (MDP)
We will wrap the .graphml data into a standard Reinforcement Learning environment (e.g., OpenAI Gymnasium).

The State Space: We will use Graph Neural Networks (GNNs) to compress the map's intersections into 64-dimensional continuous vectors. The AI will receive this vector, plus the Current Time, to understand its surroundings.
The Action Space: We will write dynamic masking algorithms. When the AI is at an intersection, the environment will strictly limit its choices to legally connected, outgoing roads.
2. The Scalarized Reward Function
This is the core logic the AI will use to learn. The environment will hand out negative points (penalties) based on the formula: Reward = - (Alpha * Travel_Time) - (Beta * Carbon_Weight) By tweaking Alpha and Beta, logistics managers can control how aggressively the AI prioritizes green routes versus fast routes.

Phase 3 & 4: Reinforcement Learning & Evaluation
Phase 3 (Training): We will deploy a Deep Q-Network (DQN) agent into the Gymnasium environment. Through millions of simulated trips, it will learn to predict traffic decay curves and avoid steep hills based on the vehicle it is driving.
Phase 4 (Evaluation): We will run 50 random Origin/Destination trips. We will compare the AI's route against a standard "Google Maps" fastest-route algorithm (Dijkstra) to explicitly quantify exactly how many tons of carbon were saved, and at what cost to total delivery time.