import osmnx as ox
import networkx as nx
import numpy as np

def calculate_carbon_weights(graph_path="../data/dc_subgraph_speeds.graphml", save_path="../data/dc_subgraph_carbon.graphml"):
    """
    Calculates the Carbon Weight for every edge based on:
    W(e) = distance(e) * EF(k) * grade_factor(e)
    (Congestion is naturally handled by the MOVES5 curve when given lower speeds)
    """
    print(f"Loading graph from {graph_path}...")
    G = ox.load_graphml(graph_path)
    
    # HYBRID APPROACH: MOVES5 Physics Curve + Time-of-Day Traffic States
    
    # MOVES5 Emission Curve (mph vs. grams per km)
    MOVES5_CURVE = [
        (2.5,  745.5), (7.5,  652.8), (12.5, 590.2),
        (17.5, 541.7), (22.5, 484.8), (27.5, 441.3),
        (32.5, 416.4), (37.5, 396.5), (42.5, 385.4),
        (47.5, 379.2), (52.5, 379.0), (57.5, 381.8),
        (62.5, 387.6), (67.5, 397.4), (72.5, 412.0),
    ]
    speeds_mph = np.array([s for s, _ in MOVES5_CURVE])
    ef_g_per_km = np.array([e for _, e in MOVES5_CURVE])

    def speed_to_ef_grams_per_meter(speed_kph):
        """Interpolates MOVES5 curve and returns emissions in grams per meter."""
        # Fallback to 15 kph if speed is 0 or negative to prevent math errors
        speed_kph = max(15.0, float(speed_kph)) 
        speed_mph = speed_kph / 1.60934
        # Interpolate to get g/km, then divide by 1000 for g/m
        g_per_km = float(np.interp(speed_mph, speeds_mph, ef_g_per_km))
        return g_per_km / 1000.0

    # Vehicle scaling factors relative to the baseline Heavy Truck MOVES5 curve
    VEHICLE_EF_SCALE = {
        'HeavyTruck': 1.0,
        'DeliveryVan': 0.529,
        'GasCar': 0.302,
    }
    
    HILL_PENALTY = {
        'HeavyTruck': 15.0, 
        'DeliveryVan': 8.0, 
        'GasCar': 4.0
    }

    # Time profiles generated in add_speeds.py
    time_profiles = ['free_flow_midnight', 'morning_rush', 'lunch_traffic', 'evening_rush']

    print("Computing Hybrid MOVES5 + Time-of-Day carbon weights...")
    for u, v, k, data in G.edges(keys=True, data=True):
        distance_m = float(data.get('length', 10.0))
        grade = float(data.get('grade', 0.0))

        # We will loop through the 4 times of day to get the specific speed
        for time_profile in time_profiles:
            # Grab the specific degraded speed calculated in add_speeds.py
            # If missing, fallback to 35 kph
            speed_kph = float(data.get(f'speed_kph_{time_profile}', 35.0))
            
            # Convert that exact speed to a baseline emission using the MOVES5 physics curve
            base_ef_g_per_m = speed_to_ef_grams_per_meter(speed_kph)
            
            # Now loop through the 3 vehicle types to scale it
            for vehicle, scale in VEHICLE_EF_SCALE.items():
                
                # Scale the base emission for the specific vehicle
                ef_k = base_ef_g_per_m * scale
                
                # Calculate the physical hill penalty for this vehicle
                hill_penalty = HILL_PENALTY[vehicle]
                if grade > 0:
                    grade_factor = 1.0 + (grade * hill_penalty)
                else:
                    grade_factor = max(0.8, 1.0 + (grade * (hill_penalty / 3.0)))
                    
                # Final calculation: Distance * Vehicle Scaled Emission * Hill Penalty
                # Note: We do NOT use 'congestion_multipliers' here because the MOVES5 curve 
                # already intrinsically models the idling penalty when we pass it the lower speed!
                final_carbon_weight = distance_m * ef_k * grade_factor
                
                # Save it to the graph
                attribute_name = f"carbon_{vehicle}_{time_profile}"
                data[attribute_name] = final_carbon_weight

        # Legacy compatibility for the visualization notebook
        data['carbon_weight_rushhour'] = data.get('carbon_HeavyTruck_evening_rush', 0.0)

    print(f"Saving graph with computed carbon weights to {save_path}...")
    ox.save_graphml(G, save_path)
    print("Done.")

if __name__ == "__main__":
    calculate_carbon_weights()