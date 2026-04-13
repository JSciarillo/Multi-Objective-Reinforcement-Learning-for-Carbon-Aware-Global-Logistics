import osmnx as ox
import networkx as nx
import numpy as np

def calculate_carbon_weights(graph_path="../data/dc_subgraph_speeds.graphml", save_path="../data/dc_subgraph_carbon.graphml"):
    """
    Calculates the Carbon Weight for every edge based on:
    W(e) = distance(e) * EF(k) * grade_factor(e) * congestion_factor(e,t)
    """
    print(f"Loading graph from {graph_path}...")
    G = ox.load_graphml(graph_path)
    
    # EF(k): Baseline Emission Factors for Multiple Vehicle Classes (g/m)
    # Derived from EPA approximations (converting grams per mile to grams per meter)
    vehicle_factors = {
        'HeavyTruck': 1360.0 / 1609.34, # ~0.845 g/m
        'DeliveryVan': 720.0 / 1609.34, # ~0.447 g/m
        'GasCar': 411.0 / 1609.34       # ~0.255 g/m
    }
    
    # Congestion Emission Multipliers based on the Time-of-Day Decay Curves in add_speeds.py
    # If speed is degraded by 0.5x, you spend 2x as long on the road, increasing idling emissions.
    congestion_multipliers = {
        'free_flow_midnight': 1.0,           # 100% speed -> 1.0x emissions
        'morning_rush': 1.0 / 0.6,           # 60% speed -> ~1.66x emissions
        'lunch_traffic': 1.0 / 0.85,         # 85% speed -> ~1.17x emissions
        'evening_rush': 1.0 / 0.5            # 50% speed -> 2.0x emissions
    }
    
    for u, v, k, data in G.edges(keys=True, data=True):
        distance_m = float(data.get('length', 10.0))
        grade = float(data.get('grade', 0.0))
        
        # Calculate emissions for every vehicle type
        for vehicle, ef_k in vehicle_factors.items():
            
            # grade_factor(e): Heavy vehicles consume exponentially more fuel on steep inclines.
            # We scale the hill penalty based on the vehicle type. 
            # Trucks struggle more on hills than a light gas car.
            if vehicle == 'HeavyTruck':
                hill_penalty = 15.0
            elif vehicle == 'DeliveryVan':
                hill_penalty = 8.0
            else: # GasCar
                hill_penalty = 4.0
                
            if grade > 0:
                grade_factor = 1.0 + (grade * hill_penalty)
            else:
                grade_factor = max(0.8, 1.0 + (grade * (hill_penalty / 3.0))) 
                
            # Base emissions for this specific vehicle on this specific physical road
            base_emissions = distance_m * ef_k * grade_factor
            
            # Now, apply the temporal congestion for every time of day
            for time_profile, traffic_penalty in congestion_multipliers.items():
                
                final_carbon_weight = base_emissions * traffic_penalty
                
                # Save as a distinct attribute, e.g., 'carbon_DeliveryVan_morning_rush'
                attribute_name = f"carbon_{vehicle}_{time_profile}"
                data[attribute_name] = final_carbon_weight
                
        # Legacy compatibility for the visualization notebook which looks for 'carbon_weight_rushhour'
        data['carbon_weight_rushhour'] = data['carbon_HeavyTruck_evening_rush']

    print(f"Saving graph with computed carbon weights to {save_path}...")
    ox.save_graphml(G, save_path)
    print("Done.")

if __name__ == "__main__":
    calculate_carbon_weights()