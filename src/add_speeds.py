import osmnx as ox
import networkx as nx
import pandas as pd

def add_speeds_and_travel_times(graph_path="../data/dc_subgraph_elev.graphml", save_path="../data/dc_subgraph_speeds.graphml"):
    """
    Imputes missing speed limits on edges, calculates free-flow travel time,
    and implements temporal congestion factors.
    """
    print(f"Loading graph from {graph_path}...")
    G = ox.load_graphml(graph_path)
    
    print("Imputing edge speed limits...")
    # add_edge_speeds imputes missing speeds based on highway type
    G = ox.routing.add_edge_speeds(G)
    
    print("Calculating baseline free-flow travel times...")
    # calculates travel time based on length and speed
    G = ox.routing.add_edge_travel_times(G)
    
    # Extract edges to verify what was added
    edges = ox.graph_to_gdfs(G, nodes=False)
    
    print("Applying temporal congestion factors (Time-of-Day Decay Curves)...")
    # Implementing "Option C": A dynamic dictionary representing different traffic states
    # based on the time of day, rather than a single static blanket factor.
    
    congestion_profiles = {
        'free_flow_midnight': 1.0,  # 12:00 AM - 5:00 AM: 100% speed
        'morning_rush': 0.6,        # 7:00 AM - 9:00 AM: 60% speed
        'lunch_traffic': 0.85,      # 12:00 PM - 2:00 PM: 85% speed
        'evening_rush': 0.5         # 4:00 PM - 7:00 PM: 50% speed (heavy commute)
    }
    
    for u, v, k, data in G.edges(keys=True, data=True):
        # Ensure we have a base speed
        speed_kph = float(data.get('speed_kph', 30.0))
        length_m = float(data.get('length', 10.0))
        
        # Calculate speed and travel time for EVERY profile
        for profile_name, factor in congestion_profiles.items():
            # Degrade the speed
            degraded_speed_kph = speed_kph * factor
            # Save the degraded speed to the edge
            data[f'speed_kph_{profile_name}'] = degraded_speed_kph
            
            # Calculate the travel time in seconds
            # time(s) = distance(m) / speed(m/s)
            degraded_speed_mps = degraded_speed_kph * 1000 / 3600
            
            if degraded_speed_mps > 0:
                data[f'travel_time_{profile_name}'] = length_m / degraded_speed_mps
            else:
                data[f'travel_time_{profile_name}'] = float('inf')
                
        # For legacy compatibility with calculate_carbon.py which looks for 'travel_time_rush_hour'
        # we will alias 'evening_rush' to 'rush_hour'
        data['speed_kph_rush_hour'] = data['speed_kph_evening_rush']
        data['travel_time_rush_hour'] = data['travel_time_evening_rush']
    
    print(f"Saving graph with speeds and dynamic congestion logic to {save_path}...")
    ox.save_graphml(G, save_path)
    print("Done.")

if __name__ == "__main__":
    add_speeds_and_travel_times()