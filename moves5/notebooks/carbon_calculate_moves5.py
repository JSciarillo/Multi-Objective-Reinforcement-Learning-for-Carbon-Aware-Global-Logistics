import osmnx as ox
import numpy as np

def calculate_carbon_weights(graph_path="data/dc_subgraph_speeds.graphml",
                              save_path="moves5/data/dc_subgraph_carbon_moves5.graphml"):
    print(f"Loading graph from {graph_path}...")
    G = ox.load_graphml(graph_path)

    MOVES5_CURVE = [
        (2.5,  745.5), (7.5,  652.8), (12.5, 590.2),
        (17.5, 541.7), (22.5, 484.8), (27.5, 441.3),
        (32.5, 416.4), (37.5, 396.5), (42.5, 385.4),
        (47.5, 379.2), (52.5, 379.0), (57.5, 381.8),
        (62.5, 387.6), (67.5, 397.4), (72.5, 412.0),
    ]
    speeds_mph = np.array([s for s, _ in MOVES5_CURVE])
    ef_g_per_km = np.array([e for _, e in MOVES5_CURVE])

    def speed_to_ef(speed_kph):
        speed_mph = float(speed_kph) / 1.60934
        return float(np.interp(speed_mph, speeds_mph, ef_g_per_km))

    HIGHWAY_SPEED = {
        'motorway': 105, 'motorway_link': 80,
        'trunk': 85,     'trunk_link': 65,
        'primary': 65,   'primary_link': 50,
        'secondary': 50, 'secondary_link': 40,
        'tertiary': 40,  'tertiary_link': 35,
        'residential': 25, 'living_street': 15,
        'unclassified': 40, 'service': 20,
    }

    HILL_PENALTY = {'HeavyTruck': 15.0, 'DeliveryVan': 8.0, 'GasCar': 4.0}
    VEHICLE_EF_SCALE = {
        'HeavyTruck': 1.0,
        'DeliveryVan': 0.529,
        'GasCar': 0.302,
    }

    print("Computing MOVES5 carbon weights...")
    for u, v, k, data in G.edges(keys=True, data=True):
        distance_m = float(data.get('length', 10.0))
        distance_km = distance_m / 1000.0
        grade = float(data.get('grade', 0.0))

        speed_kph = data.get('speed_kph', None)
        if speed_kph is None or float(speed_kph) <= 0:
            hw = data.get('highway', 'unclassified')
            if isinstance(hw, list): hw = hw[0]
            speed_kph = HIGHWAY_SPEED.get(hw, 35.0)

        base_ef = speed_to_ef(speed_kph)

        for vehicle, scale in VEHICLE_EF_SCALE.items():
            ef_k = base_ef * scale
            hill_penalty = HILL_PENALTY[vehicle]
            if grade > 0:
                grade_factor = 1.0 + (grade * hill_penalty)
            else:
                grade_factor = max(0.8, 1.0 + (grade * (hill_penalty / 3.0)))

            carbon_weight = distance_km * ef_k * grade_factor
            data[f"carbon_{vehicle}_moves5"] = carbon_weight

    print(f"Saving graph to {save_path}...")
    ox.save_graphml(G, save_path)
    print("Done.")

if __name__ == "__main__":
    calculate_carbon_weights()
