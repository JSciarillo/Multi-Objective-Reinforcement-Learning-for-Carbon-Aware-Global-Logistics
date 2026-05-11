import osmnx as ox
import networkx as nx
import rasterio
import geopandas as gpd
from shapely.geometry import box
import os
import requests

def download_srtm_tile_for_bounds(minx, miny, maxx, maxy, save_dir="data/elevation"):

    import urllib.request
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Coordinates for DC: Lat 38.89, Lon -77.03
    # Tile would be N38W078
    lat = int(miny)
    lon = int(minx) # Note minx is negative, so math.floor gives -78
    
    # SRTM GL1 (30m) filenames: N38W078.hgt
    # To reliably get the tile without AWS auth, we can use a known mirror or API.
    # Since downloading and processing large .hgt files might be brittle, let's use the OpenTopoData API
    # to add node elevations, which mimics Google's API but is free and doesn't require a raster file on disk.
    
    pass

def add_elevation_to_graph(graph_path="../data/dc_subgraph.graphml", save_path="../data/dc_subgraph_elev.graphml"):
    """
    Reads the graph, adds elevation to nodes, and calculates edge grades.
    """
    print(f"Loading graph from {graph_path}...")
    G = ox.load_graphml(graph_path)
    
    # We will use the OpenTopoData API (SRTM 30m dataset) to get elevations for nodes.
    # It acts like the Google Maps Elevation API.
    # Since there are ~830 nodes, we can query them in batches.
    import time
    import pandas as pd
    
    nodes = ox.graph_to_gdfs(G, edges=False)
    
    print(f"Fetching elevations for {len(nodes)} nodes via OpenTopoData API...")
    
    # OpenTopoData API limits: 1 call per second, max 100 locations per call.
    locations = []
    node_ids = []
    
    for idx, row in nodes.iterrows():
        locations.append(f"{row['y']},{row['x']}")
        node_ids.append(idx)
        
    elevations = {}
    batch_size = 100
    
    for i in range(0, len(locations), batch_size):
        batch_locs = "|".join(locations[i:i+batch_size])
        batch_ids = node_ids[i:i+batch_size]
        
        url = f"https://api.opentopodata.org/v1/srtm30m?locations={batch_locs}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            for j, result in enumerate(data['results']):
                elevations[batch_ids[j]] = result['elevation']
                
            time.sleep(1.5) # Be nice to the free API
        except Exception as e:
            print(f"Error fetching elevation batch {i}: {e}")
            # Fallback: assume 0 elevation if API fails (e.g. rate limit)
            for j in range(len(batch_ids)):
                elevations[batch_ids[j]] = 0.0
                
    # Add elevation to nodes
    nx.set_node_attributes(G, elevations, name="elevation")
    
    # Now use OSMnx to calculate edge grades
    print("Calculating edge grades...")
    G = ox.elevation.add_edge_grades(G)
    
    # The 'grade' attribute is the directed grade.
    # We will also compute 'grade_abs' just in case.
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'grade' not in data:
            data['grade'] = 0.0
        data['grade_abs'] = abs(data['grade'])
        
    print(f"Saving graph with elevation and grades to {save_path}...")
    ox.save_graphml(G, save_path)
    print("Done.")

if __name__ == "__main__":
    add_elevation_to_graph()
