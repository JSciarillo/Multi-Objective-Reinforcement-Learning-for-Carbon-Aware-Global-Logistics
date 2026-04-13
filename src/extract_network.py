import osmnx as ox
import networkx as nx
import os

ox.settings.cache_folder = "../cache"
ox.settings.use_cache = True # Ensures caching is enabled

"""
    Extracts a 1,000-node representative subgraph of Downtown Washington D.C.
    Uses OSMnx to fetch the drive network, simplifies the topology, and saves it.
"""
def extract_dc_subgraph(save_path="../data/dc_subgraph.graphml"):

    # Define a point in downtown D.C. (near the White House/downtown area)
    point = (38.8977, -77.0365)

    print("Fetching D.C. drivable network...")
    # Fetch drivable road network
    # We use bidirectional=False inherently when pulling drive networks,
    # but OSMnx creates a MultiDiGraph to preserve one-way streets.
    G = ox.graph_from_point(point, dist=3000, network_type="drive", simplify=True)
    
    # Get basic stats
    print(f"Network extracted. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    

    if len(G.nodes) > 1200:
        print("Pruning to approximately 1,000 nodes...")
        # Sort nodes by distance from the center point
        nodes = ox.graph_to_gdfs(G, edges=False)
        # Point geometry
        from shapely.geometry import Point
        import geopandas as gpd
        center = Point(point[1], point[0])
        # Project to local CRS to measure distance in meters
        #nodes_proj = ox.project_gdf(nodes)
        nodes_proj = ox.projection.project_gdf(nodes)
        center_gdf = gpd.GeoDataFrame(geometry=[center], crs=nodes.crs)
        #center_proj = ox.project_gdf(center_gdf).geometry.iloc[0]
        center_proj = ox.projection.project_gdf(center_gdf).geometry.iloc[0]
        
        nodes_proj['dist_to_center'] = nodes_proj.geometry.distance(center_proj)
        # Keep the 1000 closest nodes
        closest_1000 = nodes_proj.sort_values('dist_to_center').head(1000)
        
        # Induce subgraph
        G = G.subgraph(closest_1000.index).copy()
        print(f"Pruned network. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
        
        # Clean up isolated nodes or small components after pruning
        #G = ox.utils_graph.get_largest_component(G, strongly=True)
        G = ox.truncate.largest_component(G, strongly=True)
        print(f"Largest strongly connected component. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the graph
    print(f"Saving graph to {save_path}...")
    ox.save_graphml(G, save_path)
    print("Done.")
    return G

if __name__ == "__main__":
    extract_dc_subgraph()
