import osmnx as ox
import networkx as nx
import os
from shapely.geometry import Point
import geopandas as gpd

ox.settings.cache_folder = "../cache"
ox.settings.use_cache = True

# Default DC params (used by the original pipeline)
DC_CENTER     = (38.8977, -77.0365)
DC_TARGET     = 2000
DC_RADIUS_M   = 5000


def extract_subgraph(center, save_path, target_nodes=2000, radius_m=5000, city_name="subgraph"):
    """Generic OSM subgraph extractor used by both DC and the terrain study."""
    print(f"Fetching {city_name} drivable network (radius={radius_m}m)...")
    G = ox.graph_from_point(center, dist=radius_m, network_type="drive", simplify=True)
    print(f"Raw network: {len(G.nodes)} nodes, {len(G.edges)} edges")

    if len(G.nodes) > target_nodes + 200:
        print(f"Pruning to {target_nodes} nodes closest to center...")
        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        center_gdf = gpd.GeoDataFrame(geometry=[Point(center[1], center[0])], crs=nodes_gdf.crs)
        nodes_proj  = ox.projection.project_gdf(nodes_gdf)
        center_proj = ox.projection.project_gdf(center_gdf).geometry.iloc[0]
        nodes_proj["dist"] = nodes_proj.geometry.distance(center_proj)
        keep = nodes_proj.sort_values("dist").head(target_nodes).index
        G = G.subgraph(keep).copy()
        print(f"After pruning: {len(G.nodes)} nodes, {len(G.edges)} edges")

    G = ox.truncate.largest_component(G, strongly=True)
    print(f"Largest strongly connected component: {len(G.nodes)} nodes, {len(G.edges)} edges")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ox.save_graphml(G, save_path)
    print(f"Saved to {save_path}")
    return G


def extract_dc_subgraph(save_path="../data/dc_subgraph.graphml"):
    """Wrapper retained for backward-compat with the original pipeline."""
    return extract_subgraph(
        center=DC_CENTER,
        save_path=save_path,
        target_nodes=DC_TARGET,
        radius_m=DC_RADIUS_M,
        city_name="DC",
    )


if __name__ == "__main__":
    extract_dc_subgraph()
