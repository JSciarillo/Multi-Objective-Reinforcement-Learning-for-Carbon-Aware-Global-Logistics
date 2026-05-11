import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#load DC road subgraph with carbon/time edges 
G = ox.load_graphml('data/dc_subgraph_carbon.graphml')
nodes = list(G.nodes())

#edge attributes for EPA MOVES5 carbon emissions and travel time
CARBON_ATTR = 'carbon_HeavyTruck_evening_rush'
TIME_ATTR   = 'travel_time_evening_rush'

def weighted_route(G, origin, dest, w_carbon, w_time):
    """Find route minimizing w_carbon*carbon + w_time*time using Dijkstra."""
    def weight_fn(u, v, data):
      #use the first edge when there are multiple edges
        if isinstance(data, dict) and 0 in data:
            data = data[0]
        c = float(data.get(CARBON_ATTR, 0))
        t = float(data.get(TIME_ATTR,   0))
        #single cost for Dijkstra
        return w_carbon * c + w_time * t

    try:
        path = nx.dijkstra_path(G, origin, dest, weight=weight_fn)
        #find the CO2 and time along the chosen path
        total_co2  = 0.0
        total_time = 0.0
        for u, v in zip(path[:-1], path[1:]):
            ed = G.get_edge_data(u, v)
            if isinstance(ed, dict) and 0 in ed:
                ed = ed[0]
            total_co2  += float(ed.get(CARBON_ATTR, 0))
            total_time += float(ed.get(TIME_ATTR,   0))
        return total_co2, total_time, True
    except nx.NetworkXNoPath:
        #origin-destination pair is disconnected in the subgraph
        return 0, 0, False

#origin-destination pairs to sample
N_OD_PAIRS   = 50
#preference weight steps
N_PREF_SWEEP = 9

#sample origin-desination pairs
od_pairs = []
while len(od_pairs) < N_OD_PAIRS:
    o, d = random.sample(nodes, 2)
    if nx.has_path(G, o, d):
        od_pairs.append((o, d))

pareto_points = []
for o, d in od_pairs:
    for w_carbon in np.linspace(0, 1, N_PREF_SWEEP):
        w_time = 1.0 - w_carbon
        co2, time, reached = weighted_route(G, o, d, w_carbon, w_time)
        if reached:
            pareto_points.append({
                'co2':      co2,
                'time_min': time / 60,
                'w_carbon': w_carbon,
            })

df = pd.DataFrame(pareto_points)
print(f"Collected {len(df)} Pareto points")

#plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor='k')
ax.set_facecolor('k')

#color the points by their carbon preference weight
#green (low w_carbon) to red (high w_carbon)
scatter = ax.scatter(
    df['time_min'], df['co2'],
    c=df['w_carbon'], cmap='RdYlGn_r',
    s=60, alpha=0.7, edgecolors='white', linewidth=0.4,
)
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Carbon preference weight →', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

ax.set_xlabel('Travel Time (minutes)', color='white', fontsize=12)
ax.set_ylabel('CO₂ Emissions (g)', color='white', fontsize=12)
ax.set_title('Pareto Frontier: Speed vs. Emissions\n(Weighted Dijkstra, Downtown DC, Heavy Truck Rush Hour)',
             color='white', fontsize=13)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

plt.tight_layout()
plt.savefig('morl/pareto_front_dijkstra.png', dpi=200, bbox_inches='tight', facecolor='k')
plt.show()
