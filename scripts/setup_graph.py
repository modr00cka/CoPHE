import json
import pandas as pd
import copy
from tqdm import tqdm

def load_graph(json_path):
    with open(json_path) as f:
        graph_dict = json.load(f)
        return graph_dict
    
def depth_first(graph, key, depth = 0):
    ancestors = graph[key]["parents"]
    depths = [depth]
    if len(ancestors)>0:
        new_depth = depth+1
        for ancestor in ancestors:
            depths.append(depth_first(graph, ancestor, new_depth))
    return max(depths)
    
def find_leaves(graph):
    leaves = list(set(graph.keys()))
    return leaves

def depth_setup(graph, leaves):
    max_depth = 0
    print("Setting up depths")
    for key in tqdm(graph):
        graph[key]["depth"] = depth_first(graph, key)
        if max_depth< graph[key]["depth"]:
            max_depth =  graph[key]["depth"]
        graph[key]["leaves"] = [key]

        for potential_leaf in leaves:
            if key in graph[potential_leaf]["parents"]:
                graph[key]["leaves"].append(potential_leaf)

        
    print("Cleaning up descendants")
    for level in tqdm(range(max_depth, -1, -1)):
        for key in graph:
            if graph[key]["depth"]<max_depth:
                extra_leaves = []
                for leaf in graph[key]["leaves"]:
                    extra_leaves+=(graph[leaf]["leaves"])
                graph[key]["leaves"] = list(set(graph[key]["leaves"] +extra_leaves))
    return graph
    
def find_max_depth(graph):
    max_depth = 0
    for key in graph:
        max_depth = max(graph[key]["depth"], max_depth)
    return max_depth

def translate_depths(graph):
    max_depth = find_max_depth(graph)
    translated_depth_graph = copy.deepcopy(graph)
    for key in translated_depth_graph:
        translated_depth_graph[key]["depth"] = max_depth - graph[key]["depth"]
    return translated_depth_graph
    
def set_up_triplets(graph, target_depth):
    max_depth = find_max_depth(graph)
    assert target_depth<=max_depth
    triplets = []
    for key in graph:
        if graph[key]["depth"]==target_depth:
            for prediction_layer_class in graph[key]["leaves"]:
                triplets.append((prediction_layer_class, graph[key]["depth"], key))
    return triplets

def describe_graph(graph):
    max_depth = find_max_depth(graph)
    df_prep = []
    for i in range(max_depth+1):
        triplets_at_depth = set_up_triplets(graph, i)
        df_prep += triplets_at_depth
    return pd.DataFrame(df_prep, columns=['source', 'level', 'target'])

def full_setup(in_json_path, out_csv_path):
    json_graph = load_graph(in_json_path)
    leaves = find_leaves(json_graph)
    print(f"Num of leaves: {len(leaves)}")
    json_graph = depth_setup(json_graph, leaves)
    translated_graph = translate_depths(json_graph)
    describe_graph(translated_graph).to_csv(out_csv_path, index=False)

if __name__ == "__main__":
    json_path = ""
    out_path = ""
    full_setup(json_path, out_path)