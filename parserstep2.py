import aiger
import torch
import networkx as nx
from torch_geometric.data import Data
import os

def get_node_features_and_graph(aig_path, area_label, delay_label):
    if not os.path.exists(aig_path):
        raise FileNotFoundError(f"AIG file not found: {aig_path}")

    # 1. Load AIG
    try:
        circ = aiger.load(aig_path)
    except Exception as e:
        print(f"Error loading AIG: {e}")
        return None
    
    # 2. Build a NetworkX Graph for topology analysis
    G = nx.DiGraph()
    
    # Add Inputs
    for inp in circ.inputs:
        G.add_node(inp, type='PI')
        
    # Add Latches
    for latch in circ.latches:
        G.add_node(latch, type='LATCH')

    # --- FIX 1: Handle various gate types (AND vs Inverter) ---
    for gate_name, gate_obj in circ.node_map.items():
        G.add_node(gate_name, type='GATE')
        
        sources = []
        if hasattr(gate_obj, 'left') and hasattr(gate_obj, 'right'):
            sources = [gate_obj.left, gate_obj.right]
        elif hasattr(gate_obj, 'input'):
            sources = [gate_obj.input]
        else:
            # Fallback for weird objects
            continue

        for src in sources:
            src_name = src.name if hasattr(src, 'name') else str(src)
            if src_name not in ['True', 'False']:
                G.add_edge(src_name, gate_name)
    
    # --- FIX 2: robust Output Driver Identification ---
    # We need to know which NODE drives an output to set the 'Is_PO' feature.
    output_drivers = set()
    
    # Try 'omap' (Output Map) which is standard in many versions
    if hasattr(circ, 'omap'):
        for lit in circ.omap.values():
            name = lit.name if hasattr(lit, 'name') else str(lit)
            output_drivers.add(name)
            
    # Check if 'outputs' is a dictionary (name -> literal)
    elif isinstance(circ.outputs, dict):
        for lit in circ.outputs.values():
            name = lit.name if hasattr(lit, 'name') else str(lit)
            output_drivers.add(name)
            
    # Fallback: Assume output names match node names (rare)
    else:
        output_drivers = set(circ.outputs)

    # 3. Compute Complex Features
    try:
        node_levels = {}
        for node in nx.topological_sort(G):
            parents = list(G.predecessors(node))
            if not parents:
                node_levels[node] = 0
            else:
                node_levels[node] = max(node_levels.get(p, 0) for p in parents) + 1
    except nx.NetworkXUnfeasible:
        node_levels = {n: 0 for n in G.nodes()}

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # 4. Assemble PyTorch Tensors
    node_list = list(G.nodes())
    node_to_idx = {name: i for i, name in enumerate(node_list)}
    
    x_features = []

    for node in node_list:
        # Feature 1: Is PI?
        is_pi = 1.0 if node in circ.inputs else 0.0
        
        # Feature 2: Is PO? (Check against our safe list)
        is_po = 1.0 if node in output_drivers else 0.0
        
        # Feature 3: Fan-in
        f_in = float(in_degrees.get(node, 0))
        
        # Feature 4: Fan-out
        f_out = float(out_degrees.get(node, 0))
        
        # Feature 5: Max Level (Depth)
        level = float(node_levels.get(node, 0))
        
        x_features.append([is_pi, is_po, f_in, f_out, level])

    # Build Edge Index
    src_list = []
    dst_list = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            src_list.append(node_to_idx[u])
            dst_list.append(node_to_idx[v])

    x = torch.tensor(x_features, dtype=torch.float)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    y = torch.tensor([[area_label, delay_label]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)

if __name__ == "__main__":
    # Update path
    aig_path = "/workspace/ckarfa/Bhabesh/GNNImplementatio/abc/adder.aig"
    
    try:
        data = get_node_features_and_graph(aig_path, 898.31, 2613.78)
        if data:
            print(f"Success! Processed {aig_path}")
            print(f"Features for first node: {data.x[0]}")
            torch.save(data, "processed_data.pt")
            print("Saved to processed_data.pt")
    except Exception as e:
        import traceback
        traceback.print_exc()
