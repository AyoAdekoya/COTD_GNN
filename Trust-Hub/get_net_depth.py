import networkx as nx
import sys
# from create_nx import parse_verilog_netlist
import pandas as pd

import networkx as nx
import pandas as pd

def add_distance_attributes(graph: nx.DiGraph) -> None:
    """
    Adds 'dist_from_pi' and 'dist_to_po' attributes to each node in the graph.
    Nodes of type PI and CONST are treated as sources for 'dist_from_pi'.
    Nodes of type PO are treated as sinks for 'dist_to_po'.
    """

    # Find PI and PO nodes
    pi_nodes = [
        n for n, attr in graph.nodes(data=True)
        if attr.get("gate_type") == "PI" or attr.get("type") == "CONST"
    ]
    po_nodes = [
        n for n, attr in graph.nodes(data=True)
        if attr.get("gate_type") == "PO"
    ]

    # From PI paths
    all_shortest_paths_from_pi = {}
    for pi in pi_nodes:
        lengths = nx.single_source_shortest_path_length(graph, pi)
        all_shortest_paths_from_pi[pi] = lengths

    # To PO paths (reverse graph)
    G_rev = graph.reverse(copy=False)
    all_shortest_paths_to_po = {}
    for po in po_nodes:
        lengths = nx.single_source_shortest_path_length(G_rev, po)
        all_shortest_paths_to_po[po] = lengths

    # Assign attributes
    for node in graph.nodes():
        # From PI
        dists_from_pis = [
            all_shortest_paths_from_pi[pi][node]
            for pi in pi_nodes
            if node in all_shortest_paths_from_pi[pi]
        ]
        dist_from_pi = min(dists_from_pis) if dists_from_pis else float("inf")

        # To PO
        dists_to_pos = [
            all_shortest_paths_to_po[po][node]
            for po in po_nodes
            if node in all_shortest_paths_to_po[po]
        ]
        dist_to_po = min(dists_to_pos) if dists_to_pos else float("inf")

        graph.nodes[node]["dist_from_pi"] = dist_from_pi
        graph.nodes[node]["dist_to_po"] = dist_to_po

def create_distance_dataframe(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: gate_name, gate_type, dist_from_pi, dist_to_po.
    """
    rows = []
    for n, attr in graph.nodes(data=True):
        row = {
            "gate": n,
            "gate_type": attr.get("type", "UNKNOWN"),
            "dist_from_pi": attr.get("dist_from_pi", float("inf")),
            "dist_to_po": attr.get("dist_to_po", float("inf")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

import pandas as pd

def merge_gate_dataframes(distance_df: pd.DataFrame, scoap_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """
    Merges distance dataframe with SCOAP dataframe from csvB.
    Only overlapping gates are kept.
    Warns about gates in csvB not in distance_df.
    Saves merged dataframe to CSV and returns it.
    """

    # Read csvB
    scoap_df = pd.read_csv(scoap_csv_path)

    # Check required columns
    required_cols = {"gate", "net_ID", "CC0", "CC1", "CO", "is_trojan"}
    if not required_cols.issubset(set(scoap_df.columns)):
        raise ValueError(f"csvB is missing one or more required columns: {required_cols}")

    # Find gates in csvB not in distance_df
    gates_in_distance = set(distance_df["gate"])
    gates_in_scoap = set(scoap_df["gate"])

    missing_gates = gates_in_scoap - gates_in_distance

    if missing_gates:
        print("⚠️ Warning: The following gates are in csvB but not in the distance dataframe:")
        for g in missing_gates:
            print(f"  - {g}")

    # Perform inner merge to keep only overlapping gates
    merged_df = pd.merge(
        scoap_df,
        distance_df,
        on="gate",
        how="inner"
    )

    # Save merged DataFrame to CSV
    merged_df.to_csv(output_csv_path, index=False)
    print(f"✅ Merged CSV written to: {output_csv_path}")

    return merged_df

if __name__ == "__main__":
    netlist_file = sys.argv[1]
    reference_file = sys.argv[2]
    scoap_csv = sys.argv[3]

    netlist_graph = parse_verilog_netlist(netlist_file, reference_file)

    add_distance_attributes(netlist_graph)
    df = create_distance_dataframe(netlist_graph)
    merged_df = merge_gate_dataframes(df, scoap_csv, 'final.csv')
