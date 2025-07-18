# need a mapping of gates to their gate type like ib1 to buffer and so on
# need to make PO and PI nodes as well
# need to read the original .v file to get the wires that connect each node
# read csv to create the nodes, and use the wires as the only edges that exist in the graph

import sys
import csv
import os
import glob
import re
import ast
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from zScore import compute_scoap_stats, normalize_score


# Classifying gatenames into gatetypes
def gatetype(line: str) -> str | None:
    tokens = line.strip().split()
    if not tokens:
        return None

    # Extract the alphabetic prefix from the first token (gate name)
    prefix_match = re.match(r'[a-zA-Z]+', tokens[0].lower())
    if not prefix_match:
        return None

    prefix = prefix_match.group(0)

    if prefix.startswith("sdff"):
        return "scan"
    elif prefix == "dffas":
        return "async_set_ff"
    elif prefix in ("sdffar", "dffar", "dffcs", "dffarx"):
        return "async_reset_ff"
    elif prefix in ("dffle", "dffs", "dffnx", "dffx", "dff"):
        return "dff"
    elif prefix in ("buf", "ib", "nbuff", "nb1"):
        return "buffer"
    elif prefix in ("inv", "i", "not", "invx"):
        return "inverter"
    elif prefix == "hi":
        return "const1"
    elif prefix.startswith("lsdn"):
        return "shifter"
    elif prefix.startswith("iso"):
        return "isolation"
    elif prefix.startswith("oai"):
        return "oai"
    elif prefix.startswith("aoi"):
        return "aoi"
    elif prefix.startswith("ao"):
        return "ao"
    elif prefix.startswith("oa"):
        return "oa"
    elif prefix in ("nand", "nnd"):
        return "nand"
    elif prefix == "and":
        return "and"
    elif prefix.startswith("or"):
        return "or"
    elif prefix == "nor":
        return "nor"
    elif prefix == "xor":
        return "xor"
    elif prefix in ("xnor", "xnr"):
        return "xnor"
    elif prefix.startswith("mux") or "mxi" in prefix:
        return "mux"

    return None

# Get primary inputs and output nets from netlist
def get_primary(text):
    input_nets = []
    output_nets = []

    # Match lines like: input [7:0] A, B;
    pattern = re.compile(r'^\s*(?!//)(input|output)\s*(\[[^\]]+\])?\s*(.*?);', re.MULTILINE | re.DOTALL)

    def expand_multibit(base_name, msb, lsb):
        nets = []
        step = -1 if msb > lsb else 1
        for i in range(msb, lsb + step, step):
            nets.append(f"{base_name}[{i}]")
        return nets

    for kind, bus, names in pattern.findall(text):
        net_list = []
        split_names = []
        for n in names.split(','):
            net = n.strip()
            if not net:
                continue
            split_names.append(net)

        if bus:
            msb, lsb = map(int, re.findall(r"\d+", bus))
            for name in split_names:
                net_list.extend(expand_multibit(name, msb, lsb))
        else:
            net_list = split_names

        if kind == "input":
            input_nets.extend(net_list)
        else:
            output_nets.extend(net_list)

    return input_nets, output_nets

def create_graph(netlist_path, csv_path, mapping_path):
    # 1) Load gate attributes from CSV
    gate_attrs = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            name, cc0, cc1, co, trojan = row

            gate_attrs[name] = {
                # "cc0": normalize_score(cc0, "cc0"),
                # "cc1": normalize_score(cc1, "cc1"),
                # "co": normalize_score(co, "co"),
                "cc0": cc0,
                "cc1": cc1,
                "co": co,
                "is_trojan": int(trojan),
            }

    # 2) Load gate-to-net mapping
    gate_to_net = {}
    with open(mapping_path) as f:
        for line in f:
            if "->" not in line:
                continue
            gate, net = line.split("->")
            key = gate.strip()
            value = net.strip()
            try:
                value = ast.literal_eval(value)
            except:
                pass
            gate_to_net[key] = value

    # 3) Invert net-to-gate mapping
    net_to_gate = {}
    for gate, nets in gate_to_net.items():
        if isinstance(nets, (list, tuple)):
            for net in nets:
                net_to_gate[net] = gate
        else:
            net_to_gate[nets] = gate

    # 4) Parse netlist file and handle assign statements
    with open(netlist_path) as f:
        text = f.read()

    # fix this now!
    lines = text.splitlines()
    alias_map = {}

    # Match assign A = B;
    assign_re = re.compile(r'^\s*assign\s+(\S+)\s*=\s*(\S+)\s*;')
    for line in lines:
        match = assign_re.match(line)
        if match:
            lhs, rhs = match.groups()
            alias_map[lhs] = rhs

    def resolve_driver(net):
        # Follow assign aliases back to the actual driver.
        seen = set()
        while True:
            if net in net_to_gate:
                return net_to_gate[net]
            if net not in alias_map or net in seen:
                return None
            seen.add(net)
            net = alias_map[net]

    # 5) Create graph and add constants and primary inputs
    G = nx.DiGraph()
    input_ports, _ = get_primary(text)

    for const in ("CONST_0", "CONST_1"):
        if const in gate_attrs:
            attrs = gate_attrs[const].copy()
            attrs["gate_type"] = "CONST"
            G.add_node(const, **attrs)

    for pi in input_ports:
        if pi in gate_attrs:
            attrs = gate_attrs[pi].copy()
        else:
            attrs = {"cc0": None, "cc1": None, "co": None, "is_trojan": None}
        attrs["gate_type"] = "PI" #type:ignore
        G.add_node(pi, **attrs)

    for gate in gate_to_net:
        if gate in ("CONST_0", "CONST_1") or gate in input_ports:
            continue
        if gate not in G:
            attrs = gate_attrs.get(gate, {}).copy()
            attrs.setdefault("cc0", None)
            attrs.setdefault("cc1", None)
            attrs.setdefault("co", None)
            attrs.setdefault("is_trojan", None)
            attrs["gate_type"] = None
            G.add_node(gate, **attrs)

    # 6) Parse instantiations and connect edges
    inst_re = re.compile(r'^\s*(\w+)\s+(\\\S+|\w+)\s*\((.*?)\)\s*;', re.MULTILINE | re.DOTALL)

    for match in inst_re.finditer(text):
        
        # match = inst_re.match(line)
        # if not match:
        #     print("non matching line", line)
        #     continue

        prefix, inst_name, arg_str = match.groups()

        if inst_name not in G:
            # Likely a PO or unmapped instantiation
            continue

        # Determine the gate type
        gate_type = gatetype(prefix)
        G.nodes[inst_name]["gate_type"] = gate_type

        # Determine port style: named vs positional
        named_ports = re.search(r'\.\w+\s*\(', arg_str)

        if named_ports:
            # Named-port style: .A(net)
            port_pairs = re.findall(r'\.(\w+)\s*\(\s*([^)]+)\)', arg_str)
            port_map = {}
            for port, net in port_pairs:
                port_map[port] = net.strip()

            outs = [port_map[p] for p in ("Q", "QN", "Y") if p in port_map]
            ins = [net for p, net in port_map.items() if net not in outs]
        else:
            # Positional style: (out, in1, in2, ...)
            parts = []
            for item in arg_str.split(','):
                parts.append(item.strip())
            if not parts:
                continue
            outs = [parts[0]]
            ins = parts[1:]

        # Create edges from input drivers to this gate
        for net in ins:
            driver = resolve_driver(net)
            if driver:
                G.add_edge(driver, inst_name)

        for node in G.nodes():
            G.nodes[node]['outdegree'] = G.out_degree(node)
            G.nodes[node]['indegree']  = G.in_degree(node)
    return G


def nx_to_pyG(G, design_name):
    # create type set from looping through nodes
    # convert to list
    # have a mapping of nodeToindex and indexTonode to help with the pyG data
    # Create feature and label vectors
    # place them in appropriate format for a torch.geometric
    # 1. Create type set, dicts, etc
    types_set = {"scan", "async_set_ff", "async_reset_ff", "dff", "buffer", "inverter", 
                "const1", "shifter", "isolation", "oai", "aoi", "ao", "oa", "nand", "and", 
                "or", "nor", "xor", "xnor", "mux", "CONST", "PI"}
    typeToindex = {}
    types_list = sorted(types_set)
    nodeToindex = {}
    indexTonode = {}
     #map gates to a number instead of str
    for index, node in enumerate(G.nodes()):
        nodeToindex[node] = index

    #reverse mapping for when we want to get the gates back from index
    for node, index in nodeToindex.items():
        indexTonode[index] = node

    # 2. Map type to index
    for index, type in enumerate(types_list):
        typeToindex[type] = index
    
    # 3. Features and label vectors
    features = []
    labels = []
    num_types = len(types_list)

    for node in G.nodes():
        attrs = G.nodes[node]
        gate = attrs.get("gate_type")
        
        type_index = typeToindex[gate]
        type_tensor = torch.tensor(type_index, dtype=torch.long)
        # calling one hot encoding for tensors
        one_hot = F.one_hot(type_tensor, num_classes=num_types).float()
    
        # get SCOAP values
        cc0_val = attrs.get("cc0")
        cc1_val = attrs.get("cc1")
        co_val = attrs.get("co")

        cc0 = torch.tensor([cc0_val], dtype=torch.float)
        cc1 = torch.tensor([cc1_val], dtype=torch.float)
        co  = torch.tensor([co_val],  dtype=torch.float)
        indeg_val = attrs.get("indegree",  0)
        outdeg_val = attrs.get("outdegree", 0)
        indegree  = torch.tensor([indeg_val], dtype=torch.float)
        outdegree = torch.tensor([outdeg_val], dtype=torch.float)

        # add to features vector
        feature_vector = torch.cat([one_hot, cc0, cc1, co, indegree, outdegree],dim=0)
        features.append(feature_vector)
        # feature_vector = torch.cat([one_hot, cc0, cc1, co], dim=0)
        # features.append(feature_vector)

        # save its label
        is_trojan_val = attrs.get("is_trojan", 0)
        labels.append(is_trojan_val)

    # stack features and labels
    X = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.long) 

    # 4. edge index 
    edge_list = []
    for src, dst in G.edges():
        src_idx = nodeToindex[src]
        dst_idx = nodeToindex[dst]
        edge_list.append([src_idx, dst_idx])

    edge_index_tensor = torch.tensor(edge_list, dtype=torch.long)  # [num_edges, 2]
    edge_index = edge_index_tensor.t().contiguous()                # [2, num_edges]

    # 5) Assemble the PyG Data object
    data = Data(x=X, edge_index=edge_index, y=y)

    # 6) Attach mappings for later lookup
    data.node_to_index = nodeToindex
    data.index_to_node = indexTonode
    data.type_to_index = typeToindex
    data.design_name = design_name

    # build index_to_type
    index_to_type = {}
    for gate_type, idx in typeToindex.items():
        index_to_type[idx] = gate_type

    data.index_to_type = index_to_type

    return data

def printGraph(G):
    print("Printing nodes: ")
    for node in G.nodes(data=True):
        print(node)
        
    print("Printing edges: ")
    for u, v in G.edges():
        print(f"{u} -> {v}")

def create_all_data():
    verilog_files = []
    design_names = []
    data_objects = []
    verilog_files_free = []
    design_names_free = []
    data_objects_free = []

    # base_folder = "TH-Benchmarks"
    base_folder = "TH_Trojans"

    verilog_files = []
    design_names = []

    # Tj contest design files
    test_cases_folder = "Trojan_GNN/test-cases"

    for subfolder in ["trojan"]:
        sub_path = os.path.join(test_cases_folder, subfolder)
        if not os.path.isdir(sub_path):
            continue

        v_files = glob.glob(os.path.join(sub_path, "*.v"))
        for v_file in v_files:
            filename = os.path.basename(v_file)
            # Extract number from designXXX.v
            match = re.match(r"design(\d+)\.v", filename)
            if match:
                design_num = match.group(1)
                verilog_files.append(v_file)
                design_names.append(design_num)

    # Tj free contest designs
    for subfolder in ["trojan_free"]:
        sub_path = os.path.join(test_cases_folder, subfolder)
        if not os.path.isdir(sub_path):
            continue

        v_files = glob.glob(os.path.join(sub_path, "*.v"))
        for v_file in v_files:
            filename = os.path.basename(v_file)
            # Extract number from designXXX.v
            match = re.match(r"design(\d+)\.v", filename)
            if match:
                design_num = match.group(1)
                verilog_files_free.append(v_file)
                design_names_free.append(design_num)

    # Tj trusthub files
    for design_folder in os.listdir(base_folder):
        design_path = os.path.join(base_folder, design_folder)
        if not os.path.isdir(design_path):
            continue

        # Handle potential "double folder" case
        subfolder_path = os.path.join(design_path, design_folder)
        if os.path.isdir(subfolder_path):
            # Use the inner folder instead
            design_path = subfolder_path

        # Save design name
        design_name = design_folder

        # Figure out which pattern to use for this design
        if design_name.startswith("RS232"):
            # print("skipping RS232")
            # continue
            pattern = os.path.join(design_path, "src", "90nm", "*.v")
            pattern2 = None
        elif design_name.startswith("s") or design_name.startswith("wb_conmax"):
            # print("skipping s or wb")
            # continue
            pattern = os.path.join(design_path, "src", "TjIn", "*.v")
            pattern2 = os.path.join(design_path, "src", "TjFree", "*.v")
        elif design_name.startswith("TRIT-TC") or design_name.startswith("TRIT-TS"):
            # These folders contain design subfolders directly
            subdesign_folders = os.listdir(design_path)
            for sub in subdesign_folders:
                sub_path = os.path.join(design_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                if sub == 'original_designs':
                    v_files_free = glob.glob(os.path.join(sub_path, "*.v"))
                    if v_files_free:
                        verilog_files_free.extend(v_files_free)
                        for v_file in v_files_free:
                            match = re.search(r"([a-zA-Z]\d+)\.v$", v_file)
                            if match:
                                design_name_free = match.group(1) + "free"

                            design_names_free.append(design_name_free) # type: ignore
                else:
                    v_files = glob.glob(os.path.join(sub_path, "*.v"))
                    if v_files:
                        verilog_files.extend(v_files)
                        design_names.extend([sub] * len(v_files))
            continue  # Skip to next top-level folder

        else:
            continue  # Skip unknown designs

        # Find .v files matching pattern
        v_files = glob.glob(pattern)
        v_files_free = []
        if pattern2:
            v_files_free = glob.glob(pattern2)
        if v_files:
            verilog_files.extend(v_files)
            design_names.extend([design_name] * len(v_files))
        if v_files_free:
            verilog_files_free.extend(v_files_free)
            design_name_free = design_name.split('-')[0] + "-free"
            design_names_free.extend([design_name_free] * len(v_files_free))

    def create_data(netlist_path, scoap_csv_path, gate_mapping_path, design_name):
        # Your custom processing logic here
        print("Processing files:")
        print(f"  Netlist: {netlist_path}")
        print(f"  SCOAP CSV: {scoap_csv_path}")
        print(f"  Gate mapping: {gate_mapping_path}")
        G = create_graph(netlist_path, scoap_csv_path, gate_mapping_path)
        print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        try:
            data = nx_to_pyG(G, design_name)
        except:
            print(f"Cannot run nx to pyG for file {netlist_path}")
            data = None
        print("-" * 60)
        return data

    missing_files = []

    for netlist_file, design_name in zip(verilog_files, design_names):
        gate_map_file = f"Trojan_GNN/Trust-Hub/Gate_Mappings/gate_mapping{design_name}.txt"
        scoap_file    = f"Trojan_GNN/Trust-Hub/Scoap_Scores/gate_scores{design_name}.csv"

        if not (os.path.exists(gate_map_file) and os.path.exists(scoap_file)):
            missing_files.append(design_name)
            print(f"Skipping {design_name} — missing mapping or SCOAP file")
            continue

        data_objects.append(create_data(netlist_file, scoap_file, gate_map_file, design_name))

    for netlist_file, design_name in zip(verilog_files_free, design_names_free):
        gate_map_file = f"Trojan_GNN/Trust-Hub/Gate_Mappings/gate_mapping{design_name}.txt"
        scoap_file    = f"Trojan_GNN/Trust-Hub/Scoap_Scores/gate_scores{design_name}.csv"

        if not (os.path.exists(gate_map_file) and os.path.exists(scoap_file)):
            missing_files.append(design_name)
            print(f"Skipping {design_name} — missing mapping or SCOAP file")
            continue

        data_objects_free.append(create_data(netlist_file, scoap_file, gate_map_file, design_name))

    return data_objects, data_objects_free

# Command-line execution 
if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python CSVtoPyG.py <netlist.v> <attrs.csv> <mapping.txt>")
    #     sys.exit(1)

    # netlist, attrs_csv, mapping = sys.argv[1:]

    compute_scoap_stats(folder_path="Trojan_GNN/Trust-Hub/Scoap_Scores")

    # G = create_graph(netlist, attrs_csv, mapping)
    # print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # print("Printing nodes: ")
    # for node in G.nodes(data=True):
    #     print(node)
        
    # print("Printing edges: ")
    # for u, v in G.edges():
    #     print(f"{u} -> {v}")

    # print()
    # print("PyG")
    # data = nx_to_pyG(G)
    # print(data)

    data_list, data_free_list = create_all_data()

    for data in data_list:
        if data:
            print(f"Tj data object:------------------")
            print("Number of nodes:      ", data.num_nodes)
            print("Number of edges:      ", data.num_edges)
            print("Feature size per node:", data.num_node_features)

            print("\nFirst 5 node feature vectors:")
            assert(data.num_nodes is not None)
            assert(data.x is not None)
            for i in range(min(5, data.num_nodes)):
                print(f" Node {i:3d} ({data.index_to_node[i]}):", data.x[i].tolist())

            print("\nFirst 10 labels (is_trojan):")
            assert isinstance(data.y, torch.Tensor)
            for i in range(min(10, data.num_nodes)):
                print(f" Node {i:3d} ({data.index_to_node[i]}):", data.y[i].item())

    # printing tj free data list
    for data in data_free_list:
        if data:
            print(f"Tj free data object:---------------")
            print("Number of nodes:      ", data.num_nodes)
            print("Number of edges:      ", data.num_edges)
            print("Feature size per node:", data.num_node_features)

            print("\nFirst 5 node feature vectors:")
            assert(data.num_nodes is not None)
            assert(data.x is not None)
            for i in range(min(5, data.num_nodes)):
                print(f" Node {i:3d} ({data.index_to_node[i]}):", data.x[i].tolist())

            print("\nFirst 10 labels (is_trojan):")
            assert isinstance(data.y, torch.Tensor)
            for i in range(min(10, data.num_nodes)):
                print(f" Node {i:3d} ({data.index_to_node[i]}):", data.y[i].item())

