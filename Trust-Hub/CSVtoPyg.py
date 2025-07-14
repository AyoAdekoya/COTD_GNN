# need a mapping of gates to their gate type like ib1 to buffer and so on
# need to make PO and PI nodes as well
# need to read the original .v file to get the wires that connect each node
# read csv to create the nodes, and use the wires as the only edges that exist in the graph

import sys
import csv
import re
import ast
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from zScore.py import compute_scoap_stats, normalize_score


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
    elif prefix in ("sdffar", "dffar", "dffcs"):
        return "async_reset_ff"
    elif prefix in ("dffle", "dffs", "dffnx", "dffx", "dff"):
        return "dff"
    elif prefix in ("buf", "ib1", "nbuff", "nb1"):
        return "buffer"
    elif prefix in ("inv", "i", "not"):
        return "inverter"
    elif prefix == "hi1":
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
    elif prefix == "or":
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
    pattern = re.compile(r'(?m)^\s*(?!//)(input|output)\s*(\[[^\]]+\])?\s*(.+?);')

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
                "cc0": float(cc0),
                "cc1": float(cc1),
                "co": float(co),
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
    # inst_re = re.compile(r'^\s*(\w+)\s+([^\s(]+)\s*\(\s*([^)]+)\)\s*;')
    inst_re = re.compile(r'^\s*(\w+)\s+([^\s(]+)\s*\((.*?)\)\s*;', re.DOTALL | re.MULTILINE)

    for line in lines:
        match = inst_re.match(line)
        if not match:
            continue

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
    return G


def nx_to_pyG(G):
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
        
        # add to features vector
        feature_vector = torch.cat([one_hot, cc0, cc1, co], dim=0)
        features.append(feature_vector)

        # save its label
        is_trojan_val = attrs.get("is_trojan", 0)
        labels.append(is_trojan_val)

        # stack to 
    return data



# Command-line execution 
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_graph.py <netlist.v> <attrs.csv> <mapping.txt>")
        sys.exit(1)

    netlist, attrs_csv, mapping = sys.argv[1:]
    G = create_graph(netlist, attrs_csv, mapping)
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print("Printing nodes: ")
    for node in G.nodes(data=True):
        print(node)
        
    print("Printing edges: ")
    for u, v in G.edges():
        print(f"{u} -> {v}")

