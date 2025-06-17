import re
import sys
import networkx as nx

def flatten_bus(signal, width):
    return [f"{signal}[{i}]" for i in range(width)]

def parse_verilog_netlist(filepath):
    with open(filepath, 'r') as f:
        verilog = f.read()

    # 1. Extract primary inputs
    input_lines = re.findall(r'input\s+[^;]+;', verilog)
    primary_inputs = set()
    for line in input_lines:
        line = line.strip(';').replace('input', '').strip()
        if '[' in line:
            match = re.match(r'\[(\d+):(\d+)\]\s+(.+)', line)
            msb, lsb, names = int(match.group(1)), int(match.group(2)), match.group(3)
            width = abs(msb - lsb) + 1
            signals = [s.strip() for s in names.split(',')]
            for signal in signals:
                primary_inputs.update(flatten_bus(signal, width))
        else:
            signals = [s.strip() for s in line.split(',')]
            primary_inputs.update(signals)

    # 2. Extract primary outputs
    output_lines = re.findall(r'output\s+[^;]+;', verilog)
    primary_outputs = set()
    for line in output_lines:
        line = line.strip(';').replace('output', '').strip()
        if '[' in line:
            match = re.match(r'\[(\d+):(\d+)\]\s+(.+)', line)
            msb, lsb, names = int(match.group(1)), int(match.group(2)), match.group(3)
            width = abs(msb - lsb) + 1
            signals = [s.strip() for s in names.split(',')]
            for signal in signals:
                primary_outputs.update(flatten_bus(signal, width))
        else:
            signals = [s.strip() for s in line.split(',')]
            primary_outputs.update(signals)

    # 3. Extract gates
    gate_pattern = re.compile(r'(?P<type>\w+)\s+(?P<name>g\d+)\s*\((?P<content>[^;]+)\);')
    pin_conn_pattern = re.compile(r'\.(\w+)\(([^)]+)\)')

    gates = []
    for match in gate_pattern.finditer(verilog):
        gate_type = match.group('type')
        gate_name = match.group('name')
        content = match.group('content').strip()

        if gate_type.lower() == 'dff':
            pins = dict(pin_conn_pattern.findall(content))
            inputs = []
            outputs = []
            for pin, net in pins.items():
                net = net.strip()
                if pin.upper() == 'Q':
                    outputs.append(net)
                else:
                    inputs.append(net)
        else:
            parts = [s.strip() for s in content.split(',')]
            outputs = [parts[0]]
            inputs = parts[1:]

        gates.append((gate_name, gate_type, inputs, outputs))

    # 4. Build the graph
    G = nx.DiGraph()
    signal_to_gate = {}

    # Add all gates and outputs
    for name, gate_type, inputs, outputs in gates:
        G.add_node(name, type=gate_type)
        for out in outputs:
            signal_to_gate[out] = name

    # Add PI nodes
    for pi in primary_inputs:
        G.add_node(pi, type='PI')

    # Add PO nodes
    for po in primary_outputs:
        G.add_node(po, type='PO')

    # Add constant nodes
    G.add_node("CONST_0", type="CONST")
    G.add_node("CONST_1", type="CONST")

    # Add edges from inputs/constants to gates
    for name, _, inputs, _ in gates:
        for inp in inputs:
            inp = inp.strip()
            if inp == "1'b0":
                G.add_edge("CONST_0", name)
            elif inp == "1'b1":
                G.add_edge("CONST_1", name)
            elif inp in signal_to_gate:
                G.add_edge(signal_to_gate[inp], name)
            elif inp in primary_inputs:
                G.add_edge(inp, name)
            # else: might be undefined or internal net not yet processed

    # Add edges from last gate to PO
    for po in primary_outputs:
        driver = signal_to_gate.get(po)
        if driver:
            G.add_edge(driver, po)

    return G

filepath = sys.argv[1]

G = parse_verilog_netlist(filepath)

print("Printing nodes: ")
for node, attrs in G.nodes(data=True):
    print(f"{node}: {attrs['type']}")

print("Printing edges: ")
for u, v in G.edges():
    print(f"{u} → {v}")


# # Visualization - not great for large inputs

# import matplotlib.pyplot as plt

# node_colors = []
# for n in G.nodes:
#     t = G.nodes[n]['type']
#     if t == 'PI':
#         node_colors.append('lightgreen')
#     elif t == 'PO':
#         node_colors.append('salmon')
#     else:
#         node_colors.append('lightblue')

# labels = {n: f"{n}\n{G.nodes[n]['type']}" for n in G.nodes}

# pos = nx.spring_layout(G, seed=42)
# nx.draw(G, pos, with_labels=False, node_size=10, node_color=node_colors)
# # nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
# plt.title("Circuit Netlist Graph: PI → Gates → PO")
# plt.show()
