import re
import sys
import networkx as nx

# creating index strings based on the given list and len
# e.g if signal = data, this creates a list [data[0], data[1],...]
def flatten_bus(signal, width):
    return [f"{signal}[{i}]" for i in range(width)]

gates = []

def parse_verilog_netlist(filepath, reffilepath):
    with open(filepath, 'r') as f:
        verilog = f.read()

    # 1. Extract primary inputs
    # r means raw string literal, input is the word the re looks for,
    # \s+ says to include whitespaces, [^;]+ reads everything till ;,
    # the ;, stops the reading at a semicolon
    input_lines = re.findall(r'input\s+[^;]+;', verilog)
    primary_inputs = set()
    for line in input_lines:
        # strip functions removes front whitespace or whatever char 
        # u pass into it
        line = line.strip(';').replace('input', '').strip()
        # if we have a multibit signal
        if '[' in line:
            # (\d+) is reading the first # and the second # into group 1 & 2
            # (.+) just puts the rest of the line in a 3rd group
            match = re.match(r'\[(\d+):(\d+)\]\s+(.+)', line)
            msb, lsb, names = int(match.group(1)), int(match.group(2)), match.group(3) # type: ignore
            width = msb - lsb + 1
            # get however many signals instantiated on one line
            signals = [s.strip() for s in names.split(',')]\
            # update primary input with newly read ones
            for signal in signals:
                primary_inputs.update(flatten_bus(signal, width))
        # if not multibit signal
        else:
            signals = [s.strip() for s in line.split(',')]
            primary_inputs.update(signals)

    # Doing the same as above for external outputs
    # 2. Extract primary outputs
    output_lines = re.findall(r'output\s+[^;]+;', verilog)
    primary_outputs = set()
    for line in output_lines:
        line = line.strip(';').replace('output', '').strip()
        if '[' in line:
            match = re.match(r'\[(\d+):(\d+)\]\s+(.+)', line)
            msb, lsb, names = int(match.group(1)), int(match.group(2)), match.group(3) # type: ignore
            width = msb - lsb + 1
            signals = [s.strip() for s in names.split(',')]
            for signal in signals:
                primary_outputs.update(flatten_bus(signal, width))
        else:
            signals = [s.strip() for s in line.split(',')]
            primary_outputs.update(signals)



    # 3. Extract gates
    # Separates the input lines into 3 groups; Pattern, Name, and Contents
    # (?P<type>\w+), the w represents word (AND, OR, NOT, etc), (?P<name>g\d+)
    # takes g and then the number. Contents are just the input/outputs 
    gate_pattern = re.compile(r'(?P<type>\w+)\s+(?P<name>g\d+)\s*\((?P<content>[^;]+)\);')
    # for the dff with the .RN signals
    # \. matches the ., (\w+) groups the word, \( matches (, ([^)]+) is
    # to read til ), \) matches )
    pin_conn_pattern = re.compile(r'\.(\w+)\(([^)]+)\)')

    # finditer returns an iterator of objects in gate_pattern
    for match in gate_pattern.finditer(verilog):
        gate_type = match.group('type')
        gate_name = match.group('name')
        content = match.group('content').strip()

        if gate_type == 'dff':
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
            # splits the (output, input, input) into output input lists
            parts = [s.strip() for s in content.split(',')]
            outputs = [parts[0]]
            inputs = parts[1:]


        trojanedgates = trojanlabel(gates, reffilepath)
        if gate_name in trojanedgates:
            label = "Trojaned"
            gates.append((gate_name, gate_type, inputs, outputs, label))
        else:
            label = "Not_Trojaned"
            gates.append((gate_name, gate_type, inputs, outputs, label))

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

    # Add edges from inputs/gates to gates
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

def trojanlabel(gates, filepath):
    trojanedgates = []
    with open(filepath, 'r') as file:
        # ref= file.read()
        next(file)
        next(file)
        for line in file:
            trojanedgates.append(line)

    return trojanedgates


filepath = sys.argv[1]
# reffilepath = 'reference/reference/result0.txt'
reffilepath = sys.argv[2]
G = parse_verilog_netlist(filepath, reffilepath)

print("Printing nodes: ")
for node, attrs in G.nodes(data=True):
    print(f"{node}: {attrs['type']}")

print("Printing edges: ")
for u, v in G.edges():
    print(f"{u} → {v}")
