#!/usr/bin/env python3
########################################################################
# trusthub_gates.py 
# Takes in a verilog netlist in the contest format and creates two files:
# 1) scoap_inputX.txt  – the netlist in SCOAP tool format
# 2) gate_mappingX.txt  – mapping gate instance names → output net
#
# Usage: python(3) trusthub_gates.py <verilog-netlist> <design_number>
########################################################################
#
# We use "+" to symbolize the outputs of smaller gates that make up a more complex gate
#
import re
import sys

if len(sys.argv) != 3:
    print("Usage: python3 parse-netlist-to-scoap.py <verilog-netlist> <design_number>")
    sys.exit(1)

input_file      = sys.argv[1]
design_number   = sys.argv[2]
scoap_file      = f"scoap_input{design_number}.txt"
gatemap_file    = f"gate_mapping{design_number}.txt"

# global state for net IDs
net_id_map      = {"1'b0": 0, "1'b1": 1}
net_id_counter  = 2
ind = 0

# parse and emit liberty.v cells (including SDFFX1)
# , get_net_id
def parse_liberty_gate(gtype, pd):
    """
    Decompose each liberty cell into SCOAP primitives.
    Always use the REAL Q (or QN) net as the base name, appending _1, _2, … for
    intermediate signals.
    Returns (out_net, [list_of_scoap_lines]) or None if not a known cell.
    """
    global ind
    def NET(port):
        val = pd.get(port)
        if (val == "1'b1"):
            return '1'
        elif (val == "1'b0"):
            return '0'
        elif (val is not None):
            return val.strip()
        else:
            return None

    lines = []

    #SDFFX1: 
    if gtype == "SDFFX1":
        ind += 1
        out = NET("Q")
        d = NET("D")
        si = NET("SI")
        se = NET("SE")
        clk = NET("CLK")
        lines.append(f"not+se{ind} = not({se})")
        lines.append(f"{out}+1 = and({d}, not+se{ind})")
        lines.append(f"{out}+2 = and({si}, {se})")
        lines.append(f"{out}+3 = or({out}+1, {out}+2)")
        lines.append(f"{out} = dffc({out}+3, {clk})")
        if (NET("QN") is not None):
            out1 = NET("QN")
            lines.append(f"{out1} = not({out})")
        return out, lines
    
    if gtype == "DFFX2":
        out = NET("Q")
        if (NET("QN") is not None):
            out = NET("QN")
        clk = NET("CLK")
        d = NET("D")
        lines.append(f"{out} = dffc({d}, {clk})")
        return out, lines
    
    if gtype == "DFFNX2":
        out = NET("Q")
        if (NET("QN") is not None):
            out = NET("QN")
        clk = NET("CLK")
        d = NET("D")
        lines.append(f"{out}+1 = not({clk})")
        lines.append(f"{out} = dffc({d}, {out}+1)")
        return out, lines
    
    # Asynch active low set DFF
    if gtype == "DFFASX1" or gtype == "DFFASX2":
        out = NET("Q")
        clk = NET("CLK")
        setb = NET("SETB")
        d = NET("D")
        lines.append(f"{out}+1 = not({setb})")
        lines.append(f"{out}+2 = dffc({d}, {clk})")
        lines.append(f"{out} = or({out}+1, {out}+2)")
        if (NET("QN") is not None):
            outN = NET("QN")
            lines.append(f"{outN} = not({out})")
        return out, lines

    # Asych active low reset DFF
    if gtype == "DFFARX1" or gtype == "DFFARX2":
        out = NET("Q")
        clk = NET("CLK")
        rstb = NET("RSTB")
        d = NET("D")
        lines.append(f"{out}+1 = not({rstb})")
        lines.append(f"{out} = dffcr({d}, {clk}, {out}+1)")
        if (NET("QN") is not None):
            outN = NET("QN")
            lines.append(f"{outN} = not({out})")
        return out, lines

    # OA222X1: Q = (IN1|IN2) & (IN3|IN4) & (IN5|IN6)    
    if gtype == "OA222X1":
        out = NET("Q")
        a = NET("IN1")
        b = NET("IN2")
        lines.append(f"{out}+1 = or({a},{b})")
        c = NET("IN3")
        d = NET("IN4")
        lines.append(f"{out}+2 = or({c},{d})")
        e = NET("IN5")
        f6 = NET("IN6")
        lines.append(f"{out}+3 = or({e},{f6})")
        lines.append(f"{out}+4 = and({out}+1,{out}+2)")
        lines.append(f"{out}   = and({out}+4,{out}+3)")
        return out, lines
    
    # OAI222X1: QN = !(IN1|IN2) & (IN3|IN4) & (IN5|IN6)
    if gtype == "OAI222X1":
        out = NET("QN")
        a = NET("IN1")
        b = NET("IN2")
        lines.append(f"{out}+1 = or({a},{b})")
        c = NET("IN3")
        d = NET("IN4")
        lines.append(f"{out}+2 = or({c},{d})")
        e = NET("IN5")
        f6 = NET("IN6")
        lines.append(f"{out}+3 = or({e},{f6})")
        lines.append(f"{out}+4 = and({out}+1,{out}+2)")
        lines.append(f"{out}+5 = and({out}+4,{out}+3)")
        lines.append(f"{out}   = not({out}+5)")

        return out, lines 

    # LSDNX1: Q = (D===0)?0:1
    # Could technically be seen as a buffer if we ignore D = X being pulled up to 1
    if gtype == "LSDNX1":
        out = NET("Q")
        d   = NET("D")

        lines.append(f"{out} = buff({d}")
        return out, lines

    # LSDNENX1: Q = D ? ENB : 1
    # rewrite as Q = or(not(D), ENB):
    #    n1 = not(D)
    #    Q  = or(n1, ENB)
    if gtype == "LSDNENX1":
        out = NET("Q")
        d   = NET("D")
        enb = NET("ENB")

        lines.append(f"{out}+1 = not({d})")
        lines.append(f"{out}   = or({out}+1,{enb})")

        return out, lines

    # NBUFFX*: simple buffer
    if gtype.startswith("NBUFF"):
        out = NET("Q")
        inp = NET("IN")
        lines.append(f"{out} = buff({inp})")
        return out, lines

    # INVX*: simple inverter
    if gtype.startswith("INV"):
        out = NET("QN")
        inp = NET("IN")
        lines.append(f"{out} = not({inp})")
        return out, lines

    # ISOLANDX1: Q = ISO ? D : 0  →  Q = and(D,ISO)
    if gtype == "ISOLANDX1":
        out = NET("Q")
        d   = NET("D")
        iso = NET("ISO")
        lines.append(f"{out} = and({d},{iso})")
        return out, lines
    
    # ISOLORX8
    if gtype == "ISOLORX8":
        out = NET("Q")
        d   = NET("D")
        iso = NET("ISO")
        lines.append(f"{out} = or({d},{iso})")
        return out, lines

    # 2-input gates: AND2X1, NAND2X*, OR2X1, NOR2X*, XOR2X1, XNOR2X1
    two_map = {
        "AND2X1":  ("and",  "Q"),
        "AND2X2":  ("and",  "Q"),
        "AND2X4":  ("and", "Q"),
        "NAND2X0": ("nand", "QN"),
        "NAND2X1": ("nand", "QN"),
        "NAND2X2": ("nand", "QN"),
        "NAND2X4": ("nand", "QN"),
        "OR2X1":   ("or",   "Q"),
        "OR2X2":   ("or",   "Q"),
        "OR2X4":   ("or",   "Q"),
        "NOR2X0":  ("nor",  "QN"),
        "NOR2X1":  ("nor",  "QN"),
        "NOR2X2":  ("nor",  "QN"),
        "NOR2X4": ("nor", "QN"),
        "XOR2X1":  ("xor",  "Q"),
        "XOR2X2": ("xor", "Q"),
        "XNOR2X1": ("xnor", "Q"),
        "XNOR2X2": ("xnor", "Q"),
    }
    if gtype in two_map:
        op, outp = two_map[gtype]
        out = NET(outp)
        a   = NET("IN1")
        b   = NET("IN2")
        lines.append(f"{out} = {op}({a},{b})")
        return out, lines

    # 3-input gates: AND3X*, NAND3X*, OR3X1, NOR3X*, XOR3X1, XNOR3X1
    three_map = {
        "AND3X1":  ("and",  "Q"),
        "AND3X2":  ("and",  "Q"),
        "AND3X4":  ("and",  "Q"),
        "NAND3X0": ("nand", "QN"),
        "NAND3X1": ("nand", "QN"),
        "NAND3X2": ("nand", "QN"),
        "NAND3X4": ("nand", "QN"),
        "OR3X1":   ("or",   "Q"),
        "OR3X2":   ("or",   "Q"),
        "NOR3X0":  ("nor",  "QN"),
        "NOR3X1":  ("nor",  "QN"),
        "NOR3X2":  ("nor",  "QN"),
        "NOR3X4":  ("nor",  "QN"),
        "XOR3X1":  ("xor",  "Q"),
        "XOR3X2":  ("xor",  "Q"),
        "XNOR3X1": ("xnor", "Q"),
        "XNOR3X2": ("xnor", "Q"),
    }
    if gtype in three_map:
        op, outp = three_map[gtype]
        out = NET(outp)
        i1  = NET("IN1")
        i2  = NET("IN2")
        i3  = NET("IN3")
        # build (i1 op i2) first, then op with i3
        lines.append(f"{out}+1 = {op}({i1},{i2})")
        lines.append(f"{out}   = {op}({out}+1,{i3})")
        return out, lines

    # 4-input gates: AND4X1, NAND4X0, OR4X1, NOR4X*, etc.
    four_map = {
        "AND4X1":  ("and",  "Q"),
        "AND4X2":  ("and",  "Q"),
        "AND4X4":  ("and",  "Q"),
        "NAND4X0": ("nand", "QN"),
        "NAND4X1": ("nand", "QN"),
        "NOR4X0":  ("nor",  "QN"),
        "NOR4X1":  ("nor",  "QN"),
        "OR4X1":   ("or",   "Q"),
        "OR4X2":   ("or",   "Q"),
        "OR4X4":   ("or",   "Q"),

    }
    if gtype in four_map:
        op, outp = four_map[gtype]
        out = NET(outp)
        i1  = NET("IN1")
        i2  = NET("IN2")
        i3  = NET("IN3")
        i4  = NET("IN4")
        # ((i1 op i2) op (i3 op i4))
        lines.append(f"{out}+1 = {op}({i1},{i2})")
        lines.append(f"{out}+2 = {op}({i3},{i4})")
        lines.append(f"{out}   = {op}({out}+1,{out}+2)")
        return out, lines

    # AO21X1: (IN1&IN2) | IN3
    if gtype == "AO21X1":
        out = NET("Q")
        a   = NET("IN1")
        b   = NET("IN2")
        c   = NET("IN3")
        lines.append(f"{out}+1 = and({a},{b})")
        lines.append(f"{out}   = or({out}+1,{c})")
        return out, lines
    
    # AOI21X1: !(IN1&IN2) | IN3
    if gtype == "AOI21X1" or gtype == "AOI21X2":
        out = NET("QN")
        a   = NET("IN1")
        b   = NET("IN2")
        c   = NET("IN3")
        lines.append(f"{out}+1 = and({a},{b})")
        lines.append(f"{out}+2 = or({out}+1,{c})")
        lines.append(f"{out}   = not({out}+2)")

        return out, lines

    # AO22X1: (IN1&IN2) | (IN3&IN4)
    if gtype == "AO22X1":
        out = NET("Q")
        a,b = NET("IN1"), NET("IN2")
        c,d = NET("IN3"), NET("IN4")
        lines.append(f"{out}+1 = and({a},{b})")
        lines.append(f"{out}+2 = and({c},{d})")
        lines.append(f"{out}   = or({out}+1,{out}+2)")
        return out, lines
    
    # AOI22X1: !(IN1&IN2) | (IN3&IN4)
    if gtype == "AOI22X1" or gtype == "AOI22X2":
        out = NET("QN")
        a,b = NET("IN1"), NET("IN2")
        c,d = NET("IN3"), NET("IN4")
        lines.append(f"{out}+1 = and({a},{b})")
        lines.append(f"{out}+2 = and({c},{d})")
        lines.append(f"{out}+3 = or({out}+1,{out}+2)")
        lines.append(f"{out}   = not({out}+3)")
        return out, lines

    # AO221X1: (IN1&IN2)|(IN3&IN4)|IN5
    if gtype == "AO221X1":
        out = NET("Q")
        a,b = NET("IN1"), NET("IN2")
        c,d = NET("IN3"), NET("IN4")
        e   = NET("IN5")
        lines.append(f"{out}+1 = and({a},{b})")
        lines.append(f"{out}+2 = and({c},{d})")
        lines.append(f"{out}   = or({out}+1, {out}+2, {e})")   # OR of the two pairs
        return out, lines
    
    # AOI221X1: !(IN1&IN2)|(IN3&IN4)|IN5
    if gtype == "AOI221X1":
        out = NET("QN")
        a,b = NET("IN1"), NET("IN2")
        c,d = NET("IN3"), NET("IN4")
        e   = NET("IN5")
        lines.append(f"{out}+1 = and({a},{b})")
        lines.append(f"{out}+2 = and({c},{d})")
        lines.append(f"{out}+3 = or({out}+1, {out}+2, {e})")   # OR of the two pairs
        lines.append(f"{out}   = not({out}+3)")      # OR with IN5
        return out, lines

    # AO222X1: three pairs OR’d
    if gtype == "AO222X1":
        out = NET("Q")
        pairs = [("IN1","IN2"), ("IN3","IN4"), ("IN5","IN6")]
        for idx,(p1,p2) in enumerate(pairs, start=1):
            a = NET(p1); b = NET(p2)
            lines.append(f"{out}+{idx} = and({a},{b})")
        # OR the three results
        lines.append(f"{out}     = or({out}+1,{out}+2,{out}+3)")
        return out, lines
    
    # AOI222X1: three pairs inverted OR
    if gtype == "AOI222X1":
        out = NET("QN")
        pairs = [("IN1","IN2"), ("IN3","IN4"), ("IN5","IN6")]
        for idx,(p1,p2) in enumerate(pairs, start=1):
            a = NET(p1); b = NET(p2)
            lines.append(f"{out}+{idx} = and({a},{b})")
        # OR the three results
        lines.append(f"{out}+4     = or({out}+1,{out}+2,{out}+3)")
        lines.append(f"{out} = not({out}+4)")
        return out, lines

    # OA221X1: (IN1|IN2)&(IN3|IN4)&IN5
    if gtype == "OA221X1":

        out = NET("Q")
        a = NET("IN1")
        b = NET("IN2")
        lines.append(f"{out}+1 = or({a},{b})")
        c = NET("IN3")
        d = NET("IN4")
        lines.append(f"{out}+2 = or({c},{d})")
        e = NET("IN5")
        lines.append(f"{out}+3 = and({out}+1,{out}+2)")
        lines.append(f"{out}   = and({out}+3,{e})")
        return out, lines
    
    # OAI221X1: !(IN1|IN2)&(IN3|IN4)&IN5
    if gtype == "OAI221X1":
        out = NET("QN")
        a = NET("IN1")
        b = NET("IN2")
        lines.append(f"{out}+1 = or({a},{b})")
        c = NET("IN3")
        d = NET("IN4")
        lines.append(f"{out}+2 = or({c},{d})")
        e = NET("IN5")
        lines.append(f"{out}+3 = and({out}+1,{out}+2)")
        lines.append(f"{out}+4 = and({out}+3,{e})")
        lines.append(f"{out}   = not({out}+4)")
        return out, lines

    # OA21X1: (IN1|IN2)&(IN3)
    if gtype == "OA21X1":
        out = NET("Q")
        a = NET("IN1")
        b = NET("IN2")
        lines.append(f"{out}+1 = or({a},{b})")
        c = NET("IN3")
        lines.append(f"{out}   = and({out}+1,{c})")
        return out, lines
    
    # OAI21X1: !(IN1|IN2)&(IN3)
    if gtype == "OAI21X1" or gtype == "OAI21X2":
        out = NET("QN")
        a = NET("IN1")
        b = NET("IN2")
        lines.append(f"{out}+1 = or({a},{b})")
        c = NET("IN3")
        lines.append(f"{out}+2 = and({out}+1,{c})")
        lines.append(f"{out}   = not({out}+2)")
        return out, lines

    # OA22X1: (IN1|IN2)&(IN3|IN4)
    if gtype == "OA22X1":
        out = NET("Q")
        a = NET("IN1")
        b = NET("IN2")
        lines.append(f"{out}+1 = or({a},{b})")
        c = NET("IN3")
        d = NET("IN4")
        lines.append(f"{out}+2 = or({c},{d})")
        lines.append(f"{out}   = and({out}+1,{out}+2)")
        return out, lines
    
    # OAI22X1: !(IN1|IN2)&(IN3|IN4)
    if gtype == "OAI22X1":
        out = NET("QN")
        a = NET("IN1")
        b = NET("IN2")
        lines.append(f"{out}+1 = or({a},{b})")
        c = NET("IN3")
        d = NET("IN4")
        lines.append(f"{out}+2 = or({c},{d})")
        lines.append(f"{out}+3 = and({out}+1,{out}+2)")
        lines.append(f"{out}   = not({out}+3)")
        return out, lines

    # MUX21X*: (IN1 & S~)|(IN2 & S)
    if gtype == "MUX21X2" or gtype == "MUX21X1":
        out = NET("Q")
        a = NET("IN1")
        b = NET("IN2")
        s = NET("S")
        lines.append(f"{out}+1 = not({s})")
        lines.append(f"{out}+2 = and({a},{out}+1)")
        lines.append(f"{out}+3 = and({b},{s})")
        lines.append(f"{out}   = or({out}+2, {out}+3)")
        return out, lines

    #MUX41X1: 
    if  gtype == "MUX41X1":
        out = NET("Q")
        a = NET("IN1")
        b = NET("IN2")
        c = NET("IN3")
        d = NET("IN4")
        s0 = NET("S0")
        s1 = NET("S1")
        lines.append(f"not+s0   = not({s0})")
        lines.append(f"not+s1   = not({s1})")
        lines.append(f"{out}+1  = and({a}, not+s0, not+s1)")
        lines.append(f"{out}+2  = and({b}, not+s0, {s1})")
        lines.append(f"{out}+3  = and({c}, {s0}, not+s1)")
        lines.append(f"{out}+4  = and({d}, {s0}, {s1})")
        lines.append(f"{out}+5  = or({out}+1, {out}+2, {out}+3)")
        lines.append(f"{out}    = or({out}+5, {out}+4)")
        return out, lines

    return None

# type: ignore

# read and partition
with open(input_file, "r") as f:
    text = f.read()


# collect IOs and gate outputs
input_ports     = []
output_ports    = []
gate_lines      = []
gate_output_map = {}

# extract inputs/outputs/wires (for mapping/ordering)
# pattern = re.compile(r"(input|output|wire)\s*(.+?);", re.DOTALL)
pattern = re.compile(r"(input|output|wire)\s*(\[[^\]]+\])?\s*(.+?);", re.DOTALL)

def expand_multibit(base_name, msb, lsb):
    nets = []
    step = -1 if msb > lsb else 1
    for i in range(msb, lsb + step, step):
        nets.append(f"{base_name}[{i}]")
    return nets

for kind, bus, names in pattern.findall(text):
    for n in names.split(','):
        net = n.strip()
        if not net:
            continue

        if bus:
            # Extract msb and lsb from bus
            bus = bus.strip()[1:-1]  # remove brackets
            msb_str, lsb_str = bus.split(':')
            msb, lsb = int(msb_str), int(lsb_str)

            expanded_nets = expand_multibit(net, msb, lsb)
        else:
            expanded_nets = [net]

        for expanded_net in expanded_nets:
            if kind == "input":
                input_ports.append(expanded_net)
            elif kind == "output":
                output_ports.append(expanded_net)
            # (add wire handling if needed)

# process each instance
# re.MULTILINE makes ^ match the start of each line instead of only 
# the start of the string.
# re.DOTALL makes . match newlines as well, so the (.*?) can span multiple 
# lines of port connections.
gate_re = re.compile(r"^\s*(\w+)\s+" + r"(\w+)\s*" + r"\(\s*(.*?)\s*\)\s*;", re.DOTALL | re.MULTILINE)
for cell_type, inst_name, port_block in gate_re.findall(text):
    pd = dict(re.findall(r"\.(\w+)\s*\(\s*([^)]+)\)", port_block))
    # print(pd)
    # print(port_block)
    result = parse_liberty_gate(cell_type, pd)
    if result:
        out_net, scoap_lines = result
        for line in scoap_lines:
            gate_lines.append((out_net, line))
        gate_output_map[inst_name] = (out_net)
        continue

# ---- ADD THIS ----
assign_re = re.compile(r"assign\s+((?:\\[^\s]+|\w[\w\[\]\.]*)+)\s*=\s*([^;]+);")
assign_lines = []

for lhs, rhs in assign_re.findall(text):
    if rhs.strip() == "1'b0":
        assign_lines.append(f"{lhs} = 0")
    elif rhs.strip() == "1'b1":
        assign_lines.append(f"{lhs} = 1")
    else:
        assign_lines.append(f"{lhs} = {rhs.strip()}")
# --------------------

# write scoap_inputX.txt
with open(scoap_file, "w") as f:
    f.write("input(0)\n")
    f.write("input(1)\n")
    f.write("input(x)\n")

    for net in input_ports:
        f.write(f"input({net})\n")

    for net in output_ports:
        f.write(f"output({net})\n")

    for _, expr in gate_lines:
        f.write(expr + "\n")

    for assign in assign_lines:
        f.write(assign + "\n")

# write mappings
with open(gatemap_file, "w") as f:
    for gname, net in sorted(gate_output_map.items()):
        f.write(f"{gname} -> {net}\n")

print(f"Generated {scoap_file}, {gatemap_file}")