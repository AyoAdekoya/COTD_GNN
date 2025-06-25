########################################################################
# parse-scoap-output.py
# Takes in the output of the scoap tool and the gate-net-id mapping file
# and creates a csv file with five columns: the gate number, it's correspoinding
# ID as used in the scoap tool, and the CC0, CC1, and CO scores for the 
# net coming out of that gate. 
#
# Usage: > python parse-scoap-output.py scoap-outputX.txt gate_output_mappingX.txt X
# where X is the design number of the netlist.
# For example, running 
# > python parse-scoap-output.py scoap-output7.txt gate_output_mapping7.txt 7
# will create a csv file called gate_scores7.csv with the SCOAP scores for the
# nets in design7.v
#
########################################################################

import re
import csv
import sys

# === Input files via command line ===
if len(sys.argv) != 5:
    print("Usage: python3 parse_scoap_output.py <scoap_output> <gate_mapping> <trojan_list> <design_number>")
    sys.exit(1)

# === File paths ===
scoap_output = sys.argv[1]
gate_map_file = sys.argv[2]
trojan_list = sys.argv[3]
design_number = sys.argv[4]
csv_output = f"gate_scores{design_number}.csv"

# === Step 1: Parse SCOAP net scores ===
scoap_scores = {}  # net_id (int) → (CC0, CC1, CO)

def clean_inf(val):
    return "#INF" if "\u221e" in val else val  # Handle ∞

with open(scoap_output, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()

    if re.match(r"\d+_\d+\[\d+\]:", line):
        break

    if re.match(r"\d+:", line):
        net_id = int(line[:-1])
        i += 1
        if i >= len(lines):
            break

        scoap_line = lines[i].strip()
        match = re.match(r"\(([^,]+),([^)]+)\)\s+(.+)", scoap_line)
        if match:
            cc0, cc1, co = match.groups()
            cc0 = clean_inf(cc0.strip())
            cc1 = clean_inf(cc1.strip())
            co = clean_inf(co.strip())
            scoap_scores[net_id] = (cc0, cc1, co)
    i += 1

# === Step 2: Parse gate → net ID mapping ===
gate_to_netid = {}

with open(gate_map_file, "r") as f:
    for line in f:
        match = re.match(r"(\w+)\s*->\s*(.+?)\s*\(ID:\s*(\d+)\)", line)
        if match:
            gate_name, _, net_id = match.groups()
            gate_to_netid[gate_name] = int(net_id)

# === Step 3: Parse trojan gate list ===
trojaned_gates = set()
with open(trojan_list, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

if lines[0] == "TROJANED":
    try:
        start = lines.index("TROJAN_GATES") + 1
        end = lines.index("END_TROJAN_GATES")
        trojaned_gates = set(lines[start:end])
    except ValueError:
        print("Error: TROJANED file is malformed.")
        sys.exit(1)

# === Step 4: Write final CSV ===
with open(csv_output, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gate", "net_ID", "CC0", "CC1", "CO", "is_trojan"])

    for gate, net_id in sorted(gate_to_netid.items()):
        cc0, cc1, co = scoap_scores.get(net_id, ("#N/A", "#N/A", "#N/A"))
        is_trojan = 1 if gate in trojaned_gates else 0
        writer.writerow([gate, net_id, cc0, cc1, co, is_trojan])
