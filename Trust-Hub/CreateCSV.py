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
from TjLabeler import extract_tj_contest
# from trusthub_gates import gate_output_map

# === Input files via command line ===
# if len(sys.argv) != 5:
#     print("Usage: python3 parse_scoap_output.py <scoap_output> <gate_mapping> <trojan_list> <design_number>")
#     sys.exit(1)

# === File paths ===
scoap_output = sys.argv[1]
gate_map_file = sys.argv[2]
trojan_labels = sys.argv[3]
design_number = sys.argv[4]
csv_output = f"gate_scores{design_number}.csv"

# === Step 1: Parse SCOAP net scores ===
scoap_scores = {}
nets = []

def clean_inf(val):
    return "#INF" if "\u221e" in val else val  # Handle ∞

with open(scoap_output, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()

    if re.match(r"\d+_\d+\[\d+\]:", line):
        break

    if re.match(r"^\S+:", line):
        net_name = line[:-1]
        nets.append(net_name)
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
            scoap_scores[net_name] = (cc0, cc1, co)
    i += 1

# === Step 2: Parse gate → net mapping, store as net → gate instead ===
net_to_gate = {}

with open(gate_map_file, "r") as f:
    for line in f:
        match = re.match(r"(\S+)\s*->\s*(\S+)", line)
        if match:
            gate_name, net_name = match.groups()
            net_to_gate[net_name] = gate_name

# # === Step 3: Parse trojan gate list ===
# import glob


# def get_verilog_filepaths(design_folder):
#     # Use glob to find all files in a folder
#     file_list = glob.glob(f"../../{design_folder}/{design_folder}/src/TjIn/*")
#     if not file_list:
#         print("No files found!")
#         sys.exit()
#     else:
#         # Open the first file found
#         tj_filepath = file_list[0]

#     # Use glob to find all files in a folder
#     file_list2 = glob.glob(f"../../{design_folder}/{design_folder}/src/TjFree/*")
#     if not file_list2:
#         print("No files found!")
#         sys.exit()
#     else:
#         # Open the first file found
#         tj_free_filepath = file_list2[0]
    
#     return tj_filepath, tj_free_filepath


# # Get trojan gates
# if design_folder.startswith("RS"):    trojaned_gates = extract_tj_readme(f"../../{design_folder}/{design_folder}/Read me.txt/")
# elif design_folder.startswith("s"):
#     tj_filepath, tj_free_filepath = get_verilog_filepaths(design_folder)
#     trojaned_gates = extract_tj_netlist(tj_filepath, tj_free_filepath)
# elif design_folder == "test-cases":
#     trojaned_gates = extract_tj_contest(f"../{design_folder}/trojan/result{design_number}.txt")

trojaned_gates = extract_tj_contest(trojan_labels)

print(trojaned_gates)
print(net_to_gate)

# === Step 4: Write final CSV ===
with open(csv_output, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gate", "CC0", "CC1", "CO", "is_trojan"])

    for net in nets:
        if net not in net_to_gate:
            print(f"Skipping net {net}, no gate found in mapping")
            continue
        gate_name = net_to_gate[net]
        cc0, cc1, co = scoap_scores.get(net, ("#N/A", "#N/A", "#N/A"))
        is_trojan = 1 if gate_name in trojaned_gates else 0
        writer.writerow([gate_name, cc0, cc1, co, is_trojan])

