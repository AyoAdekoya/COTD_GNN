import sys
import re

# Paths
netlist_file = sys.argv[1]
existing_file = "TH_gates.txt"
output_file = "TH-gates-new.txt"

# Load existing gates
existing_gates = set()
with open(existing_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            existing_gates.add(line)

# Set for new gates
new_gates = set()

# Read and process netlist
with open(netlist_file, 'r') as f:
    data = f.read()

instances = data.split(';')

# Regex to match gate type at start
gate_regex = re.compile(r'^\s*([A-Z0-9]+)', re.IGNORECASE)

for inst in instances:
    inst = inst.strip()
    if not inst:
        continue
    match = gate_regex.match(inst)
    if match:
        gate_type = match.group(1)
        if gate_type not in existing_gates:
            new_gates.add(gate_type)

# Save new unique gates
with open(output_file, 'w') as out:
    for gt in sorted(new_gates):
        out.write(gt + '\n')

print(f"Found {len(new_gates)} new gate types not in '{existing_file}', saved to '{output_file}'.")