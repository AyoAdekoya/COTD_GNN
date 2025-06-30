#!/bin/bash
# Automates running the parse-netlist-to-scoap.py script for all contest designs
# Usage: ./get_scoapready.sh
# (Run this in the SCOAP working directory)

for i in {0..19}
do
    echo "Running design $i"
    python3 parse-netlist-to-scoap.py ../test-cases/trojan/design$i.v $i
done

for i in {20..29}
do
    echo "Running design $i"
    python3 parse-netlist-to-scoap.py ../test-cases/trojan_free/design$i.v $i
done

mkdir scoap_inputs
mkdir net_mappings
mkdir gate_mappings

mv scoap_input*.txt scoap_inputs
mv net_mapping*.txt net_mappings
mv gate_mapping*.txt gate_mappings
