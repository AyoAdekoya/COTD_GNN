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