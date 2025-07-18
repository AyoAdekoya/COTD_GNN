#!/bin/bash
# Automates running the parse-netlist-to-scoap.py script for all contest designs
# Usage: ./get_scoapready.sh
# (Run this in the Trust-Hub working directory)

# for i in {0..19}
# do
#     echo "Running design $i"
#     python3 CreateScoap.py ../test-cases/trojan/design$i.v $i
# done

# for i in {20..29}
# do
#     echo "Running design $i"
#     python3 CreateScoap.py ../test-cases/trojan_free/design$i.v $i
# done


# Define prefixes
prefixes=(s13207_T s15850_T s35932_T s1423_T)

for prefix in "${prefixes[@]}"; do
    # Loop through suffixes 001-099
    for num in $(seq -w 400 619); do
        design="${prefix}${num}"
        
        # Example: check if a design file exists
        if [[ -f "../../TH_Trojans/TRIT-TS/TRIT-TS/${design}/${design}.v" ]]; then
            echo "Processing design file: ${design}"
            
            # Example command
            python CreateScoap.py "../../TH_Trojans/TRIT-TS/TRIT-TS/${design}/${design}.v" ${design}
        else
            echo "File not found: ${design}.v"
        fi
    done
done


# mkdir Scoap_Inputs
# mkdir Gate_Mappings

# mv scoap_input*.txt Scoap_Inputs
# mv gate_mapping*.txt Gate_Mappings