#!/bin/bash
# Automates running the parse-scoap-output.py script for all contest designs
# Usage: ./get_csv.sh
# (Run this in the Trust-Hub working directory)

for i in {0..19}
do
    echo "Running design $i"
    python3 CreateCSV.py Scoap_Outputs/SCOAP_Output$i.txt Gate_Mappings/gate_mapping$i.txt ../test-cases/trojan/result$i.txt $i
done

for i in {20..29}
do
    echo "Running design $i"
    python3 CreateCSV.py Scoap_Outputs/SCOAP_Output$i.txt Gate_Mappings/gate_mapping$i.txt ../test-cases/trojan_free/result$i.txt $i
done

for scoap_file in Scoap_Outputs/SCOAP_Output*.txt; do
    # Extract suffix by removing prefix and suffix
    suffix="${scoap_file#Scoap_Outputs/SCOAP_Output}"
    suffix="${suffix%.txt}"

    map_file="gate_mapping${suffix}.txt"
    label_file="result${suffix}.txt"

    if [[ -e "Gate_Mappings/$map_file" ]]; then
        echo "Processing: $scoap_file and $map_file"
        python CreateCSV.py "$scoap_file" "Gate_Mappings/$map_file" "Trojan_Labels/$label_file" "$suffix"
    else
        echo "Missing mapping file for suffix: $suffix"
    fi
done

# mkdir Scoap_Scores
# mv gate_scores*.csv Scoap_Scores