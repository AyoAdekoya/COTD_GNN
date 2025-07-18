# globy glob

import glob
import os
import re
from CSVtoPyg import create_graph, nx_to_pyG
from zScore import compute_scoap_stats

def create_all_data():
    verilog_files = []
    design_names = []
    data_objects = []
    verilog_files_free = []
    design_names_free = []
    data_objects_free = []

    # base_folder = "TH-Benchmarks"
    base_folder = "TH_Trojans"


    verilog_files = []
    design_names = []

    # Tj contest design files
    test_cases_folder = "Trojan_GNN/test-cases"

    for subfolder in ["trojan"]:
        sub_path = os.path.join(test_cases_folder, subfolder)
        if not os.path.isdir(sub_path):
            continue

        v_files = glob.glob(os.path.join(sub_path, "*.v"))
        for v_file in v_files:
            filename = os.path.basename(v_file)
            # Extract number from designXXX.v
            match = re.match(r"design(\d+)\.v", filename)
            if match:
                design_num = match.group(1)
                verilog_files.append(v_file)
                design_names.append(design_num)

    # Tj free contest designs
    for subfolder in ["trojan_free"]:
        sub_path = os.path.join(test_cases_folder, subfolder)
        if not os.path.isdir(sub_path):
            continue

        v_files = glob.glob(os.path.join(sub_path, "*.v"))
        for v_file in v_files:
            filename = os.path.basename(v_file)
            # Extract number from designXXX.v
            match = re.match(r"design(\d+)\.v", filename)
            if match:
                design_num = match.group(1)
                verilog_files_free.append(v_file)
                design_names_free.append(design_num)

    # Tj trusthub files
    for design_folder in os.listdir(base_folder):
        design_path = os.path.join(base_folder, design_folder)
        if not os.path.isdir(design_path):
            continue

        # Handle potential "double folder" case
        subfolder_path = os.path.join(design_path, design_folder)
        if os.path.isdir(subfolder_path):
            # Use the inner folder instead
            design_path = subfolder_path

        # Save design name
        design_name = design_folder


        # Figure out which pattern to use for this design
        if design_name.startswith("RS232"):
            pattern = os.path.join(design_path, "src", "90nm", "*.v")
            pattern2 = None
        elif design_name.startswith("s") or design_name.startswith("wb_conmax"):
            pattern = os.path.join(design_path, "src", "TjIn", "*.v")
            pattern2 = os.path.join(design_path, "src", "TjFree", "*.v")
        elif design_name.startswith("TRIT-TC") or design_name.startswith("TRIT-TS"):
            # These folders contain design subfolders directly
            subdesign_folders = os.listdir(design_path)
            for sub in subdesign_folders:
                sub_path = os.path.join(design_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                if sub == 'original_designs':
                    v_files_free = glob.glob(os.path.join(sub_path, "*.v"))
                    if v_files_free:
                        verilog_files_free.extend(v_files_free)
                        for v_file in v_files_free:
                            match = re.search(r"([a-zA-Z]\d+)\.v$", v_file)
                            if match:
                                design_name_free = match.group(1) + "free"

                            design_names_free.append(design_name_free) #type: ignore
                else:
                    v_files = glob.glob(os.path.join(sub_path, "*.v"))
                    if v_files:
                        verilog_files.extend(v_files)
                        design_names.extend([sub] * len(v_files))
            continue  # Skip to next top-level folder

        else:
            continue  # Skip unknown designs

        # Find .v files matching pattern
        v_files = glob.glob(pattern)
        v_files_free = []
        if pattern2:
            v_files_free = glob.glob(pattern2)
        if v_files:
            verilog_files.extend(v_files)
            design_names.extend([design_name] * len(v_files))
        if v_files_free:
            verilog_files_free.extend(v_files_free)
            design_name_free = design_name.split('-')[0] + "-free"
            design_names_free.extend([design_name_free] * len(v_files_free))

    for file in verilog_files:
        print("Tj:", file)
    for file in verilog_files_free:
        print("Tj free: ", file)
    for name in design_names:
        print("Design name: ", name)

    for name in design_names_free:
        print("Design name free: ", name)


    def create_data(netlist_path, scoap_csv_path, gate_mapping_path):
        # Your custom processing logic here
        print("Processing files:")
        print(f"  Netlist: {netlist_path}")
        print(f"  SCOAP CSV: {scoap_csv_path}")
        print(f"  Gate mapping: {gate_mapping_path}")
        G = create_graph(netlist_path, scoap_csv_path, gate_mapping_path)
        data = nx_to_pyG(G, design_name)
        print("-" * 60)
        return data

    missing_files = []

    for netlist_file, design_name in zip(verilog_files, design_names):
        gate_map_file = f"Trojan_GNN/Trust-Hub/Gate_Mappings/gate_mapping{design_name}.txt"
        scoap_file    = f"Trojan_GNN/Trust-Hub/Scoap_Scores/gate_scores{design_name}.csv"

        if not (os.path.exists(gate_map_file) and os.path.exists(scoap_file)):
            missing_files.append(design_name)
            print(f"Skipping {design_name} — missing mapping or SCOAP file")
            continue

        data_objects.append(create_data(netlist_file, scoap_file, gate_map_file))

    for netlist_file, design_name in zip(verilog_files_free, design_names_free):
        gate_map_file = f"Trojan_GNN/Trust-Hub/Gate_Mappings/gate_mapping{design_name}.txt"
        scoap_file    = f"Trojan_GNN/Trust-Hub/Scoap_Scores/gate_scores{design_name}.csv"

        if not (os.path.exists(gate_map_file) and os.path.exists(scoap_file)):
            missing_files.append(design_name)
            print(f"Skipping {design_name} — missing mapping or SCOAP file")
            continue

        data_objects_free.append(create_data(netlist_file, scoap_file, gate_map_file))

    for file in missing_files:
        print("Missing:", file)

    return data_objects, data_objects_free


if __name__ == "__main__":
    compute_scoap_stats()
    create_all_data()