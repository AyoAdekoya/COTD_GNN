# Functions to find trojan labeled gates in RS232 readme and sXXX source code

import re
import sys
import glob
import os

def extract_tj_readme(readme_file):
    
    """
    Parses a readme file and extracts a set of Trojan gate instance names.

    Args:
        readme_file (str): Path to the readme.txt file.

    Returns:
        set: A set containing instance names (e.g., {'U293', 'U294', ...})
    """
    trojan_gates = set()

    with open(readme_file, "r") as f:
        content = f.read()

    # Find the "Trojan Circuit" section
    trojan_circuit_match = re.search(r"Trojan Circuit\s+(.*)", content, re.DOTALL | re.IGNORECASE)
    if not trojan_circuit_match:
        return trojan_gates  # Return empty set if no Trojan Circuit section

    circuit_text = trojan_circuit_match.group(1)

    # Match gate lines that contain instance names
    # Example line: NAND4X1 U293(.A(...),...)
    gate_lines = re.findall(r"^\s*\w+\s+([\w$]+)\s*\(", circuit_text, re.MULTILINE)

    trojan_gates.update(gate_lines)

    return trojan_gates


import re

def extract_tj_netlist(trojan_netlist_file, trojan_free_netlist_file):
    """
    Compares a trojan-inserted netlist and a trojan-free netlist, and returns
    the set of gates (instance names) that are unique to the trojan netlist.

    Args:
        trojan_netlist_file (str): Path to trojan netlist file.
        trojan_free_netlist_file (str): Path to trojan-free netlist file.

    Returns:
        set: A set of instance names unique to the trojan netlist.
    """
    def extract_instance_names(file_path):
        with open(file_path, "r") as f:
            text = f.read()
        # Match lines like: GATETYPE INSTANCE_NAME(...)
        instance_names = set(re.findall(r"^\s*\w+\s+([\w$]+)\s*\(", text, re.MULTILINE))
        return instance_names

    # Extract sets of instance names from each netlist
    free_gates = extract_instance_names(trojan_free_netlist_file)
    trojan_gates = extract_instance_names(trojan_netlist_file)

    # Find gates that only exist in the trojan netlist
    unique_trojan_gates = trojan_gates - free_gates

    return unique_trojan_gates

def extract_tj_contest(trojan_gates_file):
    trojan_gates = set()
    with open(trojan_gates_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if lines[0] == "TROJANED":
        try:
            start = lines.index("TROJAN_GATES") + 1
            end = lines.index("END_TROJAN_GATES")
            trojan_gates = set(lines[start:end])
        except ValueError:
            print("Error: TROJANED file is malformed.")
            sys.exit(1)
    return trojan_gates


def get_verilog_filepaths(design_folder):
    # Use glob to find all files in a folder
    file_list = glob.glob(f"../../TH-Benchmarks/{design_folder}/src/TjIn/*")
    if not file_list:
        print("No files found!")
        sys.exit()
    else:
        # Open the first file found
        tj_filepath = file_list[0]

    # Use glob to find all files in a folder
    file_list2 = glob.glob(f"../../TH-Benchmarks/{design_folder}/src/TjFree/*")
    if not file_list2:
        print("No files found!")
        sys.exit()
    else:
        # Open the first file found
        tj_free_filepath = file_list2[0]
    
    return tj_filepath, tj_free_filepath

if __name__ == "__main__":
    # Step 1: Get all Trojan-free designs
    trojan_free_files = glob.glob("../../TH-Benchmarks/TRIT-TC/original_designs/*.v")

    # Step 2: Loop through each Trojan-free design
    for tf_path in trojan_free_files:
        # Extract the base name without extension (e.g., c2670)
        design_name = os.path.splitext(os.path.basename(tf_path))[0]
        
        # Step 3: Find all Trojan folders for this design
        trojan_folders = glob.glob(f"../../TH-Benchmarks/TRIT-TC/{design_name}_T*/")
        
        for trojan_folder in trojan_folders:
            # Find Verilog file(s) in this Trojan folder
            trojan_files = glob.glob(os.path.join(trojan_folder, "*.v"))
            
            # You might have one or multiple .v files; loop over them
            for trojan_file in trojan_files:
                print(f"Processing Trojan file: {trojan_file} with Trojan-free file: {tf_path}")

                trojan_design_name = os.path.basename(os.path.normpath(trojan_folder))

                # === Call your function here ===
                trojan_gates = extract_tj_netlist(trojan_file, tf_path)
                with open(f"result{trojan_design_name}.txt", "w") as f:
                    f.write("TROJANED\n")
                    f.write("TROJAN_GATES\n")
                    for gate in trojan_gates:
                        f.write(f"{gate}\n")

                    f.write("END_TROJAN_GATES\n")





# Test readme funciton
# if __name__ == "__main__":
#     readme = sys.argv[1]
#     gates = extract_tj_readme(readme)
#     print(gates)

# Test file comparer function
# if __name__ == "__main__":

#     design_folder = sys.argv[1]

#     # Get trojan gates
#     if design_folder.startswith("RS"):    trojaned_gates = extract_tj_readme(f"../../TH-Benchmarks/{design_folder}/Read me.txt")
#     elif design_folder.startswith("s") or design_folder.startswith("wb"):
#         tj_filepath, tj_free_filepath = get_verilog_filepaths(design_folder)
#         trojaned_gates = extract_tj_netlist(tj_filepath, tj_free_filepath)
#     else:
#         print("Design type not recognized")

#     with open(f"result{design_folder}.txt", "w") as f:
#         f.write("TROJANED\n")
#         f.write("TROJAN_GATES\n")
#         for gate in trojaned_gates:
#             f.write(f"{gate}\n")

#         f.write("END_TROJAN_GATES\n")


