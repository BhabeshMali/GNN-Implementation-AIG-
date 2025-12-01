import subprocess
import re
import os

def run_abc_flow(verilog_file, library_file, output_aig):
    abc_folder = "/workspace/ckarfa/Bhabesh/GNNImplementatio/abc/"
    abc_binary = "./abc" 
    
    # Check if the library file exists before running
    if not os.path.exists(library_file):
        raise FileNotFoundError(f"Cannot find library file: {library_file}")

    # Prepare the ABC commands
    # We use -c to run in "Batch Mode" (Run & Exit)
    abc_commands = f"""
    read_lib {library_file};
    read_verilog {verilog_file};
    strash;
    write_aiger {output_aig};
    map;
    print_stats;
    """
    
    # Run ABC inside the folder
    process = subprocess.Popen(
        [abc_binary, '-c', abc_commands], 
        cwd=abc_folder,   
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    
    # Debugging: If ABC fails, print the error
    if process.returncode != 0:
        print("Error running ABC:")
        print(stderr)
        return 0, 0

    # Parse Output
    area = 0.0
    delay = 0.0
    area_match = re.search(r'area\s*=\s*(\d+\.?\d*)', stdout)
    delay_match = re.search(r'delay\s*=\s*(\d+\.?\d*)', stdout)
    
    if area_match: area = float(area_match.group(1))
    if delay_match: delay = float(delay_match.group(1))
    
    return area, delay

# Example Usage
if __name__ == "__main__":
    v_file = "/workspace/ckarfa/Bhabesh/GNNImplementatio/abc/verilog_files/adder.v"
    aig_file = "/workspace/ckarfa/Bhabesh/GNNImplementatio/abc/adder.aig"
    lib_file = "/workspace/ckarfa/Bhabesh/GNNImplementatio/abc/asap7_clean.lib"
    area, delay = run_abc_flow(v_file, lib_file, aig_file)
    print(f"Processed {v_file}: Area={area}, Delay={delay}")
