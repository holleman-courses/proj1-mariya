#!/usr/bin/env python

import argparse
import sys
import numpy as np

bytes_per_line = 12
indent = "  " # two spaces

def xxd(filename, var_name, outfile, input_type=np.uint8):
    
    try:
        if outfile is None:
            fpo = sys.stdout
        else:
            fpo = open(outfile, 'w')
    except:
        print(f"Could not open output file {outfile}")
        
        
    try:
        with open(filename, 'rb') as f:

            ####################
            # data = f.read()
            data = np.fromfile(f, dtype=input_type)
            fpo.write(f"const unsigned char {var_name}[] = {{\n{indent}") # {{ => { in f-string

            for count, byte in enumerate(data):
                fpo.write(f"0x{byte:02x}")
                if count+1 < len(data):
                    fpo.write(", ") # add a comma unless we're on the last byte
                if (count+1) % bytes_per_line == 0 or (count+1)==len(data):
                    fpo.write(f"\n{indent}") # newline + indentation every bytes_per_line bytes
            fpo.write("};\n")
            
            fpo.write(f"unsigned int {var_name}_len = {len(data)};\n")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        fpo.close()
    fpo.close()

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description='Generate C header equivalent of xxd -i')
    parser.add_argument('filename', type=str, help='Input file name')
    parser.add_argument('--var_name', type=str, default='data', help='Variable name for the generated C code')
    parser.add_argument('--out_file', type=str, default=None, help='name of file to be written')
    args = parser.parse_args()

    xxd(args.filename, args.var_name, args.out_file)
    '''
    xxd("cat_detector_int8.tflite", "cat_detect_model_data", "model.c")
