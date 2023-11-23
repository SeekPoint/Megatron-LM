# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import glob
import sys
import json
import argparse
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default=".",
        help="path where all the json files are located")

    parser.add_argument("--output_file", type=str, default="merged_output.json",
        help="filename where the merged json should go")

    args = parser.parse_args()

    json_path = args.json_path
    out_file = args.output_file

    json_files = glob.glob(json_path + '/*.json')

    counter = 0

    with open(out_file, 'w') as outfile:
        for fname in json_files:
            counter += 1

            if counter % 1024 == 0:
                gd.debuginfo(prj="mt", info=f"Merging at ", counter)

            with open(fname, 'r') as infile:
                for row in infile:
                    each_row = json.loads(row)
                    outfile.write(row)


    gd.debuginfo(prj="mt", info=f"Merged file", out_file)


