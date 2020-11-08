import os
import sys

def read_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()


def merge(files, merge_out_file):
    with open(merge_out_file, 'w') as f:
        for file in files:
            lines = read_lines(file)
            for line in lines:
                f.write(line.strip() + '\n')

files = ['data_v4/train_ner_norm_code.out', 'data_v4/dev_ner_norm_code.out']
merge(files, 'data_v4/all_merged.out')