import os
import sys

def read_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

def write_code(in_file, out_file):
    lines = read_lines(in_file)
    with open(out_file, 'w') as f:
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            f.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4] + '\t' + line[5] + '\t' + 'O' + '\t' + 'O' + '\n')

write_code('/disk2/xy_disk2/2020_IberLEF/MODEL_NER/BERT_RC/data/data_v3_p/test.out', '/disk2/xy_disk2/2020_IberLEF/MODEL_NER/BERT_RC/data/data_v4/test.out')