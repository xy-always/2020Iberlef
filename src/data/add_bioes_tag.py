import os
import sys

def readlines(file):
    with open(file, 'r') as f:
        return f.readlines()

def add_bioes(infile, outfile):
    in_lines = readlines(infile)
    with open(outfile, 'w') as f:
        for i, line in enumerate(in_lines):
            line = line.strip().split('\t')
            start_ids = line[2]
            end_ids = line[3]
            bioes_tag = []
            start = []
            end = []
            for i, sl in enumerate(start_ids.strip().split()):
                bioes_tag.append('O')
                if sl == '1':
                    start.append(i)
            for i, el in enumerate(end_ids.strip().split()):
                if el == '1':
                    end.append(i)
                    print('yes:', i)
            for i in range(len(start)):
                print(start[i], end[i])
                k = 0
                for idx in range(start[i], end[i]+1):
                    if end[i]==start[i]:
                        bioes_tag[idx] = 'S'
                    else:
                        if k == 0:
                            bioes_tag[idx] = 'B'
                        elif k == end[i]-start[i]:
                            bioes_tag[idx] = 'E'
                        else:
                            bioes_tag[idx] = 'I'
                    k+=1

            for content in line:
                f.write( content + '\t')
            f.write(" ".join([str(bioes_tag[k]) for k in range(len(bioes_tag))]) + '\n')

add_bioes('/disk2/xy_disk2/2020_IberLEF/MODEL_NER/BERT_RC/data/data_v3_p/all_merged.out', '/disk2/xy_disk2/2020_IberLEF/MODEL_NER/BERT_RC/data/data_v3_p/all_merged_bioes.out')