import os
import sys
# 得到做辅助任务的标签，即把是实体标为1，不是标为0
def read_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

def get_token_tag(infile, outfile):
    in_lines = read_lines(infile)
    with open(outfile, 'w') as f:
        for i, line in enumerate(in_lines):
            line = line.strip().split('\t')
            start_ids = line[2]
            end_ids = line[3]
            token_tag = []
            start = []
            end = []
            for i, sl in enumerate(start_ids.strip().split()):
                token_tag.append(0)
                if sl == '1':
                    start.append(i)
            for i, el in enumerate(end_ids.strip().split()):
                if el == '1':
                    end.append(i)
                    print('yes:', i)
            for i in range(len(start)):
                print(start[i], end[i])
                for j in range(start[i], end[i]+1):
                    token_tag[j] = 1
            for content in line:
                f.write( content + '\t')
            f.write(" ".join([str(token_tag[k]) for k in range(len(token_tag))]) + '\n')

if __name__ == "__main__":
    in_file = 'data_v1/train.out'
    out_file = 'data_v2/train.out'
    get_token_tag(in_file, out_file)