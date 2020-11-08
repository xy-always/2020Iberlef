# coding=utf-8
import os
import sys
from sklearn.externals import joblib
from get_hierarchy_norm import code_parse_4

def read_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

def save_code(files, outfile1, outfile2):
    code_id = {}
    id_code = {}
    code_id['O'] = 0
    id_code[0] = 'O'
    co = 1
    for file in files:
        path = file
        lines = read_lines(path)
        for line in lines:
            line = line.strip().split('\t')
            code = line[0]
            
            if str(code) not in code_id:
                code_id[code] = co
                id_code[co] = code
                co += 1
    joblib.dump(code_id, outfile1)
    joblib.dump(id_code, outfile2)
# 保存4个分层code字典
def save_hie_code(files, outfile1, outfile2, outfile3, outfile4, outfile5, outfile6, outfile7, outfile8):
    code_id_tumor = {}
    id_code_tumor = {}
    code_id_behavior = {}
    id_code_behavior = {}
    code_id_differentiation = {}
    id_code_differentiation = {}
    code_id_high = {}
    id_code_high = {}
    code_id = [code_id_tumor,code_id_behavior,code_id_differentiation,code_id_high]
    id_code = [id_code_tumor,id_code_behavior,id_code_differentiation,id_code_high]
    for i in range(4):
        code_id[i]['O'] = 0
        id_code[i][0] = 'O'
    co = 1
    for file in files:
        path = file
        lines = read_lines(path)
        for line in lines:
            line = line.strip().split('\t')
            code = line[0].strip()
            t,b,d,h = code_parse(code)
            if t not in code_id_tumor:
                code_id_tumor[t] = len(code_id_tumor)
                id_code_tumor[len(code_id_tumor)-1]=t
            if b not in code_id_behavior:
                code_id_behavior[b] = len(code_id_behavior)
                id_code_behavior[len(code_id_behavior)-1]=b
            if d not in code_id_differentiation:
                code_id_differentiation[d] = len(code_id_differentiation)
                id_code_differentiation[len(code_id_differentiation)-1]=d
            if h not in code_id_high:
                code_id_high[h] = len(code_id_high)
                id_code_high[len(code_id_high)-1]=h
    joblib.dump(code_id[0], outfile1)
    joblib.dump(id_code[0], outfile2)
    joblib.dump(code_id[1], outfile3)
    joblib.dump(id_code[1], outfile4)
    joblib.dump(code_id[2], outfile5)
    joblib.dump(id_code[2], outfile6)
    joblib.dump(code_id[3], outfile7)
    joblib.dump(id_code[3], outfile8)

# 保存3个分层code字典
def save_3_hie_code(files, outfile1, outfile2, outfile3, outfile4, outfile5, outfile6):
    code_id_tumor = {}
    id_code_tumor = {}
    code_id_behavior_differentiation = {}
    id_code_behavior_differentiation = {}
    code_id_high = {}
    id_code_high = {}
    code_id = [code_id_tumor,code_id_behavior_differentiation,code_id_high]
    id_code = [id_code_tumor,id_code_behavior_differentiation,id_code_high]
    for i in range(3):
        code_id[i]['O'] = 0
        id_code[i][0] = 'O'
    for file in files:
        path = file
        lines = read_lines(path)
        for line in lines:
            line = line.strip().split('\t')
            code = line[0].strip()
            t,b,d,h = code_parse_4(code)
            if t not in code_id_tumor:
                code_id_tumor[t] = len(code_id_tumor)
                id_code_tumor[len(code_id_tumor)-1]=t
            if b != 'O':
                if d == 'O':
                    new_2 = b
                else:
                    new_2 = b+d
            else:
                    new_2 = 'O'
            if new_2 not in code_id_behavior_differentiation:
                # print(new_2)
                code_id_behavior_differentiation[new_2] = len(code_id_behavior_differentiation)
                id_code_behavior_differentiation[len(code_id_behavior_differentiation)-1]=new_2
            if h not in code_id_high:
                code_id_high[h] = len(code_id_high)
                id_code_high[len(code_id_high)-1]=h
    # print(code_id[0])
    # print(id_code[0])
    print(code_id[1])
    print(id_code[1])
    print(code_id[2])
    print(id_code[2])
    joblib.dump(code_id[0], outfile1)
    joblib.dump(id_code[0], outfile2)
    joblib.dump(code_id[1], outfile3)
    joblib.dump(id_code[1], outfile4)
    joblib.dump(code_id[2], outfile5)
    joblib.dump(id_code[2], outfile6)

files = ['valid-codes.txt']
# outfile1 = 'data_v3_p/code_id_tumor.pkl'
# outfile2 = 'data_v3_p/id_code_tumor.pkl'
# outfile3 = 'data_v3_p/code_id_behavior.pkl'
# outfile4 = 'data_v3_p/id_code_behavior.pkl'
# outfile5 = 'data_v3_p/code_id_differentiation.pkl'
# outfile6 = 'data_v3_p/id_code_differentiation.pkl'
# outfile7 = 'data_v3_p/code_id_high.pkl'
# outfile8 = 'data_v3_p/id_code_high.pkl'
## 3个层次
outfile1 = 'data_v3_p/code_id_t_new.pkl'
outfile2 = 'data_v3_p/id_code_t_new.pkl'
outfile3 = 'data_v3_p/code_id_bd.pkl'
outfile4 = 'data_v3_p/id_code_bd.pkl'
outfile5 = 'data_v3_p/code_id_h_new.pkl'
outfile6 = 'data_v3_p/id_code_h_new.pkl'


# save_hie_code(files, outfile1, outfile2, outfile3, outfile4, outfile5, outfile6, outfile7, outfile8)
save_3_hie_code(files, outfile1, outfile2, outfile3, outfile4, outfile5, outfile6)
# code_id_tumor = joblib.load('data_v3_p/code_id_tumor.pkl')
# print(len(code_id_tumor))
# id_code_tumor = joblib.load('data_v3_p/id_code_tumor.pkl')
# print(len(id_code_tumor))
# code_id_behavior = joblib.load('data_v3_p/code_id_behavior.pkl')
# print(len(code_id_behavior))
# code_id_differentiation = joblib.load('data_v3_p/code_id_differentiation.pkl')
# print(len(code_id_differentiation))
# code_id_high = joblib.load('data_v3_p/code_id_high.pkl')
# print(len(code_id_high))
# print(code_id_tumor['8015'])
# print(id_code_tumor[12])
# code_id_bd = joblib.load('data_v3_p/code_id_bd.pkl')
# print(code_id_bd['38'])

