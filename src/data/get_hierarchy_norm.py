import os
import sys

def read_lines(filename):
	with open(filename, 'r') as f:
		return f.readlines()

def code_parse_4(code):
	if code == 'O':
		return 'O', 'O', 'O', 'O'
	else:
		code = code.split('/')
		tumor = code[0]
		behavior = code[1][0]
		if len(code[1]) == 2:
			differentiation = code[1][1]
		else:
			differentiation = 'O'
		if len(code) == 3:
			high = code[-1]
		else:
			high = 'O'
	return tumor, behavior, differentiation, high

def code_parse_3(code):
	if code == 'O':
		return 'O', 'O', 'O'
	else:
		code = code.split('/')
		tumor = ''
		bd = ''
		high = ''
		if len(code) == 1:
			print('@@'*20)
			print(code)
			tumor = code[0]
			bd = 'O'
			h = 'O'
		elif len(code) == 2:
			tumor = code[0]
			bd = code[1]
			high = 'O'
		elif len(code) == 3:
			tumor = code[0]
			bd = code[1]
			high = code[2]	
		
	return tumor, bd, high

# 被分层类型得code替代原来的code
def replace_hie_norm(raw_file, out_file):
	in_lines = read_lines(raw_file)
	with open(out_file, 'w') as f:
		for line in in_lines:
			line = line.strip().split('\t')
			norm_sequence = line[6].strip()
			code_sequence = line[7].strip()
			new_norm_sequence = []
			new_code_sequence = []
			for norm_code in norm_sequence.split():
				t, b, d, h = code_parse_4(norm_code)
				n = t+'/'+b+'/'+d+'/'+h
				new_norm_sequence.append(n)
			for code in code_sequence.split():
				t, b, d, h = code_parse_4(code)
				n = t+'/'+b+'/'+d+'/'+h
				new_code_sequence.append(n)
			f.write(line[0]+'\t'+line[1]+'\t'+line[2]+'\t'+line[3]+'\t'+line[4]+'\t'+line[5]+'\t'+' '.join(new_norm_sequence)+'\t'+' '.join(new_code_sequence)+'\n')


if __name__ == '__main__':
	# code = '7878/2'
	# tumor, behavior, differentiation, high = code_parse(code)
	# print(tumor, behavior, differentiation, high)
	replace_hie_norm('/disk2/xy_disk2/2020_IberLEF/MODEL_NER/BERT_RC/data/data_v5/dev_raw.out', '/disk2/xy_disk2/2020_IberLEF/MODEL_NER/BERT_RC/data/data_v5/dev_raw_hie.out')