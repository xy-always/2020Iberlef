# coding=utf-8
import os
import sys

def read_lines(file):
	with open(file, 'r') as f:
		return f.readlines()

def replace(in_file, out_file):
	# new_q = ['¿Cuáles son las entidades NORMALIZABLES mencionadas ?', '¿Cuáles son las entidades NO_NORMALIZABLES mencionadas ?', '¿Cuáles son las entidades PROTEINAS mencionadas ?', '¿Cuáles son las entidades UNCLEAR mencionadas ?']
	# new_q_1 = ['¿Qué entidades NORMALIZABLES se mencionan en el texto ?', '¿Qué entidades NO_NORMALIZABLES se mencionan en el texto ?', '¿Qué entidades PROTEINAS se mencionan en el texto ?', '¿Qué entidades UNCLEAR se mencionan en el texto ?']
	new_q_1 = ['¿Qué entidades MORFOLOGÍA NEOPLASIA se mencionan en el texto ?']
	lines = read_lines(in_file)
	with open(out_file, 'w') as f:
		for i, line in enumerate(lines):
			line = line.strip().split('\t')
			nq = i % 1
			f.write(new_q_1[nq] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4] + '\t' + \
							line[5] + '\t' + line[6] + '\t' + line[7] + 'n')


def merge(file1, file2, outfile):
	lines1 = read_lines(file1)
	lines2 = read_lines(file2)
	with open(outfile, 'w') as f:
		for line in lines1:
			line = line.strip()
			f.write(line + '\n')

		for line in lines2:
			f.write(line)
 
if __name__ == '__main__':
	replace('data_v3/train_raw.out', 'data_template_v2/train.out')
	# merge('data/data_v1/train.out', 'data_guideline_query_extend/dev.out', 'data_guideline_query_extend/merge_train.out')