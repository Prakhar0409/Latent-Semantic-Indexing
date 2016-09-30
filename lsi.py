import argparse
import textmining
import os
import re
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import collections

pattern = re.compile(r'\W+')


# running command below
# python3 lsi.py -z 1000 -k 100 --dir sampleio --doc_in doc_in.txt --doc_out doc_out.txt --term_in term_in.txt --term_out term_out.txt --query_in query_in.txt --query_out query_out.txt

parser = argparse.ArgumentParser(description='Process input and output file names.')
parser.add_argument('-z',action = 'store',
		type = int, required = True, 
		help = 'Dimension of the lower dimensional space',
		metavar = '<dimension>', dest = 'z')	# dimention of lower dimensional space
parser.add_argument('-k',action = 'store',
		type = int, required = True, 
		help = 'Number of similar terms/documents to be returned.',
		metavar = '<# sim terms/docs>', dest = 'k')	
parser.add_argument('--dir',action = 'store',
		type = str, required = True, 
		help = 'Directory containing input docs.',
		metavar = '<Dirname>', dest = 'dir')
inps = ['doc_in','doc_out','term_in','term_out','query_in','query_out']
for i in inps:
	parser.add_argument('--'+i,action = 'store',
			type = str, required = True, 
			help = 'Name of '+ i +' file.',
			metavar = '<'+ i +'>', dest = i)

args = parser.parse_args()

args = vars(args)
print(args)
print('###################')
z = args['z']
k = args['k']
#reading arguments done


### Preprocessing and forming termxdocument matrix
base_dir = 'docs'				#'Documents'
docs = os.listdir(base_dir)


# for d in docs:
# 	f = open(base_dir+'/'+d,'r')
# 	t = f.readline()
# 	contents = t+ f.read()
# 	t = t[:-1]		# remove /n




titles = dict()					# title vs document number
rev_titles = dict()
terms = dict()					# title vs document number
rev_terms = dict()
tdm = textmining.TermDocumentMatrix()
word_idx = 0
for d in docs:
	f = open(base_dir+'/'+d,'r')
	t = f.readline()
	contents = t + f.read()
	tdm.add_doc(contents)
	# words = contents.split()
	# for w in words:

	# 	terms[w] = word_idx++
	# terms[]
	t = t[:-1]
	titles[t] = int(d[:-4])
	rev_titles[int(d[:-4])] = t
	f.close()

tdm.write_csv('matrix.csv', cutoff=10)
print titles

td = []
flag = True
for r in tdm.rows():
	if flag == True:
		flag = False
		continue
	td.append(r)
#print(td)

td = np.array(td)								## representing the 2D list as numpy array

a = sp.csc_matrix(td.T)						# converting to a sparse matrix
a = a.asfptype()								# converting matrix to matrix of floats
[u, s, vt] = svds(a, k = 3, which = 'LM')		# u -> terms x k | s -> k x k | vt -> k x documents
v = vt.T
######## fininshed creating term doc matrix



# read doc_in file
file_names = open(args['dir']+'/'+args['doc_in'],'r').read().splitlines()
print "######### Document similarity ###########"
print(file_names)

doc_sim = []

for t in file_names:
	idx = titles[t]
	print "----------------------------"
	print idx
	print "----------------------------++++++"
	similarity = []
	# print v.shape
	d1 = v[idx-1,:]
	for i,r in enumerate(v):
		val = np.dot(d1,r)
		if (d1 == r).all():
			print "voila: ", r
			print (i+1, val)
		print (i+1,val)
		similarity.append((val,i+1))

	#dec_simi = sorted(similarity,key=lambda x:(float(x[1]),float(x[0])))	
	dec_simi = sorted(similarity)
	print "wwwwwwwwww Documents similar to Doc%d wwwwwwwwwwww" % (idx)
	print dec_simi[1:k]

#### printing out to doc_out file
f = open(args['dir']+'/'+args['doc_out'],'w')
first = True
for i,j in dec_simi[1:k]:
	if first == True:
		f.write(rev_titles[j])
		first = False
		continue
	f.write(';\t'+rev_titles[j])
f.close()
#### Similar documents printed out

#### Similar terms
# term_names = open(args['dir']+'/'+args['term_in'],'r').read().splitlines()
# print terms

# term_sim = []

# for t in term_names:
# 	idx = term[t]
# 	print "----------------------------"
# 	print idx
# 	print "----------------------------++++++"
# 	similarity = []
# 	d1 = v[idx-1,:]
# 	for i,r in enumerate(v):
# 		val = np.dot(d1,r)
# 		if (d1 == r).all():
# 			print "voila: ", r
# 			print (i+1, val)
# 		print (i+1,val)
# 		similarity.append((val,i+1))

# 	#dec_simi = sorted(similarity,key=lambda x:(float(x[1]),float(x[0])))	
# 	dec_simi = sorted(similarity)
# 	print "wwwwwwwwww Documents similar to Doc%d wwwwwwwwwwww" % (idx)
# 	print dec_simi[1:k]




