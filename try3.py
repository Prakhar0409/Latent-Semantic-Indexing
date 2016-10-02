import os,re,string,math,argparse,datetime
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

pattern = re.compile(r'\W+')


t1 = datetime.datetime.now()
# running command below
# python try3.py -z 4 -k 4 --dir test --doc_in doc_in.txt --doc_out doc_out.txt --query_in query_in.txt --query_out query_out.txt --term_in term_in.txt --term_out term_out.txt
# python try3.py -z 100 -k 9 --dir Documents --doc_in doc_in.txt --doc_out doc_out.txt --query_in query_in.txt --query_out query_out.txt --term_in term_in.txt --term_out term_out.txt
num_terms = 0
num_docs = 5000

def readCommandLine():
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
	return vars(args)

args = readCommandLine()
z = args['z']
k = args['k']

# print(args)
print('############################')
############################################################reading arguments done



base_dir= args['dir']
# doc_list = os.listdir(base_dir)

#MAKING VOCABULARY
# lexicon = set()
lexicon = dict()
titles = dict()
rev_titles = dict()

print "Making lexicon"
for idx in range(1,num_docs+1):
	d = str(idx)+".txt"
	with open(base_dir+'/'+d,'rt') as f:
		t = f.readline()
		con = t+f.read()
		# print con
		# con= con.lower()
		con = pattern.split(con)
		titles[t[:-1]] = int(d[:-4])
		rev_titles[int(d[:-4])] = t[:-1]
		
		for word in con:
			w = word.lower()
			if w == '':
				continue
			elif w in lexicon:
				a = 1
			else:
				lexicon[w] = num_terms
				num_terms +=1
		# lexicon.update([word.lower() for word in con])


# lexicon = list(lexicon)
# if '' in lexicon:
	# print lexicon.remove('')
# print lexicon
lex_dict = {v:k for k,v in lexicon.items()}
print "lexicon dict formed"
vocab_size = len(lexicon)

print "Number of distinct words: %d" % vocab_size
t2 = datetime.datetime.now()
t1diff = t2-t1
print t1diff;



row = []
col = []
freq = []
def tf(idx):
	doc = str(idx)+'.txt'
	with open(base_dir+'/'+doc,'rt') as f:
		t = f.readline()
		con = t+f.read()
		con = con.lower()
		con = pattern.split(con)
		# print con
		tf_dict = dict()
		for w in con:
			if w == '':
				continue
			elif w in tf_dict:
				tf_dict[w] += 1.0
			else:
				tf_dict[w] = 1.0
		
		for (k,v) in tf_dict.items():
			row.append(lexicon[k])
			col.append(idx-1)
			freq.append(v)

print "Calculating term doc matrix"


for i in range(1,num_docs+1):		#iterate over all documents
	tf(i)

print "read in row and col form"
t3 = datetime.datetime.now()
t2diff = t3-t2
print t2diff;


print "Converting to sparse representation"
stdm = sp.csc_matrix( (freq, (row , col)), shape=(vocab_size, num_docs))
t4 = datetime.datetime.now()
t3diff = t4-t3
print t3diff;
print "Converted to sparse representation"

# print stdm.todense()

print "Calculating svds"
# u, s, vt = svds(stdm, z, which = 'LM')			# u - nxk; v - kxm
u, s, vt = svds(stdm, k = z, which = 'LM')			# u - nxk; v - kxm
v = vt.T

# l = z - 50			
# s = s[l:]
# u = u[:,l:]
# v = v[:,l:]
# print type(v)
s1 = np.diag(s)
us = np.dot(u,s1)
vs = np.dot(v,s1)

# print u
# print vt
# print "v::::"
# print v

print s
print len(s)

# print s1
# print "vss::::"
# print vs
# print us



t5 = datetime.datetime.now()
t4diff = t5-t4
print t4diff;
print "Calculated svds"



def simiCalc(t,word_dict,mat):
	idx = word_dict[t]
	print "----------------------------"
	print idx
	print "----------------------------"
	similarity = []
	# print v.shape
	d1 = mat[idx-1,:]
	for i,r in enumerate(mat):
		val = np.dot(d1,r)
		if (d1 == r).all():
			print (i+1, val)
		if (idx - i == 1):
			print "voila: "	
			print (i+1, val)
		similarity.append((val,i+1))
	print "wwwwwwwwww Similar objects to obj%d wwwwwwwwwwww" % (idx)
	similarity.sort(key=lambda x: -x[0])
	print similarity[:k]
	return similarity

sample_dir = "sampleio/"
# sample_dir = "tp/"
# sample_dir = ""
### DOCUMENT SIMILARITY ###

## READING INPUT FILE ##
file_names = open(sample_dir+args['doc_in'],'r').read().splitlines()
print "######### Document similarity ###########"
print(file_names)


## COMPUTING SIMILARITY and writing results


fout = open(sample_dir+args['doc_out'],'w')
# print similarity

for t in file_names:
	similarity = simiCalc(t,titles,vs)
	first = True
	for i,outp in similarity[:k]:
		if first == True:
			first = False
			fout.write(rev_titles[outp])
			continue
		fout.write(';\t'+rev_titles[outp])
	fout.write('\n')
fout.close()

print "Document similarity done"
t6 = datetime.datetime.now()
t5diff = t6-t5
print t5diff;

print "-------------------------------------------------------"
print "-------------------------------------------------------"
print "-------------------------------------------------------"
print "-------------------------------------------------------"
print "-------------------------------------------------------"

## TERM SIMILARITY
### READING INPUT FILE
term_names = open(sample_dir+args['term_in'],'r').read().splitlines()
print "######### Document similarity ###########"
print(term_names)



### COMPUTING SIMILARITY and writing results
fout = open(sample_dir+args['term_out'],'w')


for t in term_names:
	similarity = simiCalc(t,lexicon,us)
	first = True
	for i,outp in similarity[:k]:
		if first == True:
			first = False
			fout.write(lex_dict[outp])
			continue
		fout.write(';\t'+lex_dict[outp])
	fout.write('\n')
fout.close()

print "Terms similarity done"
t7 = datetime.datetime.now()
t6diff = t7-t6
print t6diff;




## QUERY SIMILARITY
### READING INPUT FILE
# X1 = np.dot(us,vt)
queries = open(sample_dir+args['query_in'],'r').read().splitlines()
print "######### Query similarity ###########"
print(queries)

fout = open(sample_dir+args['query_out'],'w')

similarity = [(0,0)]* num_docs 
for query in queries:
	terms = query.split()
	for t in terms:
		idx = lexicon[t.lower()] - 1
		r = u[idx,:]
		tmp = []
		for i,c in enumerate(vs):
			x,y = similarity[i] 
			x += np.dot(c,r)
			y = i+1
			similarity[i]=(x,y)
		
		# similarity += tmp

		# similarity += [(np.dot(c,r),i+1) for i,c in enumerate(vs)]

	# print similarity
	similarity.sort(key=lambda x: -x[0])	
	print similarity[:k]
	first = True
	for val,idx in similarity[:k]:
		if first == True:
			first = False
			fout.write(rev_titles[idx])
			continue
		fout.write(';\t'+rev_titles[idx])
	fout.write('\n')
fout.close()

