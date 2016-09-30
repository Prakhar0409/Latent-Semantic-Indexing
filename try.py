import os,re,string,math,argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

regex = re.compile('[%s]' % re.escape(string.punctuation))



# running command below
# python3 lsi.py -z 1000 -k 100 --dir sampleio --doc_in doc_in.txt --doc_out doc_out.txt --term_in term_in.txt --term_out term_out.txt --query_in query_in.txt --query_out query_out.txt
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
print(args)
print('###################')
z = args['z']
k = args['k']
#reading arguments done




base_dir='Documents'
doc_list = os.listdir(base_dir)

#MAKING VOCABULARY
lexicon = set()
titles = dict()
rev_titles = dict()
#print "##############"
#print doc_list
# print "##############"
print "Making lexicon"
for d in doc_list:
	with open(base_dir+'/'+d,'rt') as f:
		t = f.readline()
		con = t+f.read()
		con= regex.sub(' ', con)
		titles[t[:-1]] = int(d[:-4])
		rev_titles[int(d[:-4])] = t[:-1]
		lexicon.update([word for word in con.lower().split()])

lexicon = list(lexicon)
lex_dict = {x:i for i,x in enumerate(lexicon)}
#print titles
#print lexicon



def tf(doc):
	with open(base_dir+'/'+doc,'rt') as f:
		t = f.readline()
		con = t+f.read()
		con = con.lower()
		con = regex.sub(' ', con)
		tf_vector = [0]*len(lexicon)
		for w in con.split():
			tf_vector[lex_dict[w]] +=1
		return tf_vector


print "Calculating term doc matrix"
tdm = []
n = len(doc_list)
for i in range(1,n+1):		#iterate over all documents
	d = str(i)+'.txt'
	tf_vector = tf(d)
	#print "doc no: %s " % d , tf_vector
	tdm.append(tf_vector)


#print np.matrix(tdm)

def l2_normalizer(tdm):
	tmp = np.matrix(tdm)
	denom = np.dot(tmp,tmp.T)
	print denom
	return [(el / math.sqrt(denom)) for el in vec]

print "Calculating l2 normalised td matrix"
tdm_l2 = tdm
for vec in tdm:
	tdm_l2.append(l2_normalizer(vec))

# print np.matrix(tdm_l2)

def numDocsContaining(idx,word):
	doccount = 0
	n= len(doc_list)
	for i in range(0,n):
		if tdm_l2[i][idx] > 0 :
			doccount +=1
	return doccount 


def idf(idx,word):
	n_samples = len(doc_list)
	df = numDocsContaining(idx,word)
	return np.log(n_samples / 1+df)
	# return df


print "Calculating IDF matrix"
idf_vector = [idf(idx, word) for idx,word in enumerate(lexicon)]
#print np.matrix(idf_vector)
idf_mat = np.zeros((len(idf_vector),len(idf_vector)))
np.fill_diagonal(idf_mat,idf_vector)
print "IDF MAT CALCULATED"
# print idf_mat

print "Calculating tdm x idf"
tdm_tfidf = np.dot(tdm_l2,idf_mat)
print "TDM TFIDF"
#print tdm_tfidf

print "Calculating l2 normalised idf mat"
tdm_tfidf_l2 = []
for tf_vector in tdm_tfidf:
	tdm_tfidf_l2.append(l2_normalizer(tf_vector))

#print "tdm_tfidf_l2"
#print np.matrix(tdm_tfidf_l2)
print len(lexicon)


tdm = np.matrix(tdm_tfidf_l2)

a = sp.csc_matrix(tdm.T)						# converting to a sparse matrix
a = a.asfptype()								# converting matrix to matrix of floats
[u, s, vt] = svds(a, k = 8, which = 'LM')		# u -> terms x k | s -> k x k | vt -> k x documents
v = vt.T
print v


## DOCUMENT SIMILARITY
### READING INPUT FILE
file_names = open(args['dir']+'/'+args['doc_in'],'r').read().splitlines()
print "######### Document similarity ###########"
print(file_names)



### COMPUTING SIMILARITY and writing results
fout = open(args['dir']+'/'+args['doc_out'],'w')
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
		similarity.sort(key=lambda x: -x[0])
	#dec_simi = sorted(similarity,key=lambda x:(float(x[1]),float(x[0])))	
	# dec_simi = sorted(similarity)
	print "wwwwwwwwww Documents similar to Doc%d wwwwwwwwwwww" % (idx)
	print similarity[:k]
	first = True
	for i,outp in similarity[:k]:
		if first == True:
			first = False
			fout.write(rev_titles[outp])
			continue
		fout.write(';\t'+rev_titles[outp])
	fout.write('\n')

fout.close()


## TERM SIMILARITY
### READING INPUT FILE
file_names = open(args['dir']+'/'+args['term_in'],'r').read().splitlines()
print "######### Document similarity ###########"
print(file_names)



### COMPUTING SIMILARITY and writing results
fout = open(args['dir']+'/'+args['term_out'],'w')
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
		#print (i+1,val)
		similarity.append((val,i+1))
		similarity.sort(key=lambda x: -x[0])
	#dec_simi = sorted(similarity,key=lambda x:(float(x[1]),float(x[0])))	
	# dec_simi = sorted(similarity)
	print "wwwwwwwwww Documents similar to Doc%d wwwwwwwwwwww" % (idx)
	print similarity[:k]
	first = True
	for i,outp in similarity[:k]:
		if first == True:
			first = False
			fout.write(rev_titles[outp])
			continue
		fout.write(';\t'+rev_titles[outp])
	fout.write('\n')

fout.close()