import os,re,string,math,argparse,datetime
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

pattern = re.compile(r'\W+')

stop_words = {'','1','2','3','4','5','6','7','8','9','0','a','about','above','across','after','again','against','all','almost','alone','along','already','also','although','always','among','an','and','another','any','anybody','anyone','anything','anywhere','are','area','areas','around','as','ask','asked','asking','asks','at','away','b','back','backed','backing','backs','be','became','because','become','becomes','been','before','began','behind','being','beings','best','better','between','big','both','but','by','c','came','can','cannot','case','cases','certain','certainly','clear','clearly','come','could','d','did','differ','different','differently','do','does','done','down','down','downed','downing','downs','during','e','each','early','either','end','ended','ending','ends','enough','even','evenly','ever','every','everybody','everyone','everything','everywhere','f','face','faces','fact','facts','far','felt','few','find','finds','first','for','four','from','full','fully','further','furthered','furthering','furthers','g','gave','general','generally','get','gets','give','given','gives','go','going','good','goods','got','great','greater','greatest','group','grouped','grouping','groups','h','had','has','have','having','he','her','here','herself','high','high','high','higher','highest','him','himself','his','how','however','i','if','important','in','interest','interested','interesting','interests','into','is','it','its','itself','j','just','k','keep','keeps','kind','knew','know','known','knows','l','large','largely','last','later','latest','least','less','let','lets','like','likely','long','longer','longest','m','made','make','making','man','many','may','me','member','members','men','might','more','most','mostly','mr','mrs','much','must','my','myself','n','necessary','need','needed','needing','needs','never','new','new','newer','newest','next','no','nobody','non','noone','not','nothing','now','nowhere','number','numbers','o','of','off','often','old','older','oldest','on','once','one','only','open','opened','opening','opens','or','order','ordered','ordering','orders','other','others','our','out','over','p','part','parted','parting','parts','per','perhaps','place','places','point','pointed','pointing','points','possible','present','presented','presenting','presents','problem','problems','put','puts','q','quite','r','rather','really','right','right','room','rooms','s','said','same','saw','say','says','second','seconds','see','seem','seemed','seeming','seems','sees','several','shall','she','should','show','showed','showing','shows','side','sides','since','small','smaller','smallest','so','some','somebody','someone','something','somewhere','state','states','still','still','such','sure','t','take','taken','than','that','the','their','them','then','there','therefore','these','they','thing','things','think','thinks','this','those','though','thought','thoughts','three','through','thus','to','today','together','too','took','toward','turn','turned','turning','turns','two','u','under','until','up','upon','us','use','used','uses','v','very','w','want','wanted','wanting','wants','was','way','ways','we','well','wells','went','were','what','when','where','whether','which','while','who','whole','whose','why','will','with','within','without','work','worked','working','works','would','x','y','year','years','yet','you','young','younger','youngest','your','yours','z'}

t1 = datetime.datetime.now()
# running command below
# python try3.py -z 4 -k 4 --dir test --doc_in doc_in.txt --doc_out doc_out.txt --query_in query_in.txt --query_out query_out.txt --term_in term_in.txt --term_out term_out.txt
# python try3.py -z 100 -k 9 --dir Documents --doc_in doc_in.txt --doc_out doc_out.txt --query_in query_in.txt --query_out query_out.txt --term_in term_in.txt --term_out term_out.txt
num_terms = 1
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



base_dir = args['dir']
# doc_list = os.listdir(base_dir)

#MAKING VOCABULARY
# lexicon = set()
lexicon = dict()
titles = dict()
rev_titles = dict()
tdf = dict()

print "Making lexicon"
for idx in range(1,num_docs+1):
	d = str(idx)+".txt"
	with open(base_dir+'/'+d,'rt') as f:
		flag = False
		t = f.readline()
		con = t+f.read()
		# print con[:-1]
		# con= con.lower()
		con = pattern.split(con)
		titles[t[:-1]] = int(d[:-4])
		rev_titles[int(d[:-4])] = t[:-1]
		
		for word in con:
			w = word.lower()
			if w in stop_words:
				continue
			elif w in lexicon:
				if flag==False:
					flag = True
					tdf[w] += 1.0 
				continue
			else:
				tdf[w] = 1.0
				lexicon[w] = num_terms
				num_terms +=1

# print lexicon
# print titles
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
			if w in stop_words:
				continue
			elif w in tf_dict:
				tf_dict[w] += 1.0
			else:
				tf_dict[w] = 1.0
		
		denom = 0.0
		for (k,v) in tf_dict.items():
			denom += v**2
		denom = math.sqrt(denom)
		for (k,v) in tf_dict.items():
			row.append(lexicon[k] - 1)
			col.append(idx-1)
			freq.append(v/denom)

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

s1 = np.diag(s)
us = np.dot(u,s1)
vs = np.dot(v,s1)

# print u
# print vt
# print "v::::"
# print v
#print s
#print len(s)



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
	print "wwwwwwwwww Similar objects to obj%d: %s wwwwwwwwwwww" % (idx,t)
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
print "######### TERM similarity ###########"
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


for query in queries:
	similarity = [(0,0)]* num_docs 
	terms = query.split()
	for t in terms:
		idx = lexicon[t.lower()]
		r = u[idx - 1,:]
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

