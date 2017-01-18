import numpy as np
from random import randint
import statistics as st
from collections import defaultdict
from scipy.stats import multivariate_normal
from scipy.sparse import dok_matrix
import csv
import json

topic_fn = "IEEE/document-topic-distributions.csv"
cite = "IEEE/referee_referers.csv"
pairs = "IEEE/pairs"
log = "IEEE/exp1/log.txt"
attribute_matrix = "IEEE/exp1/attribute_M.txt"
att_matrix = {}
Msum = {} # matrix of score
paperset = set()
doc_num = 0
att_num = 0

def initAttributeMatrix():
    global doc_num
    global att_num
    global att_matrix
    global paperset

    with open(topic_fn, "r") as f1, open(attribute_matrix,"w") as f2:
        lines = f1.readlines()
        for l in lines:
            str = l.strip().split(",")
            doc_num = int(str[0])
            i = 0
            # paperset.add(int(str[0]))
            for s in str:
                if(i > 0 and i%2 == 0):
                    att_matrix[int(str[0]), int(str[i-1])] = float(str[i])
                    if int(str[i-1])>att_num:
                        att_num = int(str[i-1])
                i+=1



def savePairs():
    print(len(paperset))
    f = open(cite)
    csv_c = csv.reader(f, delimiter='\t')
    # print(paperset)
    with open(pairs, "w") as output:
        for row in csv_c:
            l = ''.join(c for c in row[1] if c not in '[]').split(',')
            for e in l:
                e = e.strip()
                if (int(e) in paperset and int(row[0]) in paperset):
                    print(row[0],e)
                    if (row[0] != e):
                        output.write("%s,%s\n" % (row[0], e))


def read_trainset_attributes(train_set):
    x_dict = dict()
    y_dict = dict()
    for item in train_set:
        ids = item.strip().split(',')
        id1 = int(ids[0])
        id2 = int(ids[1])
        # print id1,id2
        doc1_topics= []
        doc2_topics = []
        for t in range(0,att_num):
            if (id1,t) in att_matrix:
                doc1_topics.append(t)
            if (id2,t) in att_matrix:
                doc2_topics.append(t)

        # print doc1, doc2
        # doc1_topics = []
        # for i in doc1.keys():
        #     doc1_topics.append(i[1])
        # # print(doc1_topics)
        # doc2_topics = []
        # for i in doc2.keys():
        #     doc2_topics.append(i[1])
        # print(doc2_topics)

        same_topics = set(doc1_topics).intersection(doc2_topics)
        print("same topics: ",same_topics)

        for t in same_topics:
            if att_matrix[(id1,t)] > 0.2 and att_matrix[(id2,t)]> 0.2:
                if t in x_dict:
                    x_dict[t].append(att_matrix[(id1,t)])
                    y_dict[t].append(att_matrix[(id2,t)])
                else:
                    x_dict[t]= [att_matrix[(id1, t)]]
                    y_dict[t] = [att_matrix[(id2, t)]]
                # print("attribute value: "+ str(att_matrix[(id1,t)])+" to x: "+ str(x_dict[t]))
                # logFile.write("add score for topic: %i\n" % t)
    print(x_dict.keys())
    return x_dict,y_dict


def bivariable_normal_model(x_dict,y_dict,logFile):
    global Msum
    """
    calculate bivariable_normal pdf for each topic
    :param hit: citation pairs (topic i: [[x],[y]]) build models
    :return: bn pdf for each attribute
    """
    var = {}  # pdf for each topic
    # scoreoftopic = dict()
    # Msum = {}
    print("list of topics: ", x_dict.keys())
    logFile.write("************model***********\n")
    logFile.write("list of topics: "+str(x_dict.keys())+"\n")
    for t in x_dict.keys():
        x =x_dict[t]
        y= y_dict[t]
        logFile.write("\ntopic: %i\n" % t)
        print ("topic: %i" % t)
        print("number of points for x, y: ", len(x), len(y))
        logFile.write("number of points for x, y: %s,%s\n" % (str(len(x)),str(len(y))))
        mean_x = st.mean(x)
        mean_y = st.mean(y)
        mu = [mean_x,mean_y]
        cov = np.cov(x,y)

        try :
            if(len(x)>2):
                maxValue = 0
                var[t] = multivariate_normal(mu, cov)
                zeroscore = var[t].logpdf([0, 0])
                minValue = zeroscore
                # print("MODEL mu: %s///cov: %s\n" % (str(mu), str(cov)))
                logFile.write("MODEL mu: %s\ncov: %s\n" % (str(mu), str(cov)))
                print("###############use model to calculate score##############")
                docs = []

                for id in paperset:
                    if (id,t) in att_matrix and att_matrix[id,t]>0.2:
                        docs.append(id)

                Mscore = {}
                print ("# of nonzero attributes: ",len(docs))
                logFile.write ("# of nonzero attributes: %i \n" % len(docs))
                for p1 in docs:
                    for p2 in docs:
                        if (p1 != p2):
                            p = var[t].logpdf([att_matrix[p1,t], att_matrix[p2,t]])
                            if p<minValue:
                                print("p score < zero score: ", att_matrix[p1,t], att_matrix[p2,t],minValue,p)
                                minValue = p
                            # print("attribute values: "+ str(att_matrix[(p1,t)])+str(att_matrix[(p2,t)]))
                            # print("topic: %i\tlogpdf: %f\tlength: %i" % (t,  p, len(docs)))
                            Mscore[p1,p2] = p

                print "************finish calculation************"
                 # dok_matrix((len(paperset),len(paperset)), dtype=np.float64)
                # i = 0
                if len(docs)==0:
                    print("empty docs")
                    continue
                else:
                    maxValue = max(Mscore.values())

                print("Mscore length: ", len(Mscore.values()))
                logFile.write("Mscore length: %i\t%f\n" % (len(Mscore.values()),
                                                           float(len(Mscore.values()))/(len(paperset)*(len(paperset)-1))))

                for x in paperset:
                    for y in paperset:
                        if x!=y:
                            if (x,y) not in Mscore:
                                Mscore[x, y] = 0
                            else:
                                Mscore[x, y] = (Mscore[x,y]-minValue)/(maxValue-minValue)

                            if (x,y) in Msum:
                                Msum[x,y]+=Mscore[x,y]
                            else:
                                Msum[x,y] = Mscore[x,y]

                print("maxValue, minValue:" + str(maxValue) + ' '+ str(minValue) )
                logFile.write("max score: %f\nmin score: %f\n" % (maxValue, minValue))

            else:
                print("less than 3 points****", t)
                logFile.write("topic %i less then 3 points\nx:\t%s\ny:\t%s\n" % (t, str(x), str(y)))
        except:
            print("topic no model****",t)
            print(mu)
            print(cov)
            logFile.write("topic %i no model\nx:\t%s\ny:\t%s\n" % (t,str(x),str(y)))
    print("Msum length:" , len(Msum.values()))
    print((4032081, 4032086) in Msum)
    print("60 percentile: ",np.percentile(Msum.values(), 60))
    print("70 percentile: ", np.percentile(Msum.values(), 70))
    print("80 percentile: ", np.percentile(Msum.values(), 80))

def trainmodels(logFile):
    global Msum
    """
    initiate attribute matrix: attlist
    read each line to extract attributes of node
    :return: each document topic distribution list
    """

    # fn_log = "Mashup/exp/%f_log" % t
    # fn1 = "Mashup/exp6/%f_model.pickle" % t
    # fn2 = "Mashup/exp6/%f_score.pickle" % t
    # # fn3 = "exp6/%f_model_score" %threshold
    fn0 = "IEEE/exp1/output.txt"
    with open(pairs, "r") as c ,  open(fn0, 'w') as output:
        lines = c.readlines()
        test_sets = defaultdict(list)
        count = 0
        for l in lines:
            pair = l.strip()
            test_sets[randint(0, 9)].append(pair)
            count+=1
            paperset.add(int(pair.split(',')[0]))
            paperset.add(int(pair.split(',')[1]))
        for n in range(0, 10):  # test_sets.keys():
            print("#########TEST SET Batch ", n)
            logFile.write("#########TEST SET Batch %i\n" % n)
            train_set = []
            test = set()
            for i in test_sets.keys():
                if (i != n):
                    train_set += test_sets[i]
                if (i == n):
                    for v in test_sets[i]:
                        pair = v.split(",")
                        test.add(int(pair[0]))
                        test.add(int(pair[1]))
            print("# training pair: ", len(train_set))
            print("# testset paper: ", len(test))
            print("# testset pair: ", len(test_sets[n]))
            print("\n*************read trainset attributes*************")
            x, y = read_trainset_attributes(train_set)
            print("\n*************train models:**************")
            bivariable_normal_model(x, y, logFile)
            scoreoftopic = "IEEE/exp1/%i_scorematrix.txt" % n
            # with open(scoreoftopic, 'w') as scoreFile:
            # np.savetxt(scoreFile, Msum, fmt='%.18g')
            json.dump(remap_keys(Msum), open(scoreoftopic, 'w'))

            print("\n*****************start test*****************")
            Test(test,test_sets[n],logFile,output)

def remap_keys(mapping):
    return  [{'key':k, 'value': v} for k, v in mapping.iteritems()]

def Test(paper, test_sets,logFile,output):

    for t in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        threshold = np.percentile(Msum.values(), t * 100)
        print("######threshold######## ", t)
        print("######percentile threshold######## ", threshold)
        # fn0 = "IEEE/exp1/%f_output.txt" % t
        # with open(fn0, 'w') as output:
        logFile.write("threshold and percentile: %f\t%f\n" %(t,threshold))
        output.write("%f\n" %t)
        print("calculating PRECISION*************************")
        calcPrecision(paper, test_sets, threshold, output, logFile)

        print("calculating RECALL*******************************")
        calcRecall(test_sets, threshold, output, logFile)


def calcPrecision(testset, testpair, threshold, output, logFile):
    global Msum
    num = 0
    total = 0
    print("Msum: ", len(Msum.values()))
    print("test papers #: ", len(testset))
    print("test pairs #: ", len(testpair))
    for p1 in testset:
        for p2 in testset:
            if (p1 != p2):
                # p1 = int(p1.strip())
                # p2 = int(p2.strip())
                if (p1,p2) in Msum:
                    if Msum[p1,p2] > threshold:
                        total += 1
                        # print("pair: ", string, "total: ",total)
                        if (str(p1)+','+str(p2) in testpair):
                            num += 1
                            # print("num: ", num)
                else:
                    print ("not in Msum: "+ str(p1) +' '+ str(p2))
                    logFile.write("not in Msum %s %s"  %(str(p1) ,str(p2)))
                    print(p1 in paperset)
                    print(p2 in paperset)
                # elif (string in score_dict.keys()): print(string)

    print("num of pairs\\total size:  %i   %i\n" % (num, total))
    if (total != 0):
        pcs = float(num) / total
        print("**********precision:", pcs)
        output.write("%i,\t%i,\t%f\n" % (num, total, pcs))
        logFile.write("# of pairs: %i\ntotal size: %i,\nprecision: %f\n" % (num, total, pcs))
    else:
        output.write("%i,\t0,\t0\n" % (num))
        logFile.write("testset no pair\n")

def calcRecall(testpair, threshold, output, logFile):
    num = 0
    for pair in testpair:
        str = pair.split(',')
        p1 = int(str[0])
        p2 = int(str[1])
        if (Msum[p1, p2] > threshold):
            num += 1
    print("num of pairs\\test size:  %i   %i\n" % (num, len(testpair)))
    recall = float(num) / len(testpair)
    print("**********recall:", recall)
    output.write("%i,\t%i,\t%f\n" % (num, len(testpair), recall))
    logFile.write("# of pairs: %i\ntest size: %i,\nrecall: %f\n\n" % (num, len(testpair), recall))


# att_matrix = read_attributes()
# with open(papers,'r') as pFile:
#     lines = pFile.readlines()
#     for l in lines:
#         paperset.add(l.strip())
#     papernum = len(paperset)
# for t in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
#     print("######threshold############# ",t)
#     newPrecisionRecall(att_matrix,t)
print("*****************Start initialize*****************")
initAttributeMatrix()

# with open(attribute_matrix,"r") as f:
#     m = np.loadtxt(f)
#     att_matrix = dok_matrix(m)
    # att_matrix_array = att_matrix.toarray()
with open(log, "w") as logFile:
    logFile.write("# doc: %i\n" % doc_num)
    logFile.write("# att: %i \n" % att_num)
    logFile.write("# data: %i\n" %len(paperset))
    logFile.write("\n*****************Start train model*****************\n")
    trainmodels(logFile)

# mu = 	[0.68422703165211529, 0.68386232509446065]
# cov = np.array([[ 0.04275162, -0.01158742],[-0.01158742,  0.04281575]])
# v= multivariate_normal(mu, cov)
# print(v.logpdf([0.1,0.2]))
# savePairs()

def model():
    row = np.array([0,3,1,0])
    col = np.array([0,3,0,2])
    data = np.array([0.5,0.3,0.2,0.1])

    m = dok_matrix((4,4),dtype=np.float64)
    i=0
    for x,y in zip(row,col):
        m[x,y] = data[i]
        i+=1
    b = {}
    b[1,2]=1
    b[1,3]=2
    print("percentile:  ",np.percentile(b.values(),100))
    print((1,3)in b)

    row = np.array([0, 1, 0])
    col = np.array([0, 0, 2])
    data = np.array([0.5, 0.3, 0.2, 0.1])
    a = dok_matrix((3,3),dtype=np.float64)
    i=0
    for x,y in zip(row,col):
        a[x,y] = data[i]
        i+=1
    print(a)

    # print(m+a)
    for v in m.getcol(2).keys():
        print(v[0] )
    n = dok_matrix((4, 4), dtype=np.float64)
    i = 0
    for x, y in zip(row, col):
        # print x, y
        n[y,x] = data[i]
        i += 1
    print(n[1,1])
    # print((m+n).toarray())

    # print(m.toarray())
    doc1 = m.getrow(0)
    doc2 = m.getrow(1)

    # print(doc1.viewkeys())
    topic1 = []
    for i in doc1.keys():
        topic1.append(i[1])
    # print(topic1 )
    topic2 = []
    for i in doc2.keys():
        topic2.append(i[1])
    # print(topic2)
    topic12 = set([])
    topic12 = set(topic1).intersection(topic2)
    # print(topic12)
    np.set_printoptions(precision=2)
    with open("topic.np","w") as f1, open("topic.txt","w") as f2:
        np.save(f1,doc1.toarray())
        np.save(f1,doc2.toarray())
        np.savetxt(f2,doc2.toarray())
        np.savetxt(f2,m.toarray())# ,fmt='%1.4E'
    with open("topic.np", "r") as f, open("topic.txt","r") as g:
        m = np.load(f)
        n = np.load(f)
        # a = np.loadtxt(g)
        # b = np.loadtxt(g)
        # M = np.array([])
        # M = dok_matrix(b)
    # print  m,n #, M

# model()