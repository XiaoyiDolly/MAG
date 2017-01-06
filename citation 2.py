from math import *
import statistics as st
import numpy as np
from scipy.stats import multivariate_normal, mvn
from statsmodels.sandbox.distributions.extras import mvnormcdf as cdf
import csv
from collections import defaultdict
from random import randint
import pickle
import numpy
import math


topic_fn = "yuhao2/paperId and topics.csv" #"MAG_training/document-topic-distributions.csv"
citation_fn = "yuhao2/referee_referers.csv" #"MAG_training/referee_refers.csv"
pairs = "yuhao2/pairs"
scores = "yuhao2/scores"
papers = "yuhao2/paperset"

topic_num = 12
paperset = set()  # list of publications
referee_referers = defaultdict(list)


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)

def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    if denominator !=0:
        return round(numerator / float(denominator), 3)
    else:
        return 0


def read_attributes():
    """
    initiate attribute matrix: attlist
    read each line to extract attributes of node
    :return: each document topic distribution list
    """
    f = open(topic_fn)
    csv_f = csv.reader(f)
    attlist = {}
    for row in csv_f:
        l = [(float(x) if x else 0.0) for x in row[2:]]
        # publications.append(row[0])
        attlist[row[0]] = [(0.0 if x<0 else x) for x in l]
        # print(row[0],"&&&&")
        # print(attlist[row[0]])
    f.close()
    return attlist


def generate_citation(attlist):
    """
    construct citation matrix based on cosine similarity between two publications
    :param attlist: attibute list
    :return: citation pairs (topic i: [[x],[y]])
    """
    matrix = defaultdict(dict)
    x = defaultdict(list)
    y = defaultdict(list)
    i=0
    for id1 in attlist.keys():
        for id2 in attlist.keys():
            if id1 != id2:
                matrix[id1][id2] = 1 if cosine_similarity(attlist[id1], attlist[id2])>0.8 else 0
                if matrix[id1][id2] == 1:
                    i+=1
                    for t in range(topic_num):
                        x[t].append(attlist[id1][t])   #x
                        y[t].append(attlist[id2][t])   #y
            else:
                matrix[id1][id2] = 1
    print("#citations based on similarity: ",i,"\n")
    hit = defaultdict(list)
    for t in range(topic_num):
        hit[t].append(x[t])
        hit[t].append(y[t])
    return hit

def prb_citation_pdf(attlist,model):
    """
    generate MAG based on probability
    :param attlist: to get attribute list for a node
    :param model: pdf functions for each attribute
    :return: MAG
    """
    for id1 in range(paper_num):
        for id2 in range(paper_num):
            if id1 != id2:
                p = bn_pdf(attlist[id1],attlist[id2],model)
                print(p)



def prb_citation_cdf(attlist):
    """
    generate MAG based on probability
    :param attlist: to get attribute list for a node
    :param model: pdf functions for each attribute
    :return: MAG
    """
    for id1 in range(paper_num):
        for id2 in range(paper_num):
            if id1 != id2:
                p = bn_cdf(attlist[id1], attlist[id2])
                print(p)



def real_citation(attlist):
    """
    initiate citation list from real data
    :param attlist: attibute list
    :param pub: publication title list
    :return: (topic i: [[x],[y]])
    """
    f = open(citation_fn)
    csv_c = csv.reader(f,delimiter='\t')
    x = defaultdict(list)
    y = defaultdict(list)
    i = 0
    with open(pairs,'w') as output, open(papers,'w') as pFile:
        for row in csv_c:
            l = ''.join(c for c in row[1] if c not in '[]').split(',')
            for e in l:
                keys = attlist.keys()
                if(e in keys and row[0] in keys):
                    paperset.add(row[0])
                    paperset.add(e)
                    i += 1
                    # print(row[0],e)
                    if(row[0]!=e):
                        output.write("%s,%s\n" % (row[0], e))
                        for t in range(topic_num):
                            x[t].append(attlist[row[0]][t])  # x
                            y[t].append(attlist[e][t])  # y
                            referee_referers[row[0]].append(e)
        for v in paperset:
            pFile.write("%s\n" % v)
        print("paperset length: ",len(paperset))
    f.close()
    print("#citations based on real data: ",i,"\n")
    hit = defaultdict(list)
    for t in range(topic_num):
        hit[t].append(x[t])
        hit[t].append(y[t])
        # print("topic citations***",t)
        # print(hit[t][0])
        # print(hit[t][1])
    return hit

def read_trainset(attlist, trainset):
    x = defaultdict(list)
    y = defaultdict(list)
    for t in range(topic_num):
        for pair in trainset:
            if(attlist[pair[0]][t] > 0 and attlist[pair[1]][t] > 0):
                x[t].append(attlist[pair[0]][t])  # x
                y[t].append(attlist[pair[1]][t])  # y
    hit = defaultdict(list)
    for t in range(topic_num):
        if(len(x[t])>0 and len(y[t]) >0):
            hit[t].append(x[t])
            hit[t].append(y[t])
    return hit

def bivariable_normal_model(hit,logFile):
    """
    calculate bivariable_normal pdf for each topic
    :param hit: citation pairs (topic i: [[x],[y]]) build models
    :return: bn pdf for each attribute
    """
    var = {}  # pdf for each topic
    score_matrix = defaultdict(dict)
    print("list of topics: ", hit.keys())
    logFile.write("************model***********\n")
    logFile.write("list of topics: %s\n" % str(hit.keys()))
    for t in hit.keys():
        x =hit[t][0]
        y= hit[t][1]
        logFile.write("topic: %i\n" % t)
        print("number of points for x, y: ", len(x), len(y))
        logFile.write("number of points for x, y: %s,%s\n" % (str(len(x)),str(len(y))))
        mean_x = st.mean(x)
        mean_y = st.mean(y)
        mu = [mean_x,mean_y]
        cov = np.cov(x,y)
        try :
            if(len(x)>2):
                max = -99999.0
                min = 99999.0
                var[t] = multivariate_normal(mu, cov)
                logFile.write("model mu: %s///cov: %s\n" % (str(mu), str(cov)))
                # score[t]
                for p1 in paperset:
                    for p2 in paperset:
                        if (p1 != p2):
                            p = var[t].logpdf([x[t], y[t]])
                            logFile.write("topic: %i\tpdf: %f\tlogpdf: %f\t" % (t, var[t].pdf([x[t], y[t]]), p))
                            str = p1 + "," + p2
                            score_matrix[t][str] = p
                            if (p > max): max = p
                            if (p < min): min = p
                logFile.write("max score: %f\nmin score: %f\n" % (max, min))
                print("Start Normalize topic ",t)
                for key in score_matrix[t].keys():
                    p = score_matrix[t][key]
                    if ((max-min)!=0): score_matrix[t][key] = (p-min)/(max-min)
                    else: logFile.write("Cannot normalize, max == min\n")
            else:
                print("less than 3 points****", t)
                logFile.write("topic %i less then 3 points\nx:\t%s\ny:\t%s\n" % (t, str(x), str(y)))
        except:
            print("topic no model****",t)
            print(x)
            print(y)
            logFile.write("topic %i no model\nx:\t%s\ny:\t%s\n" % (t,str(x),str(y)))
    return var

def pdf_score(attlist,model, id1, id2, logFile):
    if id1 != id2:
        p = bn_pdf(attlist[id1], attlist[id2], model, logFile)
        # print("attribute id1 and id2: \n",attlist[id1],"\n", attlist[id2])
        return p

def bn_pdf(x,y,var, logFile):
    """
    calculate possibility of citation i to j: Multiply pdf of each attribute
    :param x: attribute list of publication i
    :param y: attribute list of publication j
    :return: possibility of citation i to j
    """


    for t in range(topic_num):
        if (t in var.keys()):
            p = var[t].logpdf([x[t], y[t]])
            if (p > max): max = p
            if (p < min): min = p
            print(t, " logpdf: ", var[t].logpdf([x[t], y[t]]))
            logFile.write("topic: %i\tpdf: %f\tlogpdf: %f\t" % (t, var[t].pdf([x[t], y[t]]), a))
        print(t, " total logpdf: ",p)
    return p if p != 0 else 0

def bn_cdf______(x, y):
    p = 1
    for t in range(topic_num):
        low = np.array([x[t] - 1, y[t] - 1])
        upp = np.array([x[t] + 1, y[t] + 1])
        mu = np.array([st.mean(x), st.mean(y)])
        s = np.cov(x,y)
        # pn, i = mvn.mvnun(low, upp, mu, s)
        pn,i = (cdf(upp, mu, s, low) if np.sum(np.sqrt(np.diag(s)))!=0 else 0,0)
        p *= pn
    return log10(p) if p != 1 else 0

def PrecisionRecallfromFile( threshold):
    print(threshold)
    fn0 = "yuhao2/%f_output" % threshold
    fn1 = "yuhao2/%f_model.pickle" %threshold
    fn2 = "yuhao2/%f_score.pickle" %threshold
    with open(fn0,"w") as output, open(papers,'r') as pFile, open(pairs,"r") as test, open(fn1,"rb") as f, open(fn2,"rb") as s:
        lines = test.readlines()
        all_pairs = set()
        train_set = defaultdict(list)
        model = {}
        score_dict = {}
        # train_set = []
        for n in range(0,9):
            print("#########TEST SET Batch ", n)
            output.write("#########TEST SET Batch %i **********\n" %n)
            num = 0
            maxScore = 0
            total = 0
            model = pickle.load(f)
            train_set[n] = pickle.load(f)
            score_dict = pickle.load(s)
            for k in model.keys():
                print(n, "MODEL: ",model[k])

            print("calculating precision....")
            for p1 in paperset:
                for p2 in paperset:
                    if (p1 != p2):
                        str = p1 + "," + p2
                        score = score_dict[str]
                        if (score > maxScore): maxScore = score

            for key in score_dict.keys():
                if (score_dict[key] > threshold * maxScore):
                    total += 1
                    if (key in all_pairs):
                        num += 1
            print("num of pairs\\total size:  %i   %i" % (num, total))
            pcs = float(num) / total
            print("**********precision:", pcs)
            output.write("%i    %i   %i   %f\n" % (n, num, total, pcs))

            num = 0
            print("\ncalculating recall....")
            for pair in train_set[n]:
                if (score_dict[pair[0] + "," + pair[1]] > threshold * maxScore):
                    num += 1
            print("num of pairs\\test size:  %i   %i" % (num, len(train_set[n])))
            recall = float(num) / len(train_set[n])
            print("**********recall:", recall)
            output.write("%i    %i   %i   %f\n" % (n, num, len(train_set[n]), recall))

def newPrecisionRecall(attlist, threshold):
    print(threshold)
    fn0 = "exp5_0/%f_output.csv" % threshold
    fn_log = "exp5_0/%f_log" % threshold
    fn1 = "exp5_0/%f_model.pickle" % threshold
    fn2 = "exp5_0/%f_score.pickle" % threshold
    with open(fn0, "w") as output, open(fn_log, "w") as logFile, open(pairs, "r") as test, open(fn1, "w") as f, open(
        fn2, "w") as s:
        lines = test.readlines()
        all_pairs = set()

        score_dict = defaultdict(dict)
        test_sets = defaultdict(list)
        for l in lines:
            pair = l.strip().split(',')
            test_sets[randint(0, 9)].append(pair)
            all_pairs.add(l.strip())

        train_set=[]
        for n in test_sets.keys():
            print("#########TEST SET Batch ", n)
            logFile.write("#########TEST SET Batch %i **********\n" %n)
            num = 0
            maxScore = -999999
            minScore = 999999
            total = 0
            test = set()
            for i in test_sets.keys():
                if (i != n):
                    train_set += test_sets[i]
                if (i==n):
                    for v in test_sets[i]:
                        test.add(v[0])
                        test.add(v[1])
            print("testset paper number: ",len(test))
            print("testset pair number: ",len(test_sets[n]))
            citations = read_trainset(attlist, train_set)
            model = bivariable_normal_model(citations,logFile)

            print("store model and test set for batch ", n)
            pickle.dump(model, f)
            pickle.dump(test_sets[n], f)





def PrecisionRecall(attlist, threshold):
    print(threshold)
    fn0 = "exp5_0/%f_output.csv" % threshold
    fn_log = "exp5_0/%f_log" % threshold
    fn1 = "exp5_0/%f_model.pickle" %threshold
    fn2 = "exp5_0/%f_score.pickle" %threshold
    with open(fn0,"w") as output, open(fn_log,"w") as logFile, open(pairs,"r") as test, open(fn1,"w") as f, open(fn2,"w") as s:
        lines = test.readlines()
        all_pairs = set()

        score_dict = {}
        test_sets = defaultdict(list)
        for l in lines:
            pair = l.strip().split(',')
            test_sets[randint(0, 9)].append(pair)
            all_pairs.add(l.strip())

        train_set=[]
        for n in test_sets.keys():
            print("#########TEST SET Batch ", n)
            logFile.write("#########TEST SET Batch %i **********\n" %n)
            num = 0
            maxScore = -999999
            minScore = 999999
            total = 0
            test = set()
            for i in test_sets.keys():
                if (i != n):
                    train_set += test_sets[i]
                if (i==n):
                    for v in test_sets[i]:
                        test.add(v[0])
                        test.add(v[1])
            print("testset paper number: ",len(test))
            print("testset pair number: ",len(test_sets[n]))
            citations = read_trainset(attlist, train_set)
            model = bivariable_normal_model(citations,logFile)

            print("store model and test set for batch ", n)
            pickle.dump(model,f)
            pickle.dump(test_sets[n], f)

            for p1 in paperset:
                for p2 in paperset:
                    if (p1 != p2):
                        score = pdf_score(attlist, model, p1, p2,logFile)
                        str = p1 + "," + p2
                        score_dict[str] = score
                        if (score > maxScore): maxScore = score
                        if (score < minScore): minScore = score
            print("store scores of all pairs for batch ", n)
            pickle.dump(score_dict, s)
            logFile.write("max score: %f\nmin score: %f\n" % (maxScore, minScore))
            print("calculating precision....")
            for p1 in test:
                for p2 in test:
                    if (p1 != p2):
                        str = p1 + "," + p2
                        pair = [p1,p2]
                        if (score_dict[str]-minScore > threshold * (maxScore-minScore)):
                            total += 1
                            if(pair in test_sets[n]):
                                num += 1

            print("num of pairs\\total size:  %i   %i\n" % (num,total))
            if(total!=0):
                pcs = float(num) / total
                print("**********precision:", pcs)
                output.write("%i,%i,%i,%f\n" % (n, num, total, pcs))
                logFile.write("# of pairs: %i\ntotal size: %i,\nprecision: %f\n" % (num, total, pcs))
            else:
                output.write("%i,\t%i,\t0,\t0\n" % (n, num))
                logFile.write("testset no pair\n")
                pcs = 0
                output.write("%i,%i,%i,%f\n" % (n, num, total, pcs))

            num = 0
            print("calculating recall....")
            for pair in test_sets[n]:
                if (score_dict[pair[0]+","+pair[1]] > threshold * maxScore):
                    num += 1
            print("num of pairs\\test size:  %i   %i\n" % (num, len(test_sets[n])))
            recall = float(num)/len(test_sets[n])
            print("**********recall:", recall)
            output.write("%i,%i,%i,%f" % (n, num, len(test_sets[n]), recall))
            logFile.write("# of pairs: %i\ntest size: %i,\nrecall: %f\n\n" % (num, len(test_sets[n]), recall))

def precision(attlist,threshold):
    with open("yuhao2/precision", "w") as output, open(pairs,"r") as test, open(papers,"r") as pFile:
        lines = test.readlines()
        all_pairs = set()
        score_dict = {}
        test_sets = defaultdict(list)
        for l in lines:
            pair = l.strip().split(',')
            test_sets[randint(0, 9)].append(pair)
            all_pairs.add(l.strip())
            # print("all_pairs initialize: ",l.strip(),l.strip() in all_pairs)
        lines = pFile.readlines()
        for l in lines:
            paperset.add(l.strip())
        train_set = []
        for n in test_sets.keys():
            print("#########TEST SET Batch ", n)
            # output.write("#########TEST SET Batch %i **********\n" % n)
            num = 0
            maxScore = 0
            total = 0
            for i in test_sets.keys():
                if (i != n):
                    train_set += test_sets[i]
            citations = read_trainset(attlist, train_set)
            model = bivariable_normal_model(citations)
            for p1 in paperset:
                for p2 in paperset:
                    if (p1!=p2):
                        score = pdf_score(attlist, model, p1,p2)
                        str = p1+","+p2
                        score_dict[str] = score
                        if(score>maxScore): maxScore = score
                        print(str,"/////",score)
            print("maxscore:", maxScore)
            for key in score_dict.keys():
                if (score_dict[key] > threshold * maxScore):
                    total += 1
                    if(key in all_pairs):
                        num += 1
            output.write("num of pairs:     %i \n" % num)
            output.write("total size:       %i\n" % total)
            pcs = float(num) / total
            print("precision:", pcs, '\n')
            output.write("precision:        %f\n" % pcs)

def score_threshold(attlist):
    citations = real_citation(attlist)
    model = bivariable_normal_model(citations)
    num = 0
    score_dict = {}
    with open(pairs, "r") as test, open(scores, "w") as scorefile:
        lines = test.readlines()
        i = 0
        maxscore = 0
        for l in lines:
            # print("new pair: ", l)
            i += 1
            ids = l.strip().split(',')
            score = pdf_score(attlist, model, ids[0], ids[1])
            score_dict[int(ids[0]), int(ids[1])] = score
            if (score > maxscore):
                maxscore = score
            # if (score > 0.8):
            #     num += 1
        for k in score_dict.keys():
            score_dict[k] /=maxscore
            print(k,": ", score_dict[k])
            scorefile.write("%i,%i,%s\n" % (k[0],k[1], score_dict[k]))
        print("# of (score >0.8):  ", num)
        print("# of pairs:  ", i)
        print("recall:", float(num) / i)

# att_matrix = read_attributes()
# with open(papers,'r') as pFile:
#     lines = pFile.readlines()
#     for l in lines:
#         paperset.add(l.strip())
# for t in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
#     print("\n######threshold####    ",t)
#     PrecisionRecall(att_matrix,t)

# myDict = {'Apple': {'American':'16', 'Mexican':10, 'Chinese':5},
#           'Grapes':{'Arabian':'25','Indian':'20'} }
myDict = defaultdict(dict)
myDict['Apple']['aa'] = 2
print(myDict['Apple'],type(myDict))


# PrecisionRecallfromFile(0.600000)

# with open("0.600000_score.pickle",'rb') as s, open("0.600000_model.pickle","rb") as f:
#     for i in range(0,9):
#         max = 0
#         # model = pickle.load(f)
#
#         score_dict = pickle.load(s)
#         for key in score_dict.keys():
#             score = score_dict[key]
#             if score>max:
#                 max = score
#         print(max)



# citations = real_citation(att_matrix)
# score_threshold(att_matrix,0.8)
# recall(att_matrix)
# precision(att_matrix)


# prb_citation_pdf(att_matrix, model)

# prb_citation_cdf(att_matrix)


# p = possibility(pub.index('19730019475'),pub.index('19730005103'), att_matrix, citations)
# print("possibility:",p)





