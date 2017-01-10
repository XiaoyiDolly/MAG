from math import *
import statistics as st
import numpy as np
from scipy.stats import multivariate_normal, mvn
from statsmodels.sandbox.distributions.extras import mvnormcdf as cdf
import csv
from collections import defaultdict
from random import randint
import pickle
import bisect
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
papernum = 0

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
    for id1 in range(papernum):
        for id2 in range(papernum):
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
    for id1 in range(papernum):
        for id2 in range(papernum):
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
        for str in trainset:
            pair = str.split(",")
            if(attlist[pair[0]][t] > 0 and attlist[pair[1]][t] > 0):
                x[t].append(attlist[pair[0]][t])  # x
                y[t].append(attlist[pair[1]][t])  # y
    hit = defaultdict(list)
    for t in range(topic_num):
        if(len(x[t])>0 and len(y[t]) >0):
            hit[t].append(x[t])
            hit[t].append(y[t])
    return hit

def bivariable_normal_model(attlist,hit,logFile):
    """
    calculate bivariable_normal pdf for each topic
    :param hit: citation pairs (topic i: [[x],[y]]) build models
    :return: bn pdf for each attribute
    """
    var = {}  # pdf for each topic
    score_matrix = defaultdict(dict)
    print("list of topics: ", hit.keys())
    logFile.write("************model***********\n")
    logFile.write("list of topics: "+str(hit.keys()))
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
                maxValue = 0
                minValue = 0
                var[t] = multivariate_normal(mu, cov)
                print("MODEL mu: %s///cov: %s\n" % (str(mu), str(cov)))
                print("###############use model to calculate score##############")
                for p1 in paperset:
                    for p2 in paperset:
                        if (p1 != p2):
                            p = var[t].logpdf([attlist[p1][t], attlist[p2][t]])
                            # print("topic: %i\tpdf: %f\tlogpdf: %f\t" % (t, var[t].pdf([x[t], y[t]]), p))
                            string = p1 + "," + p2
                            score_matrix[t][string] = p
                            # if (p > max): max = p
                            # if (p < min): min = p
                maxValue = max(score_matrix[t].values())
                minValue = min(score_matrix[t].values())
                logFile.write("max score: %f\nmin score: %f\n" % (maxValue, minValue))

                print("Start Normalize topic ",t)
                for key in score_matrix[t].keys():
                    p = score_matrix[t][key]
                    if ((maxValue-minValue)!=0): score_matrix[t][key] = (p-minValue)/(maxValue-minValue)
                    else: logFile.write("Cannot normalize, max == min\n")
            else:
                print("less than 3 points****", t)
                logFile.write("topic %i less then 3 points\nx:\t%s\ny:\t%s\n" % (t, str(x), str(y)))
        except:
            print("topic no model****",t)
            print(x)
            print(y)
            logFile.write("topic %i no model\nx:\t%s\ny:\t%s\n" % (t,str(x),str(y)))
    return score_matrix

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



def newPrecisionRecall(attlist, threshold):
    print(threshold)
    fn0 = "exp6/%f_output.csv" % threshold
    fn_log = "exp6/%f_log" % threshold
    fn1 = "exp6/%f_model.pickle" % threshold
    fn2 = "exp6/%f_score.pickle" % threshold
    # fn3 = "exp6/%f_model_score" %threshold
    with open(fn0, "w") as output, open(fn_log, "w") as logFile, open(pairs, "r") as test, open(fn1, "w") as F1, open(
        fn2, "w") as F2:
        lines = test.readlines()

        test_sets = defaultdict(list)
        for l in lines:
            pair = l.strip()
            test_sets[randint(0, 9)].append(pair)


        for n in test_sets.keys():
            print("#########TEST SET Batch ", n)
            logFile.write("#########TEST SET Batch %i\n" %n)
            train_set = []
            test = set()
            for i in test_sets.keys():
                if (i != n):
                    train_set += test_sets[i]
                if (i==n):
                    for v in test_sets[i]:
                        pair = v.split(",")
                        test.add(pair[0])
                        test.add(pair[1])
            print("training pair number: ", len(train_set))
            print("testset paper number: ",len(test))
            print("testset pair number: ",len(test_sets[n]))
            citations = read_trainset(attlist, train_set)
            model = bivariable_normal_model(citations,logFile)

            # print("store model of each topic and test set for batch ", n)
            # pickle.dump(model, F1)
            # pickle.dump(test_sets[n], F1)

            score = {}
            s = []
            for t in model.keys():
                print("****************calculate total score matrix for topic: ",t)
                for key in model[t].keys():
                    if key in score: score[key] += model[t][key]
                    else: score[key] = model[t][key]
                    # print("pair ",key, " score: ",score[key])
            print("store total score matrix for batch", n)
            pickle.dump(model,F2)
            pickle.dump(score, F2)

            print("sort score list and calculate threshold****************")
            z = papernum * (papernum - 1) - len(s)
            np0 = numpy.zeros(z)
            np1 = numpy.array(score.values())
            np = numpy.concatenate([np0,np1])
            _threshold = numpy.percentile(np,threshold*100)

            # for pair in score.keys():
            #     a = score[pair]
            #     bisect.insort(s, a)
            # index = int(math.ceil(papernum*(papernum-1)* threshold-z-1))
            # _threshold = s[index]
            # print("&&&&&&&&&length//max/min: ",len(s),s[len(s)-1],s[0])
            # logFile.write("&&&&&&&&&length: %i max: %f min: %f\n" %(len(s),s[len(s)-1],s[0]))
            print("&&&&&&&&&&threshold: ",_threshold)
            logFile.write("&&&&&&&&&threshold: %f\n" % _threshold)

            print("calculating PRECISION*************************")
            calcPrecision(test, test_sets[n], score, _threshold, output, logFile)

            print("calculating RECALL*******************************")
            calcRecall(test_sets[n], score, _threshold, output, logFile)

def calcPrecision(testset, testpair, score_dict,threshold, output, logFile):
    num = 0
    total = 0
    print("test papers #: ", len(testset))
    print("test pairs #: ", len(testpair))
    for p1 in testset:
        for p2 in testset:
            if (p1 != p2):
                string = p1 + "," + p2
                if(string in score_dict.keys() and score_dict[string]>threshold):
                    total += 1
                    print("pair: ", string, "total: ",total)
                    if (string in testpair):
                        num += 1
                        print("num: ", num)
                elif (string in score_dict.keys()): print(string)

    print("num of pairs\\total size:  %i   %i\n" % (num, total))
    if (total != 0):
        pcs = float(num) / total
        print("**********precision:", pcs)
        output.write("%i,%i,%f\n" % (num, total, pcs))
        logFile.write("# of pairs: %i\ntotal size: %i,\nprecision: %f\n" % (num, total, pcs))
    else:
        output.write("%i,\t0,\t0\n" % (num))
        logFile.write("testset no pair\n")

def calcRecall(testpair, score_dict, threshold, output, logFile):
    num = 0
    for pair in testpair:
        if (pair in score_dict.keys() and score_dict[pair]> threshold):
            num += 1
    print("num of pairs\\test size:  %i   %i\n" % (num, len(testpair)))
    recall = float(num) / len(testpair)
    print("**********recall:", recall)
    output.write("%i,%i,%f" % (num, len(testpair), recall))
    logFile.write("# of pairs: %i\ntest size: %i,\nrecall: %f\n\n" % (num, len(testpair), recall))



att_matrix = read_attributes()
with open(papers,'r') as pFile:
    lines = pFile.readlines()
    for l in lines:
        paperset.add(l.strip())
    papernum = len(paperset)
for t in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
    print("######threshold############# ",t)
    newPrecisionRecall(att_matrix,t)

# x = [-2.1, -1,  4.3]
# y = [3,  1.1,  0.12]
# a = np.cov(x, y)
# print a
# for value in a:
#     for v in value:
#         print v


