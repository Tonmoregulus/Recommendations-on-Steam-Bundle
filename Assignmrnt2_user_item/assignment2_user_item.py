import json
import gzip
import random
from collections import defaultdict, Counter
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

with open('data/user_game.json', 'r') as f:
    user_game = f.readlines()


ug_data = []

for item in user_game:
    parsed = json.loads(item)
    for user, games in parsed.items():
        for game in games:
            ug_data.append((f"u{user}", f"g{game}", 1))


train=ug_data[:int(len(ug_data)*0.8)]
test=ug_data[int(len(ug_data)*0.8):]

#########################################################################################
def get_acc(predictions,test):
    n=len(predictions)
    pos=0
    for i in range(n):
        if (predictions[i]==True and test[i][2]!=None) or (predictions[i]==False and test[i][2]==None):
            pos+=1
    return pos/n

def model_on_different_threshold(thre):
    gameCount = defaultdict(int)
    totalPlay = 0

    for user, game, _ in ug_data:
        gameCount[game] += 1
        totalPlay += 1

    mostPopular = [(gameCount[x], x) for x in gameCount]
    mostPopular.sort()
    mostPopular.reverse()

    returnthre = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        returnthre.add(i)
        if count > totalPlay*thre: break
    predictions=[]
    for u, g, _ in test:
        if g in returnthre:
            predictions.append(True)
        else:
            predictions.append(False)
    return predictions

def Jaccard(s1, s2):
    nu=len(s1.intersection(s2))
    de=len(s1.union(s2))
    return nu/de

def predict(u,g,threshold):
    similarities=[]
    for og in gamesPeruser[u]:
        if og==g:continue
        similarities.append(Jaccard(usersPergame[og],usersPergame[g]))
    if max(similarities,default=0)>threshold:
        return True
    else:
        return False

def model(threshold):
    predictions=0
    for u,g,_ in test:
        if predict(u,g,threshold)==True and _!=None:
            predictions+=1
        elif predict(u,g,threshold)==False and _==None:
            predictions+=1
        else:
            pass
    return predictions/len(test)


def model_on_different_threshold2(thre1,thre2):
    gameCount = defaultdict(int)
    totalPlay = 0

    for user, game, _ in ug_data:
        gameCount[game] += 1
        totalPlay += 1

    mostPopular = [(gameCount[x], x) for x in gameCount]
    mostPopular.sort()
    mostPopular.reverse()

    returnthre = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        returnthre.add(i)
        if count > totalPlay*thre1: break
    predictions=[]
    for u,g,_ in test:
        if g in returnthre or predict(u,g,thre2)==True:
            predictions.append(True)
        else:
            predictions.append(False)
    return predictions

#####################################################################################
#####################################################################################

gameCount = defaultdict(int)
totalPlay = 0

for user, game, _  in ug_data:
    gameCount[game] += 1
    totalPlay += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPlay/2: break
##################################################
gamePeruser=defaultdict(set)
allgame=set()
for u, g, _ in train:
    gamePeruser[u].add(g)
    allgame.add(g)

allgame=list(allgame)
i=0
n=len(test)

while i<n:
    u, g, _ =test[i]
    nb = random.choice(allgame)
    while nb in gamePeruser[u]:
        nb = random.choice(allgame)
    test.append((u, nb, None))
    i += 1

#####################################################
acclist=[]
for i in range(1,200):
    thre=0.005*i
    predictions=model_on_different_threshold(thre)
    acc=get_acc(predictions,test)
    acclist.append(acc)

acclist.index(max(acclist))
threshold=0.005*(acclist.index(max(acclist))+1)
acc2=max(acclist)

usersPergame = defaultdict(set)
gamesPeruser = defaultdict(set)
for u,g,_ in train:
    usersPergame[g].add(u)
    gamesPeruser[u].add(g)
##############################################################
acclist2=[]
thresholds=[]
for i in range(100,200):
    for j in range(1,9):
        thre1=i*0.005
        thre2=j*0.01
        predictions=model_on_different_threshold2(thre1,thre2)
        acc=get_acc(predictions,test)
        thresholds.append((thre1,thre2))
        acclist2.append(acc)
        #print(acc)
max(acclist2)
acc=max(acclist2)


index=acclist2.index(acc)
thre1,thre2=thresholds[index]

