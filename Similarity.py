import json
import random
from collections import defaultdict

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

train = ug_data[:int(len(ug_data)*0.8)]
test = ug_data[int(len(ug_data)*0.8):]

def get_acc(predictions, test):
    n = len(predictions)
    pos = 0
    for i in range(n):
        if (predictions[i] == True and test[i][2] != None) or (predictions[i] == False and test[i][2] == None):
            pos += 1
    return pos / n

def Jaccard(s1, s2):
    nu = len(s1.intersection(s2))
    de = len(s1.union(s2))
    return nu / de if de > 0 else 0

def predict(user, game, threshold):
    similarities = []
    for other_game in gamesPeruser[user]:
        if other_game == game:
            continue
        similarities.append(Jaccard(usersPergame[other_game], usersPergame[game]))
    return max(similarities, default=0) > threshold

def model(threshold):
    predictions = []
    for u, g, _ in test:
        if predict(u, g, threshold):
            predictions.append(True)
        else:
            predictions.append(False)
    return predictions

usersPergame = defaultdict(set)
gamesPeruser = defaultdict(set)
for u, g, _ in train:
    usersPergame[g].add(u)
    gamesPeruser[u].add(g)

gamePeruser = defaultdict(set)
allgame = set()
for u, g, _ in train:
    gamePeruser[u].add(g)
    allgame.add(g)

allgame = list(allgame)
i = 0
n = len(test)

while i < n:
    u, g, _ = test[i]
    nb = random.choice(allgame)
    while nb in gamePeruser[u]:
        nb = random.choice(allgame)
    test.append((u, nb, None))
    i += 1

accuracy_list = []
thresholds = [i * 0.01 for i in range(1, 50)]

for threshold in thresholds:
    predictions = model(threshold)
    accuracy = get_acc(predictions, test)
    accuracy_list.append(accuracy)

best_index = accuracy_list.index(max(accuracy_list))
best_threshold = thresholds[best_index]
best_accuracy = max(accuracy_list)

print(f"Best Threshold: {best_threshold}")
print(f"Best Accuracy: {best_accuracy}")
