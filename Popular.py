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

def model_on_different_threshold(threshold):
    gameCount = defaultdict(int)
    totalPlay = 0

    for user, game, _ in ug_data:
        gameCount[game] += 1
        totalPlay += 1

    mostPopular = [(gameCount[x], x) for x in gameCount]
    mostPopular.sort(reverse=True)

    selected_games = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        selected_games.add(i)
        if count > totalPlay * threshold:
            break

    predictions = []
    for u, g, _ in test:
        if g in selected_games:
            predictions.append(True)
        else:
            predictions.append(False)

    return predictions

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
for i in range(1, 200):
    threshold = 0.005 * i
    predictions = model_on_different_threshold(threshold)
    accuracy = get_acc(predictions, test)
    accuracy_list.append(accuracy)

best_index = accuracy_list.index(max(accuracy_list))
best_threshold = 0.005 * (best_index + 1)
best_accuracy = max(accuracy_list)

print(f"Best Threshold: {best_threshold}")
print(f"Best Accuracy: {best_accuracy}")
