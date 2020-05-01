import numpy as np
from sklearn.preprocessing import normalize
import csv


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss1(predictions, labels):
    return (-labels * np.log(predictions) - (1 - labels) * np.log(1 - predictions)).mean()


def update_w():
    col1 = sigmoid((np.dot(X, theta)))
    temp = np.ones_like(col1)
    col2 = temp - col1

    yhatT = np.concatenate((col1, col2)).reshape((2, -1))
    yhat = yhatT.T

    inv = np.linalg.inv(np.dot(yhatT, yhat))
    temp2 = np.dot(inv, yhatT)

    res = np.dot(temp2, s)

    return res


def renyi_value():
    return np.linalg.norm(w * np.dot(X, theta) - s)


def grad2():

    a = w[0] - w[1]
    b = s - w[1] * np.ones_like(s)

    sigm = sigmoid(np.dot(X, theta))
    sigm2 = np.ones_like(sigm) - sigm

    sigm_dev = np.multiply(sigm, sigm2)

    inner1 = a * sigmoid(np.dot(X, theta)) - b

    inner = np.multiply(inner1, sigm_dev)

    final_res = np.dot(a * X.T, inner)

    return lam * final_res


with open('adult.data') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    y = []
    s = []

    i = 0
    for row in csv_reader:
        if i == 0:
            i += 1
            continue

        if row[9] == "Male":
            s.append(1)
        else:
            s.append(0)

        if row[14] == '>50K':
            y.append(1)
        else:
            y.append(0)


with open('adult.test') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    testY = []
    testS = []

    i = 0
    for row in csv_reader:
        if i == 0:
            i += 1
            continue

        if row[9] == "Male":
            testS.append(1)
        else:
            testS.append(0)

        if row[14] == '>50K':
            testY.append(1)
        else:
            testY.append(0)


with open('AdultTrain.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    X = []
    i = 0
    columns = ''
    for row in csv_reader:
        if i == 0:
            i += 1
            columns = row
            continue

        new_row = []
        for item in row:
            new_row.append(float(item))

        new_row.append(1)  # intercept
        X.append(new_row)


with open('AdultTest.csv') as csv_file:
    csv_reader = csv.reader(csv_file)

    testX = []
    i = 0
    columns = ''
    for row in csv_reader:
        if i == 0:
            i += 1
            columns = row
            continue

        new_row = []
        for item in row:
            new_row.append(float(item))

        new_row.append(1)  # intercept

        testX.append(new_row)


X = normalize(X, axis=0)
testX = normalize(testX, axis=0)


w = [0, 0]
lam = 75


s = np.array(s)
testS = np.array(testS)

y = np.array(y)
testY = np.array(testY)

theta = np.zeros(X.shape[1])

num_iterations = 5000
step_size = 0.01

for i in range(num_iterations):
    if i % 100 == 0:
        print(i)
    logits = np.dot(X, theta)
    probs = sigmoid(logits)
    g1 = np.dot(X.T, (probs - y))
    g2 = grad2()

    theta -= step_size * (g1 - g2)

    w = update_w()


# Training
testOut = np.dot(X, theta)
testProbs = sigmoid(testOut)

preds = testProbs >= 0.5

print(preds)
acc = (preds == y).mean()
for item in preds:
    print(item)

print("Train:")
print(acc)

num_11 = 0
num_10 = 0
num_01 = 0
num_00 = 0

for i in range(len(s)):

    if s[i] == 1 and preds[i]:
        num_11 += 1

    elif s[i] == 1 and not preds[i]:
        num_10 += 1

    elif s[i] == 0 and preds[i]:
        num_01 += 1

    else:
        num_00 += 1


x1 = num_11 / (num_11 + num_10)
x2 = num_01 / (num_01 + num_00)
print("P(y = 1 | s = 1) = ", x1)
print("P(y = 1 | s = 0) = ", x2)
print("DI: ", x2 / x1)
# print("Renyi correlation: ", renyi_value())


print("Test:")
testOut = np.dot(testX, theta)
testProbs = sigmoid(testOut)

preds = testProbs >= 0.5

acc = (preds == testY).mean()
print(acc)

num_11 = 0
num_10 = 0
num_01 = 0
num_00 = 0

for i in range(len(testS)):

    if testS[i] == 1 and preds[i]:
        num_11 += 1

    elif testS[i] == 1 and not preds[i]:
        num_10 += 1

    elif testS[i] == 0 and preds[i]:
        num_01 += 1

    else:
        num_00 += 1

x1 = num_11 / (num_11 + num_10)
x2 = num_01 / (num_01 + num_00)
print("P(y = 1 | s = 1) = ", x1)
print("P(y = 1 | s = 0) = ", x2)
print("DI: ", x2 / x1)
