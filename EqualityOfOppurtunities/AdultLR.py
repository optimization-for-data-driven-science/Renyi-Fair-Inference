import numpy as np
from sklearn.preprocessing import normalize
import csv


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss1(predictions, labels):
    return (-labels * np.log(predictions) - (1 - labels) * np.log(1 - predictions)).mean()


def update_w():

    X1 = X[0:counter_1]
    s1 = s[0:counter_1]

    col1 = sigmoid((np.dot(X1, theta)))
    temp = np.ones_like(col1)
    col2 = temp - col1

    yhatT = np.concatenate((col1, col2)).reshape((2, -1))
    yhat = yhatT.T

    inv = np.linalg.inv(np.dot(yhatT, yhat))
    temp2 = np.dot(inv, yhatT)

    res = np.dot(temp2, s1)

    return res


def renyi_value():
    return np.linalg.norm(w * np.dot(X, theta) - s)


def grad2():

    s1 = s[0:counter_1]
    X1 = X[0:counter_1]

    a = w[0] - w[1]
    b = s1 - w[1] * np.ones_like(s1)

    sigm = sigmoid(np.dot(X1, theta))
    sigm2 = np.ones_like(sigm) - sigm

    sigm_dev = np.multiply(sigm, sigm2)

    inner1 = a * sigmoid(np.dot(X1, theta)) - b

    inner = np.multiply(inner1, sigm_dev)

    final_res = np.dot(a * X1.T, inner)

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

        if row[14] == '>50K.':
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

orderedX = []
orderedS = []
orderedY = []

counter_1 = 0

for i in range(len(y)):
    if y[i] == 1:
        orderedY.append(y[i])
        orderedS.append(s[i])
        orderedX.append(X[i])
        counter_1 += 1

for i in range(len(y)):
    if y[i] == 0:
        orderedY.append(y[i])
        orderedS.append(s[i])
        orderedX.append(X[i])

X = orderedX
s = orderedS
y = orderedY

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

s = np.array(s)
testS = np.array(testS)

y = np.array(y)
testY = np.array(testY)

num_iterations = 10000
step_size = 0.05

for l in [0, 1, 10, 100, 400, 500, 1000]:
    # A largest lam corresponds to a fairer solution (EO violation would be smaller)
    print("Lambda = ", l)
    lam = l
    w = [0, 0]
    theta = np.zeros(X.shape[1])

    for i in range(num_iterations):
        if i % 1000 == 0:
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

    acc = (preds == y).mean()

    print("Lambda: ", lam)
    print("Train:")
    print("Accuracy: ", acc)

    num_11 = 0
    num_10 = 0
    num_01 = 0
    num_00 = 0

    for i in range(len(s)):
        if y[i] == 0:
            continue

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
    print("Equality of Opportunity violation: ", max(x2 - x1, x1 - x2))
    # print("Renyi correlation: ", renyi_value())

    print("Test:")
    testOut = np.dot(testX, theta)
    testProbs = sigmoid(testOut)

    preds = testProbs >= 0.5

    acc = (preds == testY).mean()
    print("Accuracy: ", acc)

    num_11 = 0
    num_10 = 0
    num_01 = 0
    num_00 = 0

    for i in range(len(testS)):
        if testY[i] == 0:
            continue

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
    print("Equality of Opportunity violation: ", max(x2 - x1, x1 - x2))
    print("-------------------------------")
