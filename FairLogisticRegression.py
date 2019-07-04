import numpy as np
from sklearn.preprocessing import normalize
import csv
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


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

    grad = 2 * a * X.T
    # total_sum = 0

    sigm = sigmoid(np.dot(X, theta))
    sigm2 = np.ones_like(sigm) - sigm

    sigm_dev = np.multiply(sigm, sigm2)

    inner1 = a * sigmoid(np.dot(X, theta)) - b

    inner = np.multiply(inner1, sigm_dev)

    final_res = np.dot(a * X.T, inner)

    return lam * final_res


with open('Bank2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    y = []
    s = []

    i = 0
    for row in csv_reader:
        if i == 0:
            i += 1
            continue

        if row[2] == "married":
            s.append(1)
        else:
            s.append(0)

        if row[16] == 'yes':
            y.append(1)
        else:
            y.append(0)

w = [0, 0]
lam = 1

# Read the bank file
data = []

with open('Bank_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file)

    X = []
    i = 0
    columns = ''
    for row in csv_reader:
        if i == 0:
            i += 1
            columns = row
            continue

        # print(row)
        # new_row = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[5])]
        new_row = []
        for item in row:
            new_row.append(float(item))

        new_row.append(1)  # intercept

        X.append(new_row)

X = normalize(X, axis=0)

testX = np.array(X[32000:])
testY = y[32000:]
testS = s[32000:]

mean = sum(testS) / len(testS)
centered = []

for val in s:
    centered.append(val - mean)

testS = testS / np.linalg.norm(testS)

X = np.array(X[0:32000])
y = y[:32000]
s = s[:32000]

mean = sum(s) / len(s)
centered = []

for val in s:
    centered.append(val - mean)

s = centered / np.linalg.norm(centered)

print(X.shape)
print(testX.shape)

"""
# SKlearn
log_reg = LogisticRegression()
log_reg.fit(X, y)

predictions = log_reg.predict(testX)
accuracy = metrics.accuracy_score(testY, predictions)
print(accuracy)
exit(0)
"""

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

# x1 = num_11 / (num_11 + num_10)
# x2 = num_01 / (num_01 + num_00)
# print("P(y = 1 | s = 1) = ", x1)
# print("P(y = 1 | s = 0) = ", x2)
# print("DI: ", x2 / x1)
print("Renyi correlation: ", renyi_value())

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
