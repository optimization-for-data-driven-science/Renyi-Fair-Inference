import csv
import numpy as np
from sklearn.preprocessing import normalize

rows = []
s1 = []


data = []
with open('bank.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        data.append([float(row[0]), float(row[1]), float(row[2])])
        s1.append(int(row[3]))

data = normalize(data, axis=0)

data = np.array(data)
# data = normalize(data, axis=0)

k = 15
n, m = data.shape
print(n, m)

s = np.zeros(shape=(n, 1))

i = 0
for item in s1:
    s[i][0] = item
    i += 1

lam = 0.02
rho = 0.1
number_of_iterations = 100

A = np.zeros(shape=(k, n))
C = np.zeros(shape=(k, m))
mu = np.zeros(shape=(n, 1))
w = np.zeros(shape=(k, 1))
ones_n = np.ones(shape=(n, 1))
identity = np.identity(n)


for current_iteration in range(number_of_iterations):
    # Updating Assignments:
    for data_ind in range(n):
        distances = []

        min_dist = -1
        min_ind = 0

        for k_ind in range(k):
            distance = np.linalg.norm(data[data_ind] - C[k_ind])
            distance = distance * distance
            distance += w[k_ind][0] * mu[data_ind][0]

            for counter in range(data_ind):
                distance += mu[data_ind][0] * mu[counter][0] * A[k_ind][counter]

            for counter in range(data_ind + 1, n):
                distance += mu[data_ind][0] + mu[counter][0] * A[k_ind][counter]

            distance += mu[data_ind][0] * mu[data_ind][0]

            if distance < min_dist:
                min_dist = distance
                min_ind = k_ind

        for count in range(k):
            if count == min_ind:
                A[count][data_ind] = 1
            else:
                A[count][data_ind] = 0

    # Updating \mu
    first = lam * identity + rho * np.dot(A.T, A)
    second = np.subtract(lam * s, np.dot(A.T, w))
    mu = np.dot(np.linalg.inv(first), second)

    # Updating centers:
    C = np.dot(A, data)
    den = np.dot(A, ones_n)
    for counter in range(k):
        dev = np.dot(A[counter], ones_n)[0]
        if dev != 0:
            C[counter] /= dev

    # Updating w:
    w = w + rho * np.dot(A, mu)

    print(current_iteration)
    loss = 0
    for i in range(k):
        for j in range(n):
            if A[i][j] != 0:
                loss += np.linalg.norm(C[i] - data[j])

    print(loss)

