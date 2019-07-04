import math
import csv
import numpy as np
from sklearn.preprocessing import normalize

rows = []
s = []
k = 10
lambda_list = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 10, 20]


data = []
with open('bank.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        data.append([float(row[0]), float(row[1]), float(row[2])])
        s.append(int(row[3]))

data = data[0:10000]

data = normalize(data, axis=0)

data = np.array(data)
# data = normalize(data, axis=0)
s = np.array(s)

n, m = data.shape
print(n, m)

for lam in lambda_list:
    for k in range(5, 21):
        a = np.zeros((n, 1))
        w = np.zeros((k, 1))
        centers = np.zeros((k, m))
        for i in range(k):
            centers[i, :] = data[i, :]

        previous_solution = np.ones_like(a)

        while np.linalg.norm(a - previous_solution) > 0:
            previous_solution = a
            # Update assignments:
            for i in range(n):  # for any data point find the cluster
                min_score = 999999999999999999999
                min_index = -1
                for j in range(k):
                    norm = np.linalg.norm(data[i, :] - centers[j, :])
                    norm1 = norm * norm
                    # norm1 = norm

                    norm = w[j][0] - s[i]
                    norm2 = lam * norm * norm
                    # norm2 = lam * norm

                    score_j = norm1 - norm2
                    if score_j < min_score:
                        min_score = score_j
                        min_index = j
                a[i][0] = min_index

                # Update the weights
                # Updating w:
                for t in range(k):
                    ith_cluster_size = 0
                    ith_cluster_group1 = 0
                    for j in range(n):
                        if a[j][0] == t:
                            ith_cluster_size += 1
                        if a[j][0] == t and s[j] == 1:
                            ith_cluster_group1 += 1
                    if ith_cluster_size != 0:
                        w[t][0] = float(ith_cluster_group1) / ith_cluster_size

            # Update Centers
            centers = np.random.rand(k, m)
            for i in range(k):
                ith_cluster_size = 0
                for j in range(n):
                    if a[j][0] == i:
                        ith_cluster_size += 1
                        centers[i, :] += data[j, :]
                if ith_cluster_size != 0:
                    centers[i, :] /= ith_cluster_size

        print('################w##########################')
        print(k)
        print("Lambda: ", lam)
        for item in w:
            for item2 in item:
                print(item2)

        loss = 0
        num = 0
        for i in range(n):
            assigned_cluster = a[i][0]
            if assigned_cluster > -1:
                norm = np.linalg.norm(centers[int(assigned_cluster), :] - data[i, :])
                num += 1
                loss += norm
        print(loss)
