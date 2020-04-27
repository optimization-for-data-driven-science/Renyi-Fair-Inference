import matplotlib.pyplot as plt

plt.style.use('bmh')

lambda_train = [0, 4, 5, 6, 8, 20, 22, 30, 40, 60, 160, 600]

lambdas = [0, 4, 5, 6, 8, 20, 22, 30, 40]

train_acc = [85.20, 85.19, 85.18, 85.15, 85.14, 85.12, 85.12, 85.11, 85.09, 85.08, 85.07, 85.06]
test_acc = [85.27, 85.12, 85.13, 85.16, 85.12, 85.08, 85.05, 85.04, 85.03]

train_eo = [0.085, 0.061, 0.057, 0.054, 0.044, 0.030, 0.028, 0.025, 0.025, 0.020, 0.006, 0.0011]
test_eo = [0.065, 0.038, 0.034, 0.031, 0.025, 0.014, 0.011, 0.003, 0.0009]

train_accuracy = []
for item in train_acc:
    train_accuracy.append(100.00 - item)

test_accuracy = []
for item in test_acc:
    test_accuracy.append(100.00 - item)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# ax1.plot(lambda_train, train_eo, color='blue')
# ax2.plot(lambda_train, train_accuracy, color='r')

ax1.plot(lambdas, test_eo, color='blue')
ax2.plot(lambdas, test_accuracy, color='r')

plt.rc('xtick', labelsize=50)

plt.show()

