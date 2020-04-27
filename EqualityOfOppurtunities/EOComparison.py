import matplotlib.pyplot as plt

# bmh
plt.style.use('bmh')

plt.scatter(0.01, 20.5, marker="o", s=200)  # Hardt

plt.scatter(0.01, 15.1, marker="P", s=200)  # Log loss

plt.scatter(0.01, 14.95, marker='*', s=200)  # Renyi

plt.scatter(0.01, 19, marker='X', s=200)  # ERM

plt.scatter(0.05, 18, marker='D', s=200)  # Zafar

plt.show()
