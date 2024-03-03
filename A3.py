import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.randint(1, 11, size=(20, 2))
classes = np.random.randint(0, 2, size=20)
class0_X = X[classes == 0]
class1_X = X[classes == 1]

plt.figure(figsize=(8, 6))
plt.scatter(class0_X[:, 0], class0_X[:, 1], color='blue', label='Class 0')
plt.scatter(class1_X[:, 0], class1_X[:, 1], color='red', label='Class 1')

for i, (x, y) in enumerate(X):
    plt.text(x, y, f'({x},{y})', fontsize=8, ha='center', va='center')

plt.title('Scatter Plot of Training Data with Labels')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
