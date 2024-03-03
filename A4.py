from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.randint(1, 11, size=(20, 2))
classes = np.random.randint(0, 2, size=20)
class0_X = X[classes == 0]
class1_X = X[classes == 1]

x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_values, y_values)
test_data = np.column_stack((xx.ravel(), yy.ravel()))

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X, classes)
predicted_classes = knn_classifier.predict(test_data)

plt.figure(figsize=(10, 8))
plt.scatter(test_data[:, 0], test_data[:, 1], c=predicted_classes, cmap=plt.cm.coolwarm, alpha=0.1)
plt.scatter(class0_X[:, 0], class0_X[:, 1], color='blue', label='Class 0')
plt.scatter(class1_X[:, 0], class1_X[:, 1], color='red', label='Class 1')
plt.title('Scatter Plot of Test Data Output')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()