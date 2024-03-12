
from sklearn.neighbors import KNeighborsClassifier  
import numpy as np  
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

X = np.random.randint(1, 11, size=(20, 2))  
classes = np.random.randint(0, 2, size=20)  

# Split data into two classes
class0_X = X[classes == 0]  
class1_X = X[classes == 1] 

# Generate a grid of x and y values for test data
x_values = np.arange(0, 10.1, 0.1)  # X values
y_values = np.arange(0, 10.1, 0.1)  # Y values
xx, yy = np.meshgrid(x_values, y_values)  # Grid for test data
test_data = np.column_stack((xx.ravel(), yy.ravel()))  # Test data points

# Instantiate KNN classifier with k=3
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Fit classifier on training data
knn_classifier.fit(X, classes)

# Predict classes for test data
predicted_classes = knn_classifier.predict(test_data)

# Scatter plot for test data points with predicted classes
plt.figure(figsize=(10, 8))
plt.scatter(test_data[:, 0], test_data[:, 1], c=predicted_classes, cmap=plt.cm.coolwarm, alpha=0.1)

# Scatter plot for class 0 and class 1 data points
plt.scatter(class0_X[:, 0], class0_X[:, 1], color='blue', label='Class 0')
plt.scatter(class1_X[:, 0], class1_X[:, 1], color='red', label='Class 1')

# Set plot title and axis labels
plt.title('Scatter Plot of Test Data Output')
plt.xlabel('X')
plt.ylabel('Y')

# Add legend and grid to the plot
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
