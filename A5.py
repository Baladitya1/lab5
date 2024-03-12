# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier  
import numpy as np 
import matplotlib.pyplot as plt  

# Set random seed for reproducibility
np.random.seed(0)

# Generate random data points X and class labels
X = np.random.randint(1, 11, size=(20, 2))  # Data points
classes = np.random.randint(0, 2, size=20)  # Class labels

# Split data into two classes
class0_X = X[classes == 0]  # Class 0 data points
class1_X = X[classes == 1]  # Class 1 data points

# Generate a grid of x and y values for test data
x_values = np.arange(0, 10.1, 0.1)  # X values
y_values = np.arange(0, 10.1, 0.1)  # Y values
xx, yy = np.meshgrid(x_values, y_values)  # Grid for test data
test_data = np.column_stack((xx.ravel(), yy.ravel()))  # Test data points

# Define k values to experiment with
k_values = [1, 3, 5, 7]

# Create a figure to contain subplots
plt.figure(figsize=(15, 10))

# Iterate over each k value
for i, k in enumerate(k_values, 1):
    # Instantiate KNN classifier with current k value
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    
    # Fit classifier on training data
    knn_classifier.fit(X, classes)
    
    # Predict classes for test data
    predicted_classes = knn_classifier.predict(test_data)

    # Create a subplot for the current k value
    plt.subplot(2, 2, i)
    
    # Scatter plot for test data points with predicted classes
    plt.scatter(test_data[:, 0], test_data[:, 1], c=predicted_classes, cmap=plt.cm.coolwarm, alpha=0.1)
    
    # Scatter plot for class 0 and class 1 data points
    plt.scatter(class0_X[:, 0], class0_X[:, 1], color='blue', label='Class 0')
    plt.scatter(class1_X[:, 0], class1_X[:, 1], color='red', label='Class 1')
    
    # Set plot title, axis labels, legend, and grid
    plt.title('k = {}'.format(k))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()
