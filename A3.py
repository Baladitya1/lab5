import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Generate random data points
X = np.random.randint(1, 11, size=(20, 2))

# Generate random class labels (0 or 1) for each data point
classes = np.random.randint(0, 2, size=20)

# Split data into two classes based on class labels
class0_X = X[classes == 0]
class1_X = X[classes == 1]

# Create a scatter plot for Class 0 data points
plt.scatter(class0_X[:, 0], class0_X[:, 1], color='blue', label='Class 0')

# Create a scatter plot for Class 1 data points
plt.scatter(class1_X[:, 0], class1_X[:, 1], color='red', label='Class 1')

# Add text labels for each data point showing their coordinates
for i, (x, y) in enumerate(X):
    plt.text(x, y, f'({x},{y})', fontsize=8, ha='center', va='center')

# Set plot title and axis labels
plt.title('Scatter Plot of Training Data with Labels')
plt.xlabel('X')
plt.ylabel('Y')

# Add legend and grid to the plot
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
