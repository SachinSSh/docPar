import numpy as np

# Creating a NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Multi-dimensional arrays
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
arr + 1  # Element-wise addition: [2, 3, 4, 5, 6]
arr * 2  # Element-wise multiplication: [2, 4, 6, 8, 10]

# Array functions
arr.sum()  # Sum of all elements
arr.mean()  # Mean of all elements
arr.max()  # Maximum value
arr.min()  # Minimum value

# Reshaping
arr.reshape(5, 1)  # Creates 5x1 matrix
