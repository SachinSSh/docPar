# Creating a 2D array
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Accessing elements
element = matrix[1][2]  # Accesses row 1, column 2: 6

# Modifying elements
matrix[0][0] = 10

# Creating a matrix with list comprehension
rows, cols = 3, 4
matrix = [[0 for j in range(cols)] for i in range(rows)]
