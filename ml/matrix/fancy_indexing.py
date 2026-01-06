# Fancy indexing with integer arrays
arr = np.arange(36).reshape(6, 6)
indices = np.array([0, 2, 4])
print("Selected rows using integer indices:")
print(arr[indices])  # Selects rows 0, 2, and 4

# Selecting specific elements using coordinate arrays
x_coords = np.array([0, 3, 5])
y_coords = np.array([1, 2, 0])
print("\nSelected elements using coordinate pairs:")
print(arr[x_coords, y_coords])  # Gets elements at (0,1), (3,2), (5,0)

# Boolean masking
mask = arr > 20
print("\nBoolean mask:")
print(mask)
print("\nElements where value > 20:")
print(arr[mask])  # Returns all elements > 20 as a flat array
