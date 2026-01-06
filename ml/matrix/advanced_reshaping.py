# Create a 3D array
arr_3d = np.arange(24).reshape(2, 3, 4)
print("Original 3D array shape:", arr_3d.shape)
print(arr_3d)

# Transpose axes
transposed = np.transpose(arr_3d, (1, 0, 2))
print("\nTransposed axes (1,0,2) shape:", transposed.shape)
print(transposed)

# Move an axis to a different position
moved = np.moveaxis(arr_3d, 0, -1)  # Move first axis to last position
print("\nMoved axis shape:", moved.shape)  # Now (3, 4, 2)
