# Matrix multiplication using einsum
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)

# Standard matrix multiplication
C1 = A.dot(B)

# Same operation with einsum
C2 = np.einsum('ij,jk->ik', A, B)

print("Matrix multiplication difference:", np.allclose(C1, C2))

# Batch matrix multiplication
batch_A = np.random.rand(10, 3, 4)  # 10 matrices of shape (3,4)
batch_B = np.random.rand(10, 4, 5)  # 10 matrices of shape (4,5)

# Batched matrix multiplication with einsum
result = np.einsum('bij,bjk->bik', batch_A, batch_B)
print("\nBatch matrix multiplication shape:", result.shape)  # (10, 3, 5)
