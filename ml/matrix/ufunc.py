# Creating structured arrays
data_type = [('name', 'U10'), ('height', float), ('age', int)]
people = np.array([
    ('Alice', 1.72, 32),
    ('Bob', 1.85, 45),
    ('Charlie', 1.78, 28)
], dtype=data_type)

print("Structured array:")
print(people)
print("\nAccessing fields:")
print(people['name'])
print(people['height'])

# Sort by height
sorted_by_height = np.sort(people, order='height')
print("\nSorted by height:")
print(sorted_by_height)

# Custom ufunc using frompyfunc
def logarithm_base_2(x):
    return np.log(x) / np.log(2)

log2_ufunc = np.frompyfunc(logarithm_base_2, 1, 1)
arr = np.array([1, 2, 4, 8, 16, 32])
print("\nCustom ufunc (log base 2):")
print(log2_ufunc(arr))
