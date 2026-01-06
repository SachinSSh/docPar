## given a 3d array representing monthly sales data (axis:0, stores, axis 1:months, axis2: products), calculate the total sales for each product across all stores for each month

import numpy as np

# Let's say we have 3 stores, 12 months, and 4 products
# Shape: (stores, months, products) = (3, 12, 4)
sales_data = np.random.randint(100, 1000, size=(3, 12, 4))

# Sum across all stores (axis 0)
total_sales_by_month_product = np.sum(sales_data, axis=0)

# Result shape will be (months, products)
# Each row represents a month, each column represents a product
print("Shape of result:", total_sales_by_month_product.shape)

print("Total sales for each product in January:")
print(total_sales_by_month_product[0])

# Example: Displaying total sales for Product 1 across all months
print("Total sales for Product 1 across all months:")
print(total_sales_by_month_product[:, 0])

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
products = ['Product 1', 'Product 2', 'Product 3', 'Product 4']

print("\nTotal sales for each product by month:")
print("Month", end="\t")
for product in products:
    print(f"{product}", end="\t")
print()

for i, month in enumerate(months):
    print(f"{month}", end="\t")
    for j in range(len(products)):
        print(f"{total_sales_by_month_product[i, j]}", end="\t")
    print()
