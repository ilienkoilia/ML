import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


path_orders = "/Users/ilailenko/Desktop/laba2/orders.csv"
path_products = "/Users/ilailenko/Desktop/laba2/products.csv"
colnames_orders=['OrderID', 'OrderDate', 'ProductID', 'UnitPrice', 'Quantity', 'Discount']
colnames_products=['ProductID', 'ProductName', 'QuantityPerUnit', 'UnitCost', 'UnitPrice', 'CategoryName']
orders = pd.read_csv(path_orders, names=colnames_orders, header=None)
products = pd.read_csv(path_products, names=colnames_products, header=None)

#Пункт 1
Category = {}
CategoryName = products['CategoryName'][1]
Category[CategoryName] = '0'


for i in range(1, len(products['CategoryName'])):
    if(products['CategoryName'][i] == CategoryName):
        Category[CategoryName] = str(int(Category[CategoryName]) +   1)
    else:
        Category[products['CategoryName'][i]] = '1'
        CategoryName = products['CategoryName'][i]
print("кол-во уникальных продуктов для каждой категории")
print(Category)

#Пункт 2
NameSeafood = []
for i in range(1, len(products['CategoryName'])):
    if(products['CategoryName'][i] == 'Морепродукты'):
        NameSeafood.append(products['ProductName'][i])
print("Все продукты в категории Морепродукты")        
print(NameSeafood)

#Пункт 3
Date = {}
for i in range(1, len(orders['OrderDate'])):
    OrderDate = orders['OrderDate'][i]
    Date[OrderDate[:7]] =  '0'
for i in range(1, len(orders['OrderDate'])):
    OrderDate = orders['OrderDate'][i]
    Date[OrderDate[:7]] = str(int(Date[OrderDate[:7]]) + int(orders['Quantity'][i]))

list_datakeys = list(Date.keys())
list_datakeys.sort()
dates = []
values_dates = []
for i in list_datakeys:
    dates.append(i)
    values_dates.append(int(Date[i]))

plt.figure(figsize=(20,10))
a = range(len(Date))
plt.plot(a, values_dates)
plt.xticks(a, dates, fontsize=3.5)
plt.show()

#Пункт 4
orders = orders.assign(OrderSum = lambda x:'') 
for i in range(1, len(orders['OrderSum'])):
    orders['OrderSum'][i] = (float(orders['UnitPrice'][i]) * float(orders['Quantity'][i]) * (1 - float(orders['Discount'][i])))
print(orders)
Sum_orders = {}

for i in range(1, len(orders['OrderID'])):
    Sum_orders[ orders['OrderID'][i]] =  '0'

for i in range(1, len(orders['OrderID'])):
    Sum_orders[orders['OrderID'][i]] = float(Sum_orders[orders['OrderID'][i]]) + float(orders['OrderSum'][i])
sorted_orders = sorted(Sum_orders.values())

print("десять самых дорогих заказов")
for i in range(len(sorted_orders)-1, len(sorted_orders)-9, -1):
    print("OrderID: ",list(Sum_orders.keys())[list(Sum_orders.values()).index(sorted_orders[i])] , "Сумма заказа: ", sorted_orders[i])

#Пункт 5
Product_Cost = {}
for i in range(1, len(products['ProductName'])):
    Product_Cost[ products['ProductName'][i]] =  '0'

for i in range(1, len(products['ProductName'])):
    Product_Cost[products['ProductName'][i]] = float(Product_Cost[products['ProductName'][i]]) + (float(products['UnitPrice'][i]) / float(products['QuantityPerUnit'][i]))
sorted_product = sorted(Product_Cost.values())
print("десять самых дорогих продуктов за штуку")

for i in range(len(sorted_product)-1, len(sorted_product)-9, -1):
    print(f"{'Продукт: ' + list(Product_Cost.keys())[list(Product_Cost.values()).index(sorted_product[i])]:<25} Стоимость продукта: {sorted_product[i]}")
    