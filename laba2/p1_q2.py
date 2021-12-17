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
orders = orders.assign(OrderSum = lambda x:'')
orders['OrderSum'][0] = 'OrderSum' 
for i in range(1, len(orders['OrderSum'])):
    orders['OrderSum'][i] = (float(orders['UnitPrice'][i]) * float(orders['Quantity'][i]) * (1 - float(orders['Discount'][i])))

Category = {}
for i in range(1, len(products['CategoryName'])):
    Category[products['CategoryName'][i]] = 0
for i in range(1, len(products['CategoryName'])):
    Category[products['CategoryName'][i]] =  Category[products['CategoryName'][i]] + 1

Sum_Id = {}
for i in range(1, len(orders['ProductID'])):
    Sum_Id[ orders['ProductID'][i]] =  0
for i in range(1, len(orders['ProductID'])):
    Sum_Id[orders['ProductID'][i]] =  float(Sum_Id[orders['ProductID'][i]]) + float(orders['OrderSum'][i])

sum = {}
for i in range(1, len(products['CategoryName'])):
    sum[ products['CategoryName'][i]] = 0
for i in range(1, len(products['CategoryName'])):
    sum[products['CategoryName'][i]] = sum[products['CategoryName'][i]] + Sum_Id[products['ProductID'][i]]

Category_average_value = {k: sum[k] / Category[k] for k in Category if k in sum}
print("Средний доход продаж какой-либо категории: ")
print(Category_average_value)

#Пункт 2
Purchase_price_Id = {}
for i in range(1, len(products['ProductID'])):
    Purchase_price_Id[ products['ProductID'][i]] =  0
for i in range(1, len(products['ProductID'])):
    Purchase_price_Id[products['ProductID'][i]] = float(products['UnitCost'][i])

Purchase_price_Sum = {}
for i in range(1, len(orders['ProductID'])):
    Purchase_price_Sum[ orders['ProductID'][i]] =  0
for i in range(1, len(orders['ProductID'])):
    Purchase_price_Sum[orders['ProductID'][i]] = Purchase_price_Sum[orders['ProductID'][i]] + (Purchase_price_Id[orders['ProductID'][i]] * float(orders['Quantity'][i]))

Profit_values = {k: Sum_Id[k] - Purchase_price_Sum[k] for k in Purchase_price_Sum if k in Sum_Id}

products = products.assign(Profit = lambda x:'')
products['Profit'][0] = 'Profit' 
for i in range(1, len(products['Profit'])):
    products['Profit'][i] = Profit_values[products['ProductID'][i]]
print(products)

#Пункт 3
# за два года
Sum_Id_of_two_years = {}
price_Sum_of_two_years = {}
for i in range(1, len(orders['ProductID'])):
    Sum_Id_of_two_years[ orders['ProductID'][i]] =  0
    price_Sum_of_two_years[ orders['ProductID'][i]] =  0
for i in range(1, len(orders['ProductID'])):
    OrderDate = orders['OrderDate'][i]
    if(OrderDate[:4] == '2005' or OrderDate[:4] == '2006'):
        Sum_Id_of_two_years[orders['ProductID'][i]] =  float(Sum_Id_of_two_years[orders['ProductID'][i]]) + float(orders['OrderSum'][i])
        price_Sum_of_two_years[orders['ProductID'][i]] = price_Sum_of_two_years[orders['ProductID'][i]] + (Purchase_price_Id[orders['ProductID'][i]] * float(orders['Quantity'][i]))

Profit_values_of_two_years = {k: Sum_Id_of_two_years[k] - price_Sum_of_two_years[k] for k in price_Sum_of_two_years if k in Sum_Id_of_two_years}

Category_Profit_of_two_years = {}
for i in range(1, len(products['CategoryName'])):
    Category_Profit_of_two_years[ products['CategoryName'][i]] = 0
for i in range(1, len(products['CategoryName'])):
    Category_Profit_of_two_years[products['CategoryName'][i]] = Category_Profit_of_two_years[products['CategoryName'][i]] + Profit_values_of_two_years[products['ProductID'][i]]

sorted_Category_Profit_2005_2006 = sorted(Category_Profit_of_two_years.values())

sum_Profit_values_2005_2006 = 0
for i in range(1, len(Profit_values_of_two_years)):
    sum_Profit_values_2005_2006 = sum_Profit_values_2005_2006 + Profit_values_of_two_years[products['ProductID'][i]]

sorted_sum = 0
print('самая большая прибыль товаров категорий за 2005-2006 года, которая состовляет 80% общей прибыли за этот период')
for i in range(len(sorted_Category_Profit_2005_2006)-1, len(sorted_Category_Profit_2005_2006)-9, -1):
    print(f"{'Категория: ' + list(Category_Profit_of_two_years.keys())[list(Category_Profit_of_two_years.values()).index(sorted_Category_Profit_2005_2006[i])]:<35} Прибыль за категорию: {sorted_Category_Profit_2005_2006[i]}")
    sorted_sum += sorted_Category_Profit_2005_2006[i]
    if(sorted_sum >= sum_Profit_values_2005_2006*0.8):
        break
#за всё время
Sum_Id = {}
price_Sum = {}
for i in range(1, len(orders['ProductID'])):
    Sum_Id[ orders['ProductID'][i]] =  0
    price_Sum[ orders['ProductID'][i]] =  0
for i in range(1, len(orders['ProductID'])):
    OrderDate = orders['OrderDate'][i]
    Sum_Id[orders['ProductID'][i]] =  float(Sum_Id[orders['ProductID'][i]]) + float(orders['OrderSum'][i])
    price_Sum[orders['ProductID'][i]] = price_Sum[orders['ProductID'][i]] + (Purchase_price_Id[orders['ProductID'][i]] * float(orders['Quantity'][i]))

Profit_values = {k: Sum_Id[k] - price_Sum[k] for k in price_Sum if k in Sum_Id}

Category_Profit = {}
for i in range(1, len(products['CategoryName'])):
    Category_Profit[ products['CategoryName'][i]] = 0
for i in range(1, len(products['CategoryName'])):
    Category_Profit[products['CategoryName'][i]] = Category_Profit[products['CategoryName'][i]] + Profit_values[products['ProductID'][i]]

sorted_Category_Profit = sorted(Category_Profit.values())

sum_Profit_values = 0
for i in range(1, len(Profit_values)):
    sum_Profit_values = sum_Profit_values + Profit_values[products['ProductID'][i]]

sorted_sum = 0
print('самая большая прибыль товаров категорий за все года, которая состовляет 80% общей прибыли за этот период')
for i in range(len(sorted_Category_Profit)-1, len(sorted_Category_Profit)-9, -1):
    print(f"{'Категория: ' + list(Category_Profit.keys())[list(Category_Profit.values()).index(sorted_Category_Profit[i])]:<35} Прибыль за категорию: {sorted_Category_Profit[i]}")
    sorted_sum += sorted_Category_Profit[i]
    if(sorted_sum >= sum_Profit_values*0.8):
        break    