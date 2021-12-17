import pandas as pd
import numpy as np


i=0
df1 = pd.read_json("/Users/ilailenko/Desktop/Лазаревский/sales.json")
df2 = pd.DataFrame(columns=['item', 'country', 'year', 'sales'])

for item in range(0, len(df1['item'].keys())):
    for country in df1['sales_by_country'][item].keys():
        for year  in df1['sales_by_country'][item].get(f'{country}').keys():
            df2.loc[i] = [df1['item'][item],country,year,df1['sales_by_country'][item].get(f'{country}').get(f'{year}')]
            i+=1      
df2.to_csv('CSV_df2.csv',sep=',',columns=['item', 'country', 'year', 'sales'])
print(df2)