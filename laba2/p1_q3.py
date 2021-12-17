import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

path = "/Users/ilailenko/Desktop/laba2/housing.csv"
df = pd.read_csv(path ,sep = ',')
one_hot = pd.get_dummies(df['ocean_proximity'])
df = df.drop('ocean_proximity',axis = 1)
df = df.join(one_hot)

def bedrooms(row):
    row['total_bedrooms']= row['total_bedrooms']/row['households']
    return row

df = df.apply(bedrooms,axis=1)

def rooms(row):
    row['total_rooms']= row['total_rooms']/row['households']
    return row

df = df.apply(rooms,axis=1)
df = df.rename({'total_bedrooms': 'average_bedrooms', 'total_rooms': 'average_rooms'}, axis=1) 

df_train_val, df_test = train_test_split(df, test_size=0.1)

df_train, df_val = train_test_split(df_train_val, test_size = 0.33)

def metric(df, column):
    mn = df[column].mean()
    st = df[column].std()
    return [mn,st]
    
def row_correct(row, mn,st, column):
    if row.isnull().sum()!=0:
        row[column] = np.random.normal(mn,st)
    return row

def fill_gaps(df, column):
    mn, st = metric(df, column)
    df = df.apply(lambda x: row_correct(x,mn,st, column), axis=1)
    return df
    
column = 'average_bedrooms'
df_val = fill_gaps(df_val, column)
df_train = fill_gaps(df_train, column)
df_test = fill_gaps(df_test, column)

print(df_test['average_bedrooms'].isnull().sum()) 
print(df_val['average_bedrooms'].isnull().sum()) 
print(df_train['average_bedrooms'].isnull().sum())

def rowNnorm(row, mn, st, column):
    row[column] = (row[column]-mn)/st
    return row

def dfNorm(df, columns):    
    for column in columns:
        mn,st = metric(df, column)
        df = df.apply(lambda x: rowNnorm(x,mn,st, column), axis=1)
    return df
    
df_test = dfNorm(df_test,['longitude', 'latitude'])
df_val = dfNorm(df_val,['longitude', 'latitude'])
df_train = dfNorm(df_train,['longitude', 'latitude'])

def print_df(df, cols):
    for col in cols:
        print('Среднее ', col, ':',  df[col].mean())
        print('Дисперсия ', col, ':',df[col].std())
        
cols = ['longitude', 'latitude']

print_df(df_test, cols)
print_df(df_val, cols)
print_df(df_train, cols)