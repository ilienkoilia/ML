import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

path = "/Users/ilailenko/Desktop/laba2/housing.csv"
df = pd.read_csv(path)

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

encoder = preprocessing.OneHotEncoder(handle_unknown="ignore")

ohe_ocean_train = encoder.fit_transform(df.ocean_proximity)
df_train = df_train.drop('ocean_proximity', axis=1).join(ohe_ocean_train)

ohe_ocean_val = encoder.transform(df_val.ocean_proxomity)
df_val = df_val.drop('ocean_proximity', axis=1).join(ohe_ocean_val)

ohe_ocean_test = encoder.transform(df_test.ocean_proxomity)
df_test = df_test.drop('ocean_proximity', axis=1).join(ohe_ocean_test)


def metric(df, column):
    mn = df[column].mean()
    st = df[column].std()
    return [mn, st]
    
def row_correct(row, mn,st, column):
    if row.isnull().sum()!=0:
        row[column] = np.random.normal(mn, st)
    return row

def fill_gaps(df, column, mn, st):
    return df.apply(lambda x: row_correct(x, mn, st, column), axis=1)


print(df_test['average_bedrooms'].isnull().sum()) 
print(df_val['average_bedrooms'].isnull().sum()) 
print(df_train['average_bedrooms'].isnull().sum())

column = 'average_bedrooms'
mn, st = metric(df_train, column)

df_train = fill_gaps(df_train, column, mn, st)
df_val = fill_gaps(df_val, column, mn, st)
df_test = fill_gaps(df_test, column, mn, st)

for column in columns:
    mn,st = metric(df_train, column)
    df_train[column] = (df_train[column] - mn) / std
    df_val[column] = (df_val[column] - mn) / std
    df_test[column] = (df_test[column] - mn) / std


def print_df(df, cols):
    for col in cols:
        print('Среднее ', col, ':',  df[col].mean())
        print('Дисперсия ', col, ':',df[col].std())
        
cols = ['longitude', 'latitude']

print_df(df_test, cols)
print_df(df_val, cols)
print_df(df_train, cols)