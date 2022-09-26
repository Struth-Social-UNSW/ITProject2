from cProfile import label
import pandas as pd
import numpy as np


###  Read csv file into DataFrame df  ###
def writeCSV():

    newDataset = 'general-WELFake.csv'

    col_names = ['text', 'label']
    #col_names = ['text', 'label']
    #df = pd.read_csv('combo-fake-true.csv', skiprows = 1, names = col_names, usecols=[0,2])

    # Read from large WELFake dataset
    df = pd.read_csv('WELFake_Dataset.csv', skiprows = 1, names = col_names, usecols=[2,3])

    # Read from kaggle covid dataset
    #df = pd.read_csv('kaggle-covid-news.csv', skiprows = 1, names = col_names, usecols=[1,2])

    # Ensure label column is either 'real' or 'fake'
    df['label'].mask(df['label'] == 1, 'real', inplace=True)
    df['label'].mask(df['label'] == 0, 'fake', inplace=True)
    df['label'].mask(df['label'] == 'T', 'real', inplace=True)
    df['label'].mask(df['label'] == 'F', 'fake', inplace=True)
    df['label'].mask(df['label'] == 'REAL', 'real', inplace=True)
    df['label'].mask(df['label'] == 'TRUE', 'real', inplace=True)
    df['label'].mask(df['label'] == 'FAKE', 'fake', inplace=True)
    df['label'].mask(df['label'] == 'Real', 'real', inplace=True)
    df['label'].mask(df['label'] == 'True', 'real', inplace=True)
    df['label'].mask(df['label'] == 'Fake', 'fake', inplace=True)
    
    # Read from kaggle covid dataset
    df2 = pd.read_csv('kaggle-covid-news.csv', skiprows = 1, names = col_names, usecols=[1,2])

    # Ensure label column is either 'real' or 'fake'
    df2['label'].mask(df2['label'] == 1, 'real', inplace=True)
    df2['label'].mask(df2['label'] == 0, 'fake', inplace=True)
    df2['label'].mask(df2['label'] == 'T', 'real', inplace=True)
    df2['label'].mask(df2['label'] == 'F', 'fake', inplace=True)
    df2['label'].mask(df2['label'] == 'REAL', 'real', inplace=True)
    df2['label'].mask(df2['label'] == 'TRUE', 'real', inplace=True)
    df2['label'].mask(df2['label'] == 'FAKE', 'fake', inplace=True)
    df2['label'].mask(df2['label'] == 'Real', 'real', inplace=True)
    df2['label'].mask(df2['label'] == 'True', 'real', inplace=True)
    df2['label'].mask(df2['label'] == 'Fake', 'fake', inplace=True)


    # Read from Fake/True dataset
    df3 = pd.read_csv('write-combo-fake-true.csv', skiprows = 1, names = col_names, usecols=[1,2])

    # Ensure label column is either 'real' or 'fake'
    df3['label'].mask(df3['label'] == 1, 'real', inplace=True)
    df3['label'].mask(df3['label'] == 0, 'fake', inplace=True)
    df3['label'].mask(df3['label'] == 'T', 'real', inplace=True)
    df3['label'].mask(df3['label'] == 'F', 'fake', inplace=True)
    df3['label'].mask(df3['label'] == 'REAL', 'real', inplace=True)
    df3['label'].mask(df3['label'] == 'TRUE', 'real', inplace=True)
    df3['label'].mask(df3['label'] == 'FAKE', 'fake', inplace=True)
    df3['label'].mask(df3['label'] == 'Real', 'real', inplace=True)
    df3['label'].mask(df3['label'] == 'True', 'real', inplace=True)
    df3['label'].mask(df3['label'] == 'Fake', 'fake', inplace=True)

   
   # Create new large combo dataset

    new_col_names = ['text', 'label']
    #df.to_csv('write-combo-fake-true.csv', index=True, index_label='id', header=new_col_names)
    df.to_csv(newDataset, index=True, index_label='id', header=new_col_names)

    # Append covid dataset to combo news dataset from previous index number
    df2.index += df.index[-1]+1
    df2.to_csv(newDataset, mode = 'a', index=True, index_label='id', header=False)

    # Append general fake/true news dataset to combo news dataset from previous index number
    df3.index += df.index[-1]+1
    df3.to_csv(newDataset, mode = 'a', index=True, index_label='id', header=False)


    df4 = pd.read_csv(newDataset, skiprows = 1)
    #df4 = df4.sample(frac=1).reset_index(drop=True)
    #df4 = df4.sample(frac=1)

    print(df4)