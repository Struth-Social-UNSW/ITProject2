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

    # Read from Fake/True dataset
    #df = pd.read_csv('write-combo-fake-true.csv', skiprows = 1, names = col_names, usecols=[1,2])

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
    
       
   # Create new large combo dataset

    new_col_names = ['text', 'label']
    #df.to_csv('write-combo-fake-true.csv', index=True, index_label='id', header=new_col_names)
    df.to_csv(newDataset, index=True, index_label='id', header=new_col_names)

    # Append covid dataset to combo news dataset from previous index number
    #df2.index += df.index[-1]+1
    #df2.to_csv(newDataset, mode = 'a', index=True, index_label='id', header=False)

    # Append general fake/true news dataset to combo news dataset from previous index number
    #df3.index += df.index[-1]+1
    #df3.to_csv(newDataset, mode = 'a', index=True, index_label='id', header=False)


    df = pd.read_csv(newDataset, skiprows = 1)
    #df4 = df4.sample(frac=1).reset_index(drop=True)
    #df4 = df4.sample(frac=1)

    print(df)



    
########################################################

###   Main Program   ###

writeCSV()