# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 20:01:42 2024

@author: 11478
"""

import pandas as pd
from sklearn.model_selection import train_test_split

dataset_name = 'Hepatotoxicity'
dataset = dataset_name + '/' + dataset_name + '_mito'

data = pd.read_csv(dataset + '.csv')
train, test = train_test_split(data, test_size=0.20, random_state=42, shuffle=True)

train.to_csv(dataset + '_train2.csv')
test.to_csv(dataset + '_test2.csv')

# train = pd.read_csv(dataset + '_train.csv')
# test = pd.read_csv(dataset + '_test.csv')

print(f'{dataset} dataset is split into train, test and valid sets')
print(f'Train set size: {len(train)}')
print(f'train y=0 : {len(train[train["hepatotoxicity"]==0])} \n train y=1 : {len(train[train["hepatotoxicity"]==1])}')
print(f'Test set size: {len(test)}')
print(f'test y=0 : {len(test[test["hepatotoxicity"]==0])} \n test y=1 : {len(test[test["hepatotoxicity"]==1])}')
print(f'Total set size: {len(data)}')
