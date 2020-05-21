import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv', header=0, na_values='?', delimiter=',')


print('Количество строк и столбцов:', data.shape)
print('Первые пять строк:\n', data.head())


print('Разбивка по числовым данным:\n',data.describe(), '\n')

X = data.drop('id', axis='columns') 
X = X.drop('title', axis='columns') 
X = X.drop('num_points', axis='columns') 
X = X.drop('num_comments', axis='columns') 
X = X.drop('created_at', axis='columns') 


print(X.count(axis=0))