import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv', header=0, na_values='?', delimiter=',', nrows = 20)


print('Количество строк и столбцов:', data.shape)
#print('Первые пять строк:\n', data.head(10))



X = data.drop('id', axis='columns') 
X = X.drop('title', axis='columns') 
X = X.drop('num_points', axis='columns') 
X = X.drop('num_comments', axis='columns') 
X = X.drop('created_at', axis='columns') 

X.dropna(subset=['url'], inplace=True)

X['test'] = 1

print(X.count(axis=0))

print(X.head(20))



# выводим количество пользователей и ссылок
n_users = X['author'].unique().shape[0]
n_items = X['url'].unique().shape[0]


print('Уникальных пользователей:', n_users)
print('Уникальных ссылок:', n_items)

print(X.loc[X['author'] == 'altstar'])

matr = X.pivot_table(index='author',columns='url', values='test',fill_value=0)
print(matr)

from sklearn.metrics.pairwise import pairwise_distances

# считаем косинусное расстояние для пользователей и фильмов
item_similarity = pairwise_distances(matr.T, metric='cosine')
print(item_similarity)




def predict(ratings, similarity):
    pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

item_prediction = predict(matr, item_similarity)



from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, ground_truth):
    return sqrt(mean_squared_error(prediction, ground_truth))

print('Item-based CF RMSE: ' + str(rmse(item_prediction, matr)))