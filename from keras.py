import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model

import warnings
warnings.filterwarnings('ignore')
from random import randint

data = pd.read_csv('data.csv', header=0, na_values='?', delimiter=',', nrows =40000)


print('Количество строк и столбцов:', data.shape)

# Убираем ненужные столбцы
X = data.drop('id', axis='columns') 
X = X.drop('title', axis='columns') 
X = X.drop('num_points', axis='columns') 
X = X.drop('num_comments', axis='columns') 
X = X.drop('created_at', axis='columns') 

#Убираем строки где пустые ссылки
X.dropna(subset=['url'], inplace=True)

# добавляем новый столбец он будет хранить у каких пользоватлей есть переходы по сылкам
X['test'] = 1
X['url_text'] = data['url'] 
X['author_text'] = data['author']

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#превращаем категориалные признаки в числовые
X['url'] = labelencoder.fit_transform(X['url'])
X['author'] = labelencoder.fit_transform(X['author'])

print(X.head())


from sklearn.model_selection import train_test_split
train, test = train_test_split(X, test_size=0.2, random_state=42)

n_author = len(X.author.unique())
n_url = len(X.url.unique())

# Создаем url embedding path
url_input = Input(shape=[1], name="Url-Input")
url_embedding = Embedding(n_url+1, 5, name="Url-Embedding")(url_input)
url_vec = Flatten(name="Flatten-Urls")(url_embedding)

# Создаем user embedding path
author_input = Input(shape=[1], name="Author-Input")
author_embedding = Embedding(n_author+1, 5, name="Author-Embedding")(author_input)
author_vec = Flatten(name="Flatten-Author")(author_embedding)

#Подготовка dot слоя и создание модели
prod = Dot(name="Dot-Product", axes=1)([url_vec, author_vec])
model = Model([author_input, url_input], prod)
model.compile('adam', 'mean_squared_error')

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

from keras.models import load_model

if os.path.exists('model.h5'):
    model = load_model('model.h5')
else:
    history = model.fit([train.author, train.url], train.test, epochs=5, verbose=1)
    model.save('model.h5')
    plt.plot(history.history['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Training Error")


#Создание рекомендация для 101-го пользователя
url_data = np.array(list(set(X.url)))
print(url_data[:5])

author = np.array([101 for i in range(len(url_data))])
print(author[:5])

# Передаем все ссылки и нашего пользователя
predictions = model.predict([author, url_data])

predictions = np.array([a[0] for a in predictions])

recommended_url_ids = (-predictions).argsort()[:5]

print('id ссылок, которые рекомендуем:',recommended_url_ids)

print('Прогнозируемы результат в вещественных числах:', predictions[recommended_url_ids])

url_itog = X[X['url'].isin(recommended_url_ids)]

#удаляем дубликаты
url_itog.drop_duplicates(subset ="url_text", inplace = True) 

for url in url_itog['url_text']:
    print('Рекомендуем - ', url)