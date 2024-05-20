import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split

salary_df = pd.read_csv("model/dataset_salary.csv")
labelencoder = LabelEncoder()
salary_df['Место проживания'] = labelencoder.fit_transform(salary_df['Место проживания'])
X = salary_df.drop(["Статус"], axis=1)
Y = salary_df["Статус"]
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=3)

shuffle_index = np.random.permutation(salary_df.shape[0])
req_data = salary_df.iloc[shuffle_index]
train_size = int(req_data.shape[0]*0.7)
train_df = req_data.iloc[:train_size,:]
train = train_df.values

from math import sqrt
def euclidean_distance(x_test, x_train):
    distance = 0
    for i in range(len(x_test)-1):
        distance += (x_test[i]-x_train[i])**2
    return sqrt(distance)

def get_neighbors(x_test, x_train, num_neighbors):
    distances = []
    data = []
    for i in x_train:
        distances.append(euclidean_distance(x_test,i))
        data.append(i)
    distances = np.array(distances)
    data = np.array(data)
    sort_indexes = distances.argsort()             #argsort() функция возвращает индексы путем сортировки данных о расстояниях в порядке возрастания
    data = data[sort_indexes]                      #изменяем наши данные на основе отсортированных индексов, чтобы мы могли получить ближайших соседей
    return data[:num_neighbors]

def prediction(x_test, x_train, num_neighbors):
    classes = []
    neighbors = get_neighbors(x_test, x_train, num_neighbors)
    for i in neighbors:
        classes.append(i[-1])
    predicted = max(classes, key=classes.count)
    return predicted

def accuracy(y_true, y_pred):
  num_correct = 0
  for i in range(len(y_true)):
      if y_true[i]==y_pred[i]:
          num_correct+=1
  accuracy = num_correct/len(y_true)
  return accuracy

