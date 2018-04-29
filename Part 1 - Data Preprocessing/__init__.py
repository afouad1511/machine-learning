import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3]
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
imputer = Imputer(missing_values= 'NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])

labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 1:3] = imputer.transform(X[:, 1:3])
onhotencoder = OneHotEncoder(categorical_features = [0])
X = onhotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()

y = labelencoder.fit_transform(y)


from sklearn.cross_validation import train_test_split


print(X)