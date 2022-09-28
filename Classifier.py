#binary classification neural network using scikit learn 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Fish dataset


#read data from csv file
df = pd.read_csv('fishLearn.csv', sep=',', header=0)
data_test=pd.read_csv('fishTestNoLabel.csv', sep=',', header=0)

#data visualization
print(df.sample(3))
print(df.describe())

print(data_test.sample(3))
print(data_test.describe())


#data preprocessing
#split data into features and labels
X = df.drop('species',axis=1)
y = df['species']


#split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#train the model
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=5000)
mlp.fit(X_train,y_train)

#predict the test data
predictions = mlp.predict(X_test)

#evaluate the model
print(confusion_matrix(y_test,predictions))

#plot the data
plt.scatter(df['lightness'],df['width'],c=df['species'])
plt.show()

# #predict the test data
# predictions2 = mlp.predict(data_test)

# #evaluate the model
# print(confusion_matrix(y_test,predictions2))

# #plot the data
# plt.scatter(df['lightness'],df['width'],c=df['species'])
# plt.show()

