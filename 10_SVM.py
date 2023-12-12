# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('Iris.csv')
x= data.drop('Target', axis=1)
y= data['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(x_train, y_train)

y_pred = svm_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Y-test:\n{y_test}     \n\nY-pred:{y_pred}")
print(f"Accuracy: {accuracy:.2f}")

"""OUTPUT:-
   Y-test:
   73     Iris-versicolor
   18         Iris-setosa
   118     Iris-virginica
   78     Iris-versicolor
   76     Iris-versicolor
   31         Iris-setosa
   64     Iris-versicolor
   141     Iris-virginica
   68     Iris-versicolor
   82     Iris-versicolor
   110     Iris-virginica
   12         Iris-setosa
   36         Iris-setosa
   9          Iris-setosa
   19         Iris-setosa
   56     Iris-versicolor
   104     Iris-virginica
   69     Iris-versicolor
   55     Iris-versicolor
   132     Iris-virginica
   29         Iris-setosa
   127     Iris-virginica
   26         Iris-setosa
   128     Iris-virginica
   131     Iris-virginica
   145     Iris-virginica
   108     Iris-virginica
   143     Iris-virginica
   45         Iris-setosa
   30         Iris-setosa
   Name: Target, dtype: object
   Y-pred:['Iris-versicolor' 'Iris-setosa' 'Iris-virginica' 'Iris-versicolor'
      'Iris-versicolor' 'Iris-setosa' 'Iris-versicolor' 'Iris-virginica'
      'Iris-versicolor' 'Iris-versicolor' 'Iris-virginica' 'Iris-setosa'
      'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-versicolor'
      'Iris-virginica' 'Iris-versicolor' 'Iris-versicolor' 'Iris-virginica'
      'Iris-setosa' 'Iris-virginica' 'Iris-setosa' 'Iris-virginica'
      'Iris-virginica' 'Iris-virginica' 'Iris-virginica' 'Iris-virginica'
      'Iris-setosa' 'Iris-setosa']
   Accuracy: 1.00
"""
