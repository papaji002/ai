import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Iris.csv')
X = df.drop(['ID', 'Target'], axis=1).values
y = df['Target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

classification_accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", classification_accuracy)


"""OUTPUT:-
   Classification Accuracy: 1.0
"""
