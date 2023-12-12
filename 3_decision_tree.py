'''
install graphviz library from https://graphviz.gitlab.io/_pages/Download/Download_windows.html
update your path variable to include C:\Program Files (x86)\Graphviz2.38\bin
'''

# for building the model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# for plotting tree
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus

col_names = ['Reservation', 'Raining', 'Bad Service', 'Saturday', 'Result']
hoteldata = pd.read_csv("hotel_for_decision_tree.csv", header=None, names=col_names)
feature_cols = ['Reservation', 'Raining', 'Bad Service', 'Saturday']

X = hoteldata[feature_cols]  # Feature Columns
Y = hoteldata.Result  # Target variable
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# 70% training and 30% test
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf = clf.fit(x_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(x_test)
print(f"ytest:\n{y_test}")
print("ypred: ",y_pred)

# Accuracy of the model
print("Accuracy: ", accuracy_score(y_test, y_pred))

# ploting into graph
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, feature_names=feature_cols, class_names=['Leave', 'Wait'])
try:
   graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
   graph.write_png("Hotel_neural_network.png")
except Exception as e:
   print("Error.... ",e)
finally:
   dot_data.close()


"""OUTPUT:-
   ytest:
   6     leave
   3      wait
   13    leave
   2     leave
   14    leave
   7      wait
   Name: Result, dtype: object
   ypred:  ['leave' 'leave' 'wait' 'leave' 'leave' 'wait']
   Accuracy:  0.6666666666666666
"""
