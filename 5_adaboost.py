import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

col_names = ['Reservation', 'Raining', 'Bad Service', 'Saturday', 'Result']
hoteldata = pd.read_csv("dtree.csv", header=None, names=col_names)
feature_cols =  ['Reservation', 'Raining', 'Bad Service', 'Saturday']

X= hoteldata[feature_cols]
Y= hoteldata.Result
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

adahotel = AdaBoostClassifier(n_estimators=6, learning_rate=2)
adahotel = adahotel.fit(x_train, y_train)

y_pred = adahotel.predict(x_test)
print(f"y_test=\n{y_test}  \ny_pred= {y_pred}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


"""OUTPUT:-
   y_test=
   6     leave
   3      wait
   13    leave
   2     leave
   14    leave
   7      wait
   Name: Result, dtype: object
   y_pred= ['leave' 'leave' 'leave' 'leave' 'leave' 'wait']
   Accuracy: 0.8333333333333334
"""
