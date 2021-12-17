import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')
data = data.apply(lambda x: x.str.replace(',','.'))
X = data.iloc[:, :4].to_numpy().astype(np.float32)
y = data.iloc[:, 4].to_numpy().astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
model = SVR(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.xlabel("Predicted values", fontsize=13)
plt.ylabel("Target values", fontsize=13)
plt.plot([420, 500], [420, 500], c='r', label='t = y')
plt.scatter(y_pred, y_test)
plt.legend()
plt.show()