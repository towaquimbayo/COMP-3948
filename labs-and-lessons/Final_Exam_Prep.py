import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
df = pd.read_csv("users/towa/downloads/Sample.csv", skiprows=1, sep=",")

X = df[["FeatureA", "FeatureB"]]
X = sm.add_constant(X)
Y = df[["Target"]]

# CROSS FOLD VALIDATION
k_fold = KFold(n_splits=10, shuffle=True)
rmse_list = []  # for LINEAR Regression ONLY
# for LOGISTIC Regression ONLY
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for train_index, test_index in k_fold.split(X):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    sc_x = (
        RobustScaler()
    )  # SCALING or use MinMaxScaler(), StandardScaler() for LOGISTIC
    x_train = RobustScaler().fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    # for LINEAR Regression
    model = sm.OLS(y_train, x_train).fit()  # Linear OLS
    pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    rmse_list.append(rmse)

    # for LOGISTIC Regression
    model = LogisticRegression(fit_intercept=True, solver="liblinear")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy_list.append(accuracy_score(y_test, predictions))
    precision_list.append(precision_score(y_test, predictions))
    recall_list.append(recall_score(y_test, predictions))
    f1_list.append(f1_score(y_test, predictions))

print("Average rmse: " + str(np.mean(rmse_list)))  # LINEAR ONLY
# LOGISTIC STATS FROM HERE
print(f"Average Accuracy: {np.mean(accuracy_list)}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_list)}")
print(f"Average Precision: {np.mean(precision_list)}")
print(f"Standard Deviation of Precision: {np.std(precision_list)}")
print(f"Average Recall: {np.mean(recall_list)}")
print(f"Standard Deviation of Recall: {np.std(recall_list)}")
print(f"Average F1-score: {np.mean(f1_list)}")
print(f"Standard Deviation of F1-score: {np.std(f1_list)}")

Y = df["Target"]  # reset y
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.8
)  # Train: 80% Test 20%

# Linear Regression
model = sm.OLS(y_train, x_train).fit()
predictions = model.predict(x_test)  # make the predictions by the model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(model.summary())

# Logistic Regression
model = LogisticRegression(fit_intercept=True, solver="liblinear", random_state=0)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
