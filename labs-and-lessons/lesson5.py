# Exercise 3
def ex3():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "computerPurchase.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Separate into x and y values.
    X = df[["Age", "EstimatedSalary"]]
    y = df["Purchased"]

    # ## SECTION A ########################################
    # # Split data.
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # ## SECTION A ########################################
    #
    # ## SECTION B ########################################
    # # Perform logistic regression.
    # logisticModel = LogisticRegression(fit_intercept=True, solver="liblinear")
    #
    # # Fit the model.
    # logisticModel.fit(X_train, y_train)
    # y_pred = logisticModel.predict(X_test)
    # ## SECTION B ########################################

    ## SECTION A ########################################
    from sklearn.preprocessing import MinMaxScaler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    sc_x = MinMaxScaler()
    X_train_scaled = sc_x.fit_transform(X_train)  # Fit and transform X.
    X_test_scaled = sc_x.transform(X_test)  # Transform X.
    ## SECTION A ########################################

    ## SECTION B ########################################
    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, solver="liblinear")
    # Fit the model.
    logisticModel.fit(X_train_scaled, y_train)
    y_pred = logisticModel.predict(X_test_scaled)
    ## SECTION B ########################################

    # Show confusion matrix and accuracy scores.
    cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])

    print("\nAccuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(cm)


# Exercise 4
def ex4():
    import pandas as pd
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "computerPurchase.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Separate into x and y values.
    X = df[["Age", "EstimatedSalary"]]
    y = df["Purchased"]

    # Split data.
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    def showAutomatedScalerResults(X):
        sc_x = StandardScaler()
        X_Scale = sc_x.fit_transform(X)
        salary = X.iloc[1][1]
        scaledSalary = X_Scale[1][1]  # Get first scaled salary.
        print("The first unscaled salary in the list is: " + str(salary))
        print("$20,000 scaled using StandardScaler() is: " + str(scaledSalary))

    def getSD_with_zeroDegreesFreedom(X):
        mean = X["EstimatedSalary"].mean()

        # StandardScaler calculates the standard deviation with zero degrees of freedom.
        s1 = df["EstimatedSalary"].std(ddof=0)
        print("sd with 0 degrees of freedom automated: " + str(s1))

        # This is the same calculation manually. (**2 squares the result)
        s2 = np.sqrt(np.sum(((X["EstimatedSalary"] - mean) ** 2)) / (len(X)))
        print("sd with 0 degrees of freedom manually:  " + str(s2))

        return s1

    print("*** Showing automated results: ")
    showAutomatedScalerResults(X)

    print("\n*** Showing manually calculated results: ")
    sd = getSD_with_zeroDegreesFreedom(X)
    mean = df["EstimatedSalary"].mean()
    scaled = (20000 - mean) / sd

    print("$20,000 scaled manually is: " + str(scaled))


# Exercise 9
def ex9():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    from sklearn.preprocessing import RobustScaler
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "computerPurchase.csv"
    df = pd.read_csv(PATH + CSV_DATA, sep=",")

    import numpy as np
    from sklearn.model_selection import KFold

    # prepare cross validation with three folds and 1 as a random seed.
    kfold = KFold(n_splits=3, shuffle=True)
    accuracyList = []
    precisionList = []
    recallList = []
    foldCount = 0

    def getTestAndTrainData(trainIndexes, testIndexes, df):
        dfTrain = df.iloc[trainIndexes, :]  # Gets all rows with train indexes.
        dfTest = df.iloc[trainIndexes, :]

        X_train = dfTrain[["EstimatedSalary", "Age"]]
        X_test = dfTest[["EstimatedSalary", "Age"]]
        y_train = dfTrain[["Purchased"]]
        y_test = dfTest[["Purchased"]]
        return X_train, X_test, y_train, y_test

    for trainIdx, testIdx in kfold.split(df):
        X_train, X_test, y_train, y_test = getTestAndTrainData(trainIdx, testIdx, df)

        # Recommended to only fit on training data.
        # Scaling only needed for X since y ranges between 0 and 1.
        scalerX = RobustScaler()
        X_train_scaled = scalerX.fit_transform(X_train)  # Fit and transform.
        X_test_scaled = scalerX.transform(X_test)  # Transform only.

        # Perform logistic regression.
        logisticModel = LogisticRegression(fit_intercept=True, solver="liblinear")
        # Fit the model.
        logisticModel.fit(X_train_scaled, y_train)

        y_pred = logisticModel.predict(X_test_scaled)
        y_prob = logisticModel.predict_proba(X_test_scaled)

        # Show confusion matrix and accuracy scores.
        y_test_array = np.array(y_test["Purchased"])
        cm = pd.crosstab(
            y_test_array, y_pred, rownames=["Actual"], colnames=["Predicted"]
        )

        print("\n***K-fold: " + str(foldCount))
        foldCount += 1

        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracyList.append(accuracy)
        precisionList.append(metrics.precision_score(y_test, y_pred))
        recallList.append(metrics.recall_score(y_test, y_pred))
        print("\nAccuracy: ", accuracy)
        print("\nConfusion Matrix")
        print(cm)

        from sklearn.metrics import classification_report, roc_auc_score

        print(classification_report(y_test, y_pred))

        from sklearn.metrics import average_precision_score

        average_precision = average_precision_score(y_test, y_pred)

        print("Average precision-recall score: {0:0.2f}".format(average_precision))

        # calculate scores
        auc = roc_auc_score(
            y_test,
            y_prob[:, 1],
        )
        print("Logistic: ROC AUC=%.3f" % (auc))

    print("\nAccuracy and Standard Deviation For All Folds:")
    print("*********************************************")
    print("Average accuracy: " + str(np.mean(accuracyList)))
    print("Accuracy std: " + str(np.std(accuracyList)))
    print("Average precision: " + str(np.mean(precisionList)))
    print("Precision std: " + str(np.std(precisionList)))
    print("Average recall: " + str(np.mean(recallList)))
    print("Recall std: " + str(np.std(recallList)))


# Exercise 12
def ex12():
    import pandas as pd
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import numpy as np
    from sklearn import metrics

    wine = datasets.load_wine()
    dataset = pd.DataFrame(
        data=np.c_[wine["data"], wine["target"]],
        columns=wine["feature_names"] + ["target"],
    )

    # Create copy to prevent overwrite.
    X = dataset.copy()
    del X["target"]  # Remove target variable
    del X["hue"]  # Remove unwanted features
    del X["ash"]
    del X["magnesium"]
    del X["malic_acid"]
    del X["alcohol"]

    y = dataset["target"]

    # Adding an intercept *** This is required ***. Don't forget this step.
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn.preprocessing import RobustScaler

    sc_x = RobustScaler()
    X_train_scaled = sc_x.fit_transform(X_train)

    # Create y scaler. Only scale y_train since evaluation
    # will use the actual size y_test.
    sc_y = RobustScaler()
    y_train_scaled = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

    # Save the fitted scalers.
    from pickle import dump, load

    dump(sc_x, open("sc_x.pkl", "wb"))
    dump(sc_y, open("sc_y.pkl", "wb"))

    # Build model with training data.
    model = sm.OLS(y_train_scaled, X_train_scaled).fit()
    dump(model, open("model.pkl", "wb"))

    # Load the scalers.
    loaded_scalerX = load(open("sc_x.pkl", "rb"))
    loaded_scalery = load(open("sc_y.pkl", "rb"))
    loaded_model = load(open("model.pkl", "rb"))

    X_test_scaled = loaded_scalerX.transform(X_test)
    unscaledPredictions = loaded_model.predict(X_test_scaled)  # make predictions

    # Rescale predictions back to actual size range.
    predictions = loaded_scalery.inverse_transform(
        np.array(unscaledPredictions).reshape(-1, 1)
    )

    print(loaded_model.summary())
    print(
        "Root Mean Squared Error:",
        np.sqrt(metrics.mean_squared_error(y_test, predictions)),
    )


def main():
    ex3()
    ex4()
    ex9()
    ex12()


if __name__ == "__main__":
    main()
