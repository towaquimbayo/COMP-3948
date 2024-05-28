# Exercise 4
def ex4():
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "bank-additional-full.csv"
    df = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=";",
        names=(
            "age",
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
            "y",
        ),
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(df.head())
    print(df.describe().transpose())
    print(df.info())

    targetList = []
    for i in range(0, len(df)):
        if df.loc[i]["y"] == "yes":
            targetList.append(1)
        else:
            targetList.append(0)
    df["target"] = targetList

    tempDf = df[
        [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ]
    ]  # Isolate columns
    dummyDf = pd.get_dummies(
        tempDf,
        columns=[
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ],
    )  # Get dummies
    df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original df

    X = df[
        [
            "age",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
            "job_admin.",
            "job_blue-collar",
            "job_entrepreneur",
            "job_housemaid",
            "job_management",
            "job_retired",
            "job_self-employed",
            "job_services",
            "job_student",
            "job_technician",
            "job_unemployed",
            "job_unknown",
            "marital_divorced",
            "marital_married",
            "marital_single",
            "marital_unknown",
            "education_basic.4y",
            "education_basic.6y",
            "education_basic.9y",
            "education_high.school",
            "education_illiterate",
            "education_professional.course",
            "education_university.degree",
            "education_unknown",
            "default_no",
            "default_unknown",
            "default_yes",
            "housing_no",
            "housing_unknown",
            "housing_yes",
            "loan_no",
            "loan_unknown",
            "loan_yes",
            "contact_cellular",
            "contact_telephone",
            "month_apr",
            "month_aug",
            "month_dec",
            "month_jul",
            "month_jun",
            "month_mar",
            "month_may",
            "month_nov",
            "month_oct",
            "month_sep",
            "day_of_week_fri",
            "day_of_week_mon",
            "day_of_week_thu",
            "day_of_week_tue",
            "day_of_week_wed",
            "poutcome_failure",
            "poutcome_nonexistent",
            "poutcome_success",
        ]
    ]
    y = df[["target"]]
    print(X.head())

    # Create the object of the model
    model = LogisticRegression(solver="liblinear")

    # Specify the number of  features to select
    rfe = RFE(estimator=model, n_features_to_select=15)

    # fit the model
    rfe = rfe.fit(X, y)

    # Please uncomment the following lines to see the result
    print("\n\nFEATURES SELECTED\n\n")
    print(rfe.support_)

    # Show top features.
    for i in range(0, len(X.keys())):
        if rfe.support_[i]:
            print(X.keys()[i])


# Exercise 5 and 6
def ex5():
    # Show chi-square scores for each feature.
    # There is 1 degree freedom since 1 predictor during feature evaluation.
    # Generally, >=3.8 is good)
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    from sklearn.metrics import roc_curve
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import (
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
    )
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    def buildAndEvaluateClassifier(features, X, y):
        # Re-assign X with significant columns only after chi-square test.
        X = X[features]

        # Split data.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Perform logistic regression.
        logisticModel = LogisticRegression(fit_intercept=True, solver="liblinear")

        # Fit the model.
        logisticModel.fit(X_train, y_train)
        y_pred = logisticModel.predict(X_test)
        y_prob = logisticModel.predict_proba(X_test)
        # print(y_pred)

        # Show accuracy scores.
        print("Results without scaling:")

        print("Recall:    " + str(recall_score(y_test, y_pred)))
        print("Precision: " + str(precision_score(y_test, y_pred)))
        print("F1:        " + str(f1_score(y_test, y_pred)))
        print("Accuracy:  " + str(accuracy_score(y_test, y_pred)))
        return X_test, y_test, y_pred, y_prob

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "bank-additional-full.csv"
    df = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=";",
        names=(
            "age",
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
            "y",
        ),
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(df.head())
    print(df.describe().transpose())
    print(df.info())

    targetList = []
    for i in range(0, len(df)):
        if df.loc[i]["y"] == "yes":
            targetList.append(1)
        else:
            targetList.append(0)
    df["target"] = targetList

    tempDf = df[
        [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ]
    ]  # Isolate columns
    dummyDf = pd.get_dummies(
        tempDf,
        columns=[
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ],
    )  # Get dummies
    df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original df

    X = df[
        [
            "age",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
            "job_admin.",
            "job_blue-collar",
            "job_entrepreneur",
            "job_housemaid",
            "job_management",
            "job_retired",
            "job_self-employed",
            "job_services",
            "job_student",
            "job_technician",
            "job_unemployed",
            "job_unknown",
            "marital_divorced",
            "marital_married",
            "marital_single",
            "marital_unknown",
            "education_basic.4y",
            "education_basic.6y",
            "education_basic.9y",
            "education_high.school",
            "education_illiterate",
            "education_professional.course",
            "education_university.degree",
            "education_unknown",
            "default_no",
            "default_unknown",
            "default_yes",
            "housing_no",
            "housing_unknown",
            "housing_yes",
            "loan_no",
            "loan_unknown",
            "loan_yes",
            "contact_cellular",
            "contact_telephone",
            "month_apr",
            "month_aug",
            "month_dec",
            "month_jul",
            "month_jun",
            "month_mar",
            "month_may",
            "month_nov",
            "month_oct",
            "month_sep",
            "day_of_week_fri",
            "day_of_week_mon",
            "day_of_week_thu",
            "day_of_week_tue",
            "day_of_week_wed",
            "poutcome_failure",
            "poutcome_nonexistent",
            "poutcome_success",
        ]
    ]
    y = df[["target"]]
    print(X.head())

    # Create the object of the model
    model = LogisticRegression(solver="liblinear")

    # Specify the number of  features to select
    rfe = RFE(estimator=model, n_features_to_select=15)

    # fit the model
    rfe = rfe.fit(X, y)

    # Please uncomment the following lines to see the result
    print("\n\nFEATURES SELECTED\n\n")
    print(rfe.support_)

    # Show top features.
    for i in range(0, len(X.keys())):
        if rfe.support_[i]:
            print(X.keys()[i])

    test = SelectKBest(score_func=chi2, k=15)

    # Use scaled data to fit KBest
    XScaled = MinMaxScaler().fit_transform(X)
    chiScores = test.fit(XScaled, y)  # Summarize scores
    np.set_printoptions(precision=3)

    # Search here for insignificant features.
    print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))

    # Create a sorted list of the top features.
    dfFeatures = pd.DataFrame()
    for i in range(0, len(chiScores.scores_)):
        headers = list(X.keys())
        featureObject = {
            "feature": headers[i],
            "chi-square score": chiScores.scores_[i],
        }
        dfFeatures = dfFeatures._append(featureObject, ignore_index=True)

    print("\nTop Features")
    dfFeatures = dfFeatures.sort_values(by=["chi-square score"], ascending=False)
    print(dfFeatures.tail(40))

    # Chi-square with top 15 features.
    X_test, y_test, y_pred, y_prob = buildAndEvaluateClassifier(
        dfFeatures["feature"], X, y
    )

    auc = roc_auc_score(
        y_test,
        y_prob[:, 1],
    )
    print("Logistic: ROC AUC=%.3f" % (auc))

    # calculate roc curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob[:, 1])
    plt.plot(lr_fpr, lr_tpr, marker=".", label="Logistic")
    plt.plot([0, 1], [0, 1], "--", label="No Skill")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    print(df["target"].value_counts())


# Exercise 7
def ex7():
    # Show chi-square scores for each feature.
    # There is 1 degree freedom since 1 predictor during feature evaluation.
    # Generally, >=3.8 is good)
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    from sklearn.metrics import roc_curve
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import (
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
    )
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    def buildAndEvaluateClassifier(features, X, y):
        # Re-assign X with significant columns only after chi-square test.
        X = X[features]

        # Split data.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Perform logistic regression.
        logisticModel = LogisticRegression(fit_intercept=True, solver="liblinear")

        # Fit the model.
        logisticModel.fit(X_train, y_train)
        y_pred = logisticModel.predict(X_test)
        y_prob = logisticModel.predict_proba(X_test)
        # print(y_pred)

        # Show accuracy scores.
        print("Results without scaling:")

        print("Recall:    " + str(recall_score(y_test, y_pred)))
        print("Precision: " + str(precision_score(y_test, y_pred)))
        print("F1:        " + str(f1_score(y_test, y_pred)))
        print("Accuracy:  " + str(accuracy_score(y_test, y_pred)))
        return X_test, y_test, y_pred, y_prob

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "bank-additional-full.csv"
    df = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=";",
        names=(
            "age",
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
            "y",
        ),
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(df.head())
    print(df.describe().transpose())
    print(df.info())

    targetList = []
    for i in range(0, len(df)):
        if df.loc[i]["y"] == "yes":
            targetList.append(1)
        else:
            targetList.append(0)
    df["target"] = targetList

    tempDf = df[
        [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ]
    ]  # Isolate columns
    dummyDf = pd.get_dummies(
        tempDf,
        columns=[
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ],
    )  # Get dummies
    df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original df

    X = df[
        [
            "age",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
            "job_admin.",
            "job_blue-collar",
            "job_entrepreneur",
            "job_housemaid",
            "job_management",
            "job_retired",
            "job_self-employed",
            "job_services",
            "job_student",
            "job_technician",
            "job_unemployed",
            "job_unknown",
            "marital_divorced",
            "marital_married",
            "marital_single",
            "marital_unknown",
            "education_basic.4y",
            "education_basic.6y",
            "education_basic.9y",
            "education_high.school",
            "education_illiterate",
            "education_professional.course",
            "education_university.degree",
            "education_unknown",
            "default_no",
            "default_unknown",
            "default_yes",
            "housing_no",
            "housing_unknown",
            "housing_yes",
            "loan_no",
            "loan_unknown",
            "loan_yes",
            "contact_cellular",
            "contact_telephone",
            "month_apr",
            "month_aug",
            "month_dec",
            "month_jul",
            "month_jun",
            "month_mar",
            "month_may",
            "month_nov",
            "month_oct",
            "month_sep",
            "day_of_week_fri",
            "day_of_week_mon",
            "day_of_week_thu",
            "day_of_week_tue",
            "day_of_week_wed",
            "poutcome_failure",
            "poutcome_nonexistent",
            "poutcome_success",
        ]
    ]
    y = df[["target"]]
    print(X.head())

    # Create the object of the model
    model = LogisticRegression(solver="liblinear")

    # Specify the number of  features to select
    rfe = RFE(estimator=model, n_features_to_select=15)

    # fit the model
    rfe = rfe.fit(X, y)

    # Please uncomment the following lines to see the result
    print("\n\nFEATURES SELECTED\n\n")
    print(rfe.support_)

    # Show top features.
    for i in range(0, len(X.keys())):
        if rfe.support_[i]:
            print(X.keys()[i])

    test = SelectKBest(score_func=chi2, k=15)

    # Use scaled data to fit KBest
    XScaled = MinMaxScaler().fit_transform(X)
    chiScores = test.fit(XScaled, y)  # Summarize scores
    np.set_printoptions(precision=3)

    # Search here for insignificant features.
    print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))

    # Create a sorted list of the top features.
    dfRfeFeatures = pd.DataFrame()
    for i in range(0, len(rfe.support_)):
        headers = list(X.keys())
        featureObject = {
            "feature": headers[i],
            "rfe score": rfe.support_[i],
        }
        dfRfeFeatures = dfRfeFeatures._append(featureObject, ignore_index=True)

    dfFeatures = pd.DataFrame()
    for i in range(0, len(chiScores.scores_)):
        headers = list(X.keys())
        featureObject = {
            "feature": headers[i],
            "chi-square score": chiScores.scores_[i],
        }
        dfFeatures = dfFeatures._append(featureObject, ignore_index=True)

    print("\nTop Features")
    dfFeatures = dfFeatures.sort_values(by=["chi-square score"], ascending=False)
    # print(dfFeatures.tail(40))
    print(dfFeatures.head(40))

    # Chi-square with top 15 features.
    X_test, y_test, y_pred, y_prob = buildAndEvaluateClassifier(
        dfFeatures["feature"], X, y
    )

    auc = roc_auc_score(
        y_test,
        y_prob[:, 1],
    )
    print("Logistic: ROC AUC=%.3f" % (auc))

    X_test2, y_test2, y_pred2, y_prob2 = buildAndEvaluateClassifier(
        dfRfeFeatures["feature"], X, y
    )

    auc = roc_auc_score(
        y_test2,
        y_prob2[:, 1],
    )
    print("Logistic: ROC AUC=%.3f" % (auc))


# Exercise 8
def ex8():
    import pandas as pd
    from sklearn.feature_selection import f_regression
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import numpy as np
    import os

    def buildAndEvaluateClassifier(features, X, y):
        # Re-assign X with significant columns only after chi-square test.
        X = X[features]

        # Split data.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Perform logistic regression.
        logisticModel = LogisticRegression(fit_intercept=True, solver="liblinear")

        # Fit the model.
        logisticModel.fit(X_train, y_train)
        y_pred = logisticModel.predict(X_test)
        y_prob = logisticModel.predict_proba(X_test)
        # print(y_pred)

        # Show accuracy scores.
        print("Results without scaling:")

        print("Recall:    " + str(recall_score(y_test, y_pred)))
        print("Precision: " + str(precision_score(y_test, y_pred)))
        print("F1:        " + str(f1_score(y_test, y_pred)))
        print("Accuracy:  " + str(accuracy_score(y_test, y_pred)))
        return X_test, y_test, y_pred, y_prob

    # Read the data:
    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "bank-additional-full.csv"
    df = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=";",
        names=(
            "age",
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
            "y",
        ),
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    targetList = []
    for i in range(0, len(df)):
        if df.loc[i]["y"] == "yes":
            targetList.append(1)
        else:
            targetList.append(0)
    df["target"] = targetList

    tempDf = df[
        [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ]
    ]  # Isolate columns
    dummyDf = pd.get_dummies(
        tempDf,
        columns=[
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
        ],
    )  # Get dummies
    df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original df

    X = df[
        [
            "age",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
            "job_admin.",
            "job_blue-collar",
            "job_entrepreneur",
            "job_housemaid",
            "job_management",
            "job_retired",
            "job_self-employed",
            "job_services",
            "job_student",
            "job_technician",
            "job_unemployed",
            "job_unknown",
            "marital_divorced",
            "marital_married",
            "marital_single",
            "marital_unknown",
            "education_basic.4y",
            "education_basic.6y",
            "education_basic.9y",
            "education_high.school",
            "education_illiterate",
            "education_professional.course",
            "education_university.degree",
            "education_unknown",
            "default_no",
            "default_unknown",
            "default_yes",
            "housing_no",
            "housing_unknown",
            "housing_yes",
            "loan_no",
            "loan_unknown",
            "loan_yes",
            "contact_cellular",
            "contact_telephone",
            "month_apr",
            "month_aug",
            "month_dec",
            "month_jul",
            "month_jun",
            "month_mar",
            "month_may",
            "month_nov",
            "month_oct",
            "month_sep",
            "day_of_week_fri",
            "day_of_week_mon",
            "day_of_week_thu",
            "day_of_week_tue",
            "day_of_week_wed",
            "poutcome_failure",
            "poutcome_nonexistent",
            "poutcome_success",
        ]
    ]
    y = df[["target"]]

    #  f_regression is a scoring function to be used in a feature selection procedure
    #  f_regression will compute the correlation between each regressor and the target
    print("\n\nFEATURES SELECTED FOR FAST FEATURE\n\n")
    ffs = f_regression(X, y)
    dfFastFeatures = pd.DataFrame()
    for i in range(0, len(X.columns) - 1):
        headers = list(X.keys())
        featureObject = {
            "feature": headers[i],
            "fast score": ffs[0][i],
        }
        dfFastFeatures = dfFastFeatures._append(featureObject, ignore_index=True)

    # Show top features.
    dfFastFeatures = dfFastFeatures.sort_values(by=["fast score"], ascending=False)
    print(dfFastFeatures.head(15))

    # Create the object of the model
    model = LogisticRegression(solver="liblinear")
    rfe = RFE(estimator=model, n_features_to_select=15)
    rfe = rfe.fit(X, y)
    dfRfeFeatures = pd.DataFrame()
    print("\n\nFEATURES SELECTED FOR RFE FEATURE\n\n")
    for i in range(0, len(rfe.support_)):
        headers = list(X.keys())
        featureObject = {
            "feature": headers[i],
            "rfe score": rfe.support_[i],
        }
        dfRfeFeatures = dfRfeFeatures._append(featureObject, ignore_index=True)

    # Show top features.
    dfRfeFeatures = dfRfeFeatures.sort_values(by=["rfe score"], ascending=False)
    print(dfRfeFeatures.head(15))

    test = SelectKBest(score_func=chi2, k=15)
    # Use scaled data to fit KBest
    chiScores = test.fit(MinMaxScaler().fit_transform(X), y)  # Summarize scores
    np.set_printoptions(precision=3)
    dfFeatures = pd.DataFrame()
    print("\n\nFEATURES SELECTED FOR CHI-SQUARE\n\n")
    for i in range(0, len(chiScores.scores_)):
        headers = list(X.keys())
        featureObject = {
            "feature": headers[i],
            "chi-square score": chiScores.scores_[i],
        }
        dfFeatures = dfFeatures._append(featureObject, ignore_index=True)
    dfFeatures = dfFeatures.sort_values(by=["chi-square score"], ascending=False)
    print(dfFeatures.head(15))

    # Fast with top 15 features.
    X_test, y_test, y_pred, y_prob = buildAndEvaluateClassifier(
        dfFastFeatures["feature"], X, y
    )

    # RFE with top 15 features.
    X_test, y_test, y_pred, y_prob = buildAndEvaluateClassifier(
        dfRfeFeatures["feature"], X, y
    )

    # Chi-square with top 15 features.
    X_test, y_test, y_pred, y_prob = buildAndEvaluateClassifier(
        dfFeatures["feature"], X, y
    )


def main():
    # ex4()
    # ex5()
    # ex7()
    ex8()


if __name__ == "__main__":
    main()
