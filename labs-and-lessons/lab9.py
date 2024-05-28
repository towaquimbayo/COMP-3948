# Exercise 1
def ex1():
    from mlxtend.data import loadlocal_mnist
    from sklearn.metrics import classification_report
    import os

    X, y = loadlocal_mnist(
        images_path=f"{os.environ['DATASET_DIRECTORY']}t10k-images.idx3-ubyte",
        labels_path=f"{os.environ['DATASET_DIRECTORY']}t10k-labels.idx1-ubyte",
    )

    # http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
    print("Dimensions: %s x %s" % (X.shape[0], X.shape[1]))
    print("\n1st row", X[0])

    # Split the data.
    from sklearn.model_selection import train_test_split

    # test_size: what proportion of original data is used for test set
    train_img, test_img, train_lbl, test_lbl = train_test_split(
        X, y, test_size=1 / 7.0, random_state=0
    )
    # Show image.
    print("Image size: ")
    print(train_img[0].shape)

    import matplotlib.pyplot as plt
    import numpy as np

    first_image = train_img[0]
    first_image = np.array(first_image, dtype="float")
    print(len(first_image))
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap="gray")
    plt.show()

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # Fit on training set only.
    scaler.fit(train_img)

    # Apply transform to both the training set and the test set.
    train_img = scaler.transform(train_img)
    test_img = scaler.transform(test_img)

    from sklearn.decomposition import PCA

    # Make an instance of the PCA.
    pca = PCA(0.95)
    pca.fit(train_img)

    # Data is transformed with PCA.
    train_img = pca.transform(train_img)
    test_img = pca.transform(test_img)

    from sklearn.linear_model import LogisticRegression

    # all parameters not specified are set to their defaults
    # default solver is incredibly slow which is why it was changed to 'lbfgs'
    logisticRegr = LogisticRegression(solver="lbfgs", max_iter=1000)
    logisticRegr.fit(train_img, train_lbl)
    y_pred = logisticRegr.predict(test_img)
    score = logisticRegr.score(test_img, test_lbl)
    print(score)

    # Show confusion matrix and accuracy scores.
    import pandas as pd

    cm = pd.crosstab(test_lbl, y_pred, rownames=["Actual"], colnames=["Predicted"])

    print("\n*** Confusion Matrix")
    print(cm)

    print("\n*** Classification Report")
    print(classification_report(test_lbl, y_pred))


# Exercise 2
def ex2():
    import pandas as pd
    import numpy as np
    from sklearn import model_selection
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.preprocessing import StandardScaler
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "Hitters.csv"

    # Drop null values.
    df = pd.read_csv(PATH + CSV_DATA).dropna()
    df.info()

    dummies = pd.get_dummies(df[["League", "Division", "NewLeague"]])
    y = df.Salary

    print("\nSalary stats: ")
    print(y.describe())

    # Drop the column with the independent variable (Salary),
    # and columns for which we created dummy variables.
    X_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype(
        "float64"
    )

    # Define the feature set X.
    X = pd.concat(
        [X_, dummies[["League_N", "Division_W", "NewLeague_N"]]], axis=1
    ).astype("float64")

    # Calculate and show VIF Scores for original data.
    vif = pd.DataFrame()
    vif["VIF Factor"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    vif["features"] = X.columns
    print("\nOriginal VIF Scores")
    print(vif)

    # Standardize the data.
    X_scaled = StandardScaler().fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_scaled, y, test_size=0.25, random_state=1
    )

    # Transform the data using PCA for first 80% of variance.
    pca = PCA(0.8)
    X_reduced_train = pca.fit_transform(X_train)
    X_reduced_test = pca.transform(X_test)

    print("\nPrincipal Components")
    print(pca.components_)

    print("\nExplained variance: ")
    print(pca.explained_variance_)

    # Train regression model on training data
    model = LinearRegression()
    model.fit(X_reduced_train, y_train)

    # Prediction with test data
    pred = model.predict(X_reduced_test)
    print()

    # Show stats about the regression.
    mse = mean_squared_error(y_test, pred)
    RMSE = np.sqrt(mse)
    print("\nRMSE: " + str(RMSE))

    print("\nModel Coefficients")
    print(model.coef_)

    print("\nModel Intercept")
    print(model.intercept_)

    from sklearn.metrics import r2_score

    print("\nr2_score", r2_score(y_test, pred))

    # For each principal component, calculate the VIF and save in dataframe
    vif = pd.DataFrame()

    # Show the VIF score for the principal components.
    print()
    vif["VIF Factor"] = [
        variance_inflation_factor(X_reduced_train, i)
        for i in range(X_reduced_train.shape[1])
    ]
    print(vif)


# Exercise 3
def ex3():
    import pandas as pd
    import numpy as np
    from sklearn import model_selection
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.preprocessing import StandardScaler
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "Hitters.csv"

    # Drop null values.
    df = pd.read_csv(PATH + CSV_DATA).dropna()
    df.info()

    dummies = pd.get_dummies(df[["League", "Division", "NewLeague"]])
    y = df.Salary

    print("\nSalary stats: ")
    print(y.describe())

    # Drop the column with the independent variable (Salary),
    # and columns for which we created dummy variables.
    X_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype(
        "float64"
    )

    # Define the feature set X.
    X = pd.concat(
        [X_, dummies[["League_N", "Division_W", "NewLeague_N"]]], axis=1
    ).astype("float64")

    # Calculate and show VIF Scores for original data.
    vif = pd.DataFrame()
    vif["VIF Factor"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    vif["features"] = X.columns
    print("\nOriginal VIF Scores")
    print(vif)

    # Standardize the data.
    X_scaled = StandardScaler().fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_scaled, y, test_size=0.25, random_state=1
    )

    # Transform the data using PCA for first 90% of variance.
    pca = PCA(0.9)
    X_reduced_train = pca.fit_transform(X_train)
    X_reduced_test = pca.transform(X_test)

    print("\nPrincipal Components")
    print(pca.components_)

    print("\nExplained variance: ")
    print(pca.explained_variance_)

    # Train regression model on training data
    model = LinearRegression()
    model.fit(X_reduced_train, y_train)

    # Prediction with test data
    pred = model.predict(X_reduced_test)
    print()

    # Show stats about the regression.
    mse = mean_squared_error(y_test, pred)
    RMSE = np.sqrt(mse)
    print("\nRMSE: " + str(RMSE))

    print("\nModel Coefficients")
    print(model.coef_)

    print("\nModel Intercept")
    print(model.intercept_)

    from sklearn.metrics import r2_score

    print("\nr2_score", r2_score(y_test, pred))

    # For each principal component, calculate the VIF and save in dataframe
    vif = pd.DataFrame()

    # Show the VIF score for the principal components.
    print()
    vif["VIF Factor"] = [
        variance_inflation_factor(X_reduced_train, i)
        for i in range(X_reduced_train.shape[1])
    ]
    print(vif)


# Exercise 4
def ex4():
    import pandas as pd
    from sklearn import model_selection
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np
    import pandas as pd
    import os

    PATH = os.environ["DATASET_DIRECTORY"]
    CSV_DATA = "USA_Housing.csv"
    df = pd.read_csv(
        PATH + CSV_DATA,
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=",",
        names=(
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
            "Price",
            "Address",
        ),
    )
    # Show all columns.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    df2 = df._get_numeric_data()

    X = df2.copy()
    X.drop(["Price"], inplace=True, axis=1)
    y = df2.copy()
    y = y[["Price"]]

    print("\nPrice stats: ")
    print(y.describe())

    # Calculate and show VIF Scores for original data.
    vif = pd.DataFrame()
    vif["VIF Factor"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    vif["features"] = X.columns
    print("\nOriginal VIF Scores")
    print(vif)

    # Standardize the data.
    X_scaled = StandardScaler().fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_scaled, y, test_size=0.25, random_state=1
    )

    # Transform the data using PCA for first 80% of variance.
    pca = PCA(0.8)
    X_reduced_train = pca.fit_transform(X_train)
    X_reduced_test = pca.transform(X_test)

    print("\nPrincipal Components")
    print(pca.components_)

    print("\nExplained variance: ")
    print(pca.explained_variance_)

    # Train regression model on training data
    model = LinearRegression()
    model.fit(X_reduced_train, y_train)

    # Prediction with test data
    pred = model.predict(X_reduced_test)
    print()

    # Show stats about the regression.
    mse = mean_squared_error(y_test, pred)
    RMSE = np.sqrt(mse)
    print("\nRMSE: " + str(RMSE))

    print("\nModel Coefficients")
    print(model.coef_)

    print("\nModel Intercept")
    print(model.intercept_)

    from sklearn.metrics import r2_score

    print("\nr2_score", r2_score(y_test, pred))

    # For each principal component, calculate the VIF and save in dataframe
    vif = pd.DataFrame()

    # Show the VIF score for the principal components.
    print()
    vif["VIF Factor"] = [
        variance_inflation_factor(X_reduced_train, i)
        for i in range(X_reduced_train.shape[1])
    ]
    print(vif)


# Exercise 5
def ex5():
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample
    from sklearn.linear_model import LogisticRegression
    import os

    df = pd.read_csv(f"{os.environ['DATASET_DIRECTORY']}milk.csv")

    # Split the data at the start to hold back a test set.
    train, test = train_test_split(df, test_size=0.2)

    X_train = train.copy()
    X_test = test.copy()
    del X_train["labels"]
    del X_test["labels"]
    del X_train["dates"]
    del X_test["dates"]

    y_train = train["labels"]
    y_test = test["labels"]

    # Scale X values.
    xscaler = StandardScaler()
    Xtrain_scaled = xscaler.fit_transform(X_train)
    Xtest_scaled = xscaler.transform(X_test)

    # Generate PCA components.
    pca = PCA(0.95)

    # Always fit PCA with train data. Then transform the train data.
    X_reduced_train = pca.fit_transform(Xtrain_scaled)

    # Transform test data with PCA
    X_reduced_test = pca.transform(Xtest_scaled)

    print("\nPrincipal Components")
    print(pca.components_)

    print("\nExplained variance: ")
    print(pca.explained_variance_)

    # Train regression model on training data
    model = LogisticRegression(solver="liblinear")
    model.fit(X_reduced_train, y_train)

    # Predict with test data.
    preds = model.predict(X_reduced_test)

    report = classification_report(y_test, preds)
    print(report)


# Exercise 6 and 7
def ex6():
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample
    from sklearn.linear_model import LogisticRegression
    import os

    df = pd.read_csv(f"{os.environ['DATASET_DIRECTORY']}milk.csv")

    # Split the data at the start to hold back a test set.
    train, test = train_test_split(df, test_size=0.2)

    X_train = train.copy()
    X_test = test.copy()
    del X_train["labels"]
    del X_test["labels"]
    del X_train["dates"]
    del X_test["dates"]

    y_train = train["labels"]
    y_test = test["labels"]

    # Scale X values.
    xscaler = StandardScaler()
    Xtrain_scaled = xscaler.fit_transform(X_train)
    Xtest_scaled = xscaler.transform(X_test)

    # Generate PCA components.
    pca = PCA(0.8)

    # Always fit PCA with train data. Then transform the train data.
    X_reduced_train = pca.fit_transform(Xtrain_scaled)

    # Transform test data with PCA
    X_reduced_test = pca.transform(Xtest_scaled)

    print("\nPrincipal Components")
    print(pca.components_)

    print("\nExplained variance: ")
    print(pca.explained_variance_)

    # Train regression model on training data
    model = LogisticRegression(solver="liblinear")
    model.fit(X_reduced_train, y_train)

    # Predict with test data.
    preds = model.predict(X_reduced_test)

    report = classification_report(y_test, preds)
    print(report)


# Exercise 8
def ex8():
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample
    from sklearn.linear_model import LogisticRegression
    import os

    df = pd.read_csv(f"{os.environ['DATASET_DIRECTORY']}milk.csv")

    # Split the data at the start to hold back a test set.
    train, test = train_test_split(df, test_size=0.2)

    X_train = train.copy()
    X_test = test.copy()
    del X_train["labels"]
    del X_test["labels"]
    del X_train["dates"]
    del X_test["dates"]

    y_train = train["labels"]
    y_test = test["labels"]

    # Scale X values.
    xscaler = StandardScaler()
    Xtrain_scaled = xscaler.fit_transform(X_train)
    Xtest_scaled = xscaler.transform(X_test)

    # Generate PCA components.
    pca = PCA(0.8)

    # Always fit PCA with train data. Then transform the train data.
    X_reduced_train = pca.fit_transform(Xtrain_scaled)

    # Transform test data with PCA
    X_reduced_test = pca.transform(Xtest_scaled)

    # eig_vals, eig_vecs = np.linalg.eig(np.corrcoef(np.transpose(Xtrain_scaled)))
    # print("\nEigenvectors \n%s" % eig_vecs)
    # print("\nEigenvalues \n%s" % eig_vals)

    print("\nPrincipal Components")
    print(pca.components_)

    print("\nExplained variance: ")
    print(pca.explained_variance_)

    # Train regression model on training data
    model = LogisticRegression(solver="liblinear")
    model.fit(X_reduced_train, y_train)

    # Predict with test data.
    preds = model.predict(X_reduced_test)

    report = classification_report(y_test, preds)
    print(report)

    import matplotlib.pyplot as plt

    cov_mat = np.cov(np.transpose(X_reduced_train))

    # Perform an Eigen decomposition on the covariance matrix:
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print("\nEigenvectors \n%s" % eig_vecs)
    print("\nEigenvalues \n%s" % eig_vals)

    plt.plot([1, 2], eig_vals, "ro-", linewidth=2)
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.show()

    # Calculate cumulative values.
    sumEigenvalues = eig_vals.sum()
    cumulativeValues = []
    cumulativeSum = 0
    for i in range(0, len(eig_vals) + 1):
        cumulativeValues.append(cumulativeSum)
        if i < len(eig_vals):
            cumulativeSum += eig_vals[i] / sumEigenvalues

    # Show cumulative variance plot.
    import matplotlib.pyplot as plt

    plt.plot([0, 1, 2], cumulativeValues, "ro-", linewidth=2)
    plt.title("Variance Explained by Principal Components")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.show()


def main():
    # ex1()
    # ex2()
    # ex3()
    # ex4()
    # ex5()
    # ex6()
    ex8()


if __name__ == "__main__":
    main()
