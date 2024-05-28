# Exercise 12
def ex12():
    from sklearn.preprocessing import StandardScaler
    from sklearn import datasets
    import numpy as np

    # Load iris data set and apply standard scaler.
    iris = datasets.load_iris()
    X = iris.data
    featureNames = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    y = iris.target
    X_std = StandardScaler().fit_transform(X)

    print(featureNames)

    # Generate covariance matrix to show bivariate relationships.
    cov_mat = np.cov(np.transpose(X_std))
    print("\nCovariance matrix: \n%s" % cov_mat)

    # When data is standardized, the covariance matrix is same as the
    # correlation matrix.
    cor_mat = np.corrcoef(np.transpose(X_std))
    print("\nCorrelation matrix: \n%s" % cor_mat)

    # Perform an Eigen decomposition on the covariance matrix:
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print("\nEigenvectors \n%s" % eig_vecs)
    print("\nEigenvalues \n%s" % eig_vals)

    from sklearn.preprocessing import StandardScaler
    from sklearn import datasets
    import numpy as np

    # Load iris data set and apply standard scaler.
    iris = datasets.load_iris()
    X = iris.data
    featureNames = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    y = iris.target
    X_std = StandardScaler().fit_transform(X)

    print(featureNames)

    # Generate covariance matrix to show bivariate relationships.
    cov_mat = np.cov(np.transpose(X_std))
    print("\nCovariance matrix: \n%s" % cov_mat)

    # When data is standardized, the covariance matrix is same as the
    # correlation matrix.
    cor_mat = np.corrcoef(np.transpose(X_std))
    print("\nCorrelation matrix: \n%s" % cor_mat)

    # Perform an Eigen decomposition on the covariance matrix:
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print("\nEigenvectors \n%s" % eig_vecs)
    print("\nEigenvalues \n%s" % eig_vals)

    #######
    # Show the scree plot.
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as sklearnPCA
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn import datasets
    import numpy as np
    import pandas as pd

    # Load iris data set and apply standard scaler.
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_std = StandardScaler().fit_transform(X)

    # Split x and y into test and training.
    X_train, X_test, y_train, y_test = train_test_split(
        X_std, y, test_size=0.25, random_state=0
    )

    # Create principal components. (Use only 2 significant eigen vectors)
    sklearn_pca = sklearnPCA(n_components=2)

    # Transform the data.
    X_train = sklearn_pca.fit_transform(X_train)

    # Transform test data.
    X_test = sklearn_pca.transform(X_test)

    # Perform logistic regression.
    logisticModel = LogisticRegression(
        fit_intercept=True, random_state=0, solver="liblinear"
    )
    logisticModel.fit(X_train, y_train)

    # Generate predictions.
    y_pred = logisticModel.predict(X_test)

    # Show model coefficients and intercept.
    print("\n*** Intercept: ")
    print(logisticModel.intercept_)

    print("\n*** Model Coefficients: ")
    print(logisticModel.coef_)

    # Show confusion matrix and accuracy scores.
    cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])

    print("\n*** Confusion Matrix")
    print(cm)

    # Show accuracy.
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:  " + str(accuracy))

    # For each X, calculate VIF and save in dataframe
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif = pd.DataFrame()
    vif["VIF Factor for Components"] = [
        variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])
    ]
    print(vif)


def main():
    ex12()


if __name__ == "__main__":
    main()
