# Exercise 1, 2, 3, 4
def ex1():
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    Data = {
        # 2 2 2 2 2 2 2 2 2 2
        "x": [
            25,
            34,
            22,
            27,
            33,
            33,
            31,
            22,
            35,
            34,  # 0 0 0 0 0 0 0 0 0 0
            67,
            54,
            57,
            43,
            50,
            57,
            59,
            52,
            65,
            47,  # 1 1 1 1 1 1 1 1 1 1
            49,
            48,
            35,
            33,
            44,
            45,
            38,
            43,
            51,
            46,
        ],
        # 2 2 2 2 2 2 2 2 2 2
        "y": [
            79,
            51,
            53,
            78,
            59,
            74,
            73,
            57,
            69,
            75,  # 0 0 0 0 0 0 0 0 0 0
            51,
            32,
            40,
            47,
            53,
            36,
            35,
            58,
            59,
            50,  # 1 1 1 1 1 1 1 1 1 1
            25,
            20,
            14,
            12,
            20,
            5,
            29,
            27,
            8,
            7,
        ],
    }

    # Perform clustering.
    df = DataFrame(Data, columns=["x", "y"])
    kmeans = KMeans(n_clusters=3, random_state=142).fit(df)
    centroids = kmeans.cluster_centers_

    # Show centroids.
    print("\n*** centroids ***")
    print(centroids)

    # Show sample labels.
    print("\n*** sample labels ***")
    print(kmeans.labels_)

    # Parameters: [c for color, s for dot size]
    plt.scatter(df["x"], df["y"], c=kmeans.labels_, s=50, alpha=0.5)

    # Shows the 3 centroids in red.
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=100, alpha=0.3)
    plt.show()


# Exercise 5, 6
def ex5():
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    Data = {
        # 2 2 2 2 2 2 2 2 2 2
        "x": [
            25,
            34,
            22,
            27,
            33,
            33,
            31,
            22,
            35,
            34,  # 0 0 0 0 0 0 0 0 0 0
            67,
            54,
            57,
            43,
            50,
            57,
            59,
            52,
            65,
            47,  # 1 1 1 1 1 1 1 1 1 1
            49,
            48,
            35,
            33,
            44,
            45,
            38,
            43,
            51,
            46,
        ],
        # 2 2 2 2 2 2 2 2 2 2
        "y": [
            79,
            51,
            53,
            78,
            59,
            74,
            73,
            57,
            69,
            75,  # 0 0 0 0 0 0 0 0 0 0
            51,
            32,
            40,
            47,
            53,
            36,
            35,
            58,
            59,
            50,  # 1 1 1 1 1 1 1 1 1 1
            25,
            20,
            14,
            12,
            20,
            5,
            29,
            27,
            8,
            7,
        ],
    }

    # Perform clustering.
    df = DataFrame(Data, columns=["x", "y"])
    kmeans = KMeans(n_clusters=4, random_state=142).fit(df)
    centroids = kmeans.cluster_centers_

    # Show centroids.
    print("\n*** centroids ***")
    print(centroids)

    # Show sample labels.
    print("\n*** sample labels ***")
    print(kmeans.labels_)

    # Parameters: [c for color, s for dot size]
    plt.scatter(df["x"], df["y"], c=kmeans.labels_, s=50, alpha=0.5)

    # Shows the 3 centroids in red.
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=100, alpha=0.3)
    plt.show()


# Exercise 7
def ex7():
    from pandas import DataFrame
    import math
    import numpy as np

    # Centroids and labels are already known so assign them manually.
    centroids = np.zeros((3, 2))
    centroids[0][0] = 55.1
    centroids[0][1] = 46.1
    centroids[1][0] = 43.2
    centroids[1][1] = 16.7
    centroids[2][0] = 29.6
    centroids[2][1] = 66.8

    clusterLabels = [0, 1, 2]

    # Receives X_test values and finds closest cluster for each.
    def predictCluster(centroids, X_test, clusterLabels):
        bestClusterList = []

        for i in range(0, len(X_test)):
            smallestDistance = None
            bestCluster = None

            # Compare each X value proximity with all centroids.
            for row in range(centroids.shape[0]):
                distance = 0

                # Get absolute distance between centroid and X.
                for col in range(centroids.shape[1]):
                    distance += math.sqrt(
                        (centroids[row][col] - X_test.iloc[i][col]) ** 2
                    )

                # Initialize bestCluster and smallestDistance during first iteration.
                # OR re-assign bestCluster if smaller distance to centroid found.
                if bestCluster == None or distance < smallestDistance:
                    bestCluster = clusterLabels[row]
                    smallestDistance = distance

            bestClusterList.append(bestCluster)

        return bestClusterList

    X_test = DataFrame({"x": [27, 40, 55], "y": [68, 12, 35]})
    predictions = predictCluster(centroids, X_test, clusterLabels)
    print(predictions)


# Exercise 8
def ex8():
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage
    from matplotlib import pyplot as plt

    X = np.array(
        [
            [5, 3],
            [10, 15],
            [15, 12],
            [24, 10],
            [30, 30],
            [85, 70],
            [71, 80],
            [60, 78],
            [70, 55],
            [80, 91],
        ]
    )

    # ---------------------------------------------
    # Draw data scatter with labels by each point.
    # ---------------------------------------------
    labels = range(1, 11)
    plt.scatter(X[:, 0], X[:, 1])

    # Add labels to points.
    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(-1, 2),  # position of text relative to point.
            textcoords="offset points",
            ha="right",
            va="bottom",
        )
    plt.show()

    # ---------------------------------------------
    # Draw dendrogram.
    # ---------------------------------------------
    from scipy.cluster.hierarchy import dendrogram, linkage

    linked = linkage(X, "single")
    labelList = range(1, 11)
    dendrogram(
        linked,
        orientation="top",
        labels=labelList,
        distance_sort="descending",
        show_leaf_counts=True,
    )
    plt.show()

    from sklearn.cluster import AgglomerativeClustering

    cluster = AgglomerativeClustering(
        n_clusters=3, affinity="euclidean", linkage="ward"
    )
    cluster.fit_predict(X)
    print(cluster.labels_)

    plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap="rainbow")
    plt.show()


# Exercise 9
def ex9():
    # Load libraries
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import matplotlib.pyplot as plt

    import pandas as pd

    # Load data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    iris["feature_names"]

    # Scale data.
    minmax = MinMaxScaler()
    data_scaled = minmax.fit_transform(X)
    data_scaled = pd.DataFrame(
        data_scaled,
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )
    print(data_scaled.head())

    # Draw dendrogram.
    import scipy.cluster.hierarchy as shc

    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(shc.linkage(data_scaled, method="ward"))
    plt.show()

    # Predict clusters.
    from sklearn.cluster import AgglomerativeClustering

    cluster = AgglomerativeClustering(
        n_clusters=3, affinity="euclidean", linkage="complete"
    )
    cluster.fit_predict(data_scaled)
    print(cluster.labels_)

    # Draw scatter.
    plt.figure(figsize=(10, 7))
    plt.scatter(
        data_scaled["sepal length (cm)"],
        data_scaled["sepal width (cm)"],
        c=cluster.labels_,
        alpha=0.5,
    )
    plt.xlabel("Sepal Length", fontsize=20)
    plt.ylabel("Sepal Width", fontsize=20)
    plt.show()


# Exercise 10
def ex10():
    # Load libraries
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import matplotlib.pyplot as plt

    import pandas as pd

    # Load data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    iris["feature_names"]

    # Scale data.
    minmax = MinMaxScaler()
    data_scaled = minmax.fit_transform(X)
    data_scaled = pd.DataFrame(
        data_scaled,
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )
    print(data_scaled.head())

    # Draw dendrogram.
    import scipy.cluster.hierarchy as shc

    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(shc.linkage(data_scaled, method="ward"))
    plt.show()

    # Predict clusters.
    from sklearn.cluster import AgglomerativeClustering

    cluster = AgglomerativeClustering(
        n_clusters=3, affinity="manhattan", linkage="complete"
    )
    cluster.fit_predict(data_scaled)
    print(cluster.labels_)

    # Draw scatter.
    plt.figure(figsize=(10, 7))
    plt.scatter(
        data_scaled["petal length (cm)"],
        data_scaled["petal width (cm)"],
        c=cluster.labels_,
        alpha=0.5,
    )
    plt.xlabel("Petal Length", fontsize=20)
    plt.ylabel("Petal Width", fontsize=20)
    plt.show()

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    nmi_result = normalized_mutual_info_score(y, cluster.labels_)
    ars_result = adjusted_rand_score(y, cluster.labels_)
    print("\n*** Normalized mutual information score: " + str(nmi_result))
    print("*** Adjusted rand score: " + str(ars_result))


# Exercise 11
def ex11():
    # Load libraries
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import matplotlib.pyplot as plt

    import pandas as pd

    # Load data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    iris["feature_names"]
    df = pd.DataFrame(X, columns=iris["feature_names"])
    print(df)

    # Plot the data
    plt.figure(figsize=(6, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.xlabel("Iris X axis")
    plt.ylabel("Iris Y axis")
    plt.title("Visualization of raw data")
    plt.show()

    # Scale data.
    standard = StandardScaler()
    X_std = standard.fit_transform(X)

    # Run local implementation of kmeans
    km = KMeans(n_clusters=2, max_iter=100)
    km.fit(X_std)
    centroids = km.cluster_centers_

    data_scaled = pd.DataFrame(
        X_std,
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )
    print(data_scaled.head())

    # Plot the clustered data
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(
        X_std[km.labels_ == 0, 0],
        X_std[km.labels_ == 0, 1],
        c="green",
        label="cluster 1",
    )
    plt.scatter(
        X_std[km.labels_ == 1, 0],
        X_std[km.labels_ == 1, 1],
        c="blue",
        label="cluster 2",
    )
    plt.scatter(
        centroids[:, 0], centroids[:, 1], marker="*", s=300, c="r", label="centroid"
    )
    plt.legend()
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("Iris X axis")
    plt.ylabel("Iris Y axis")
    plt.title("Visualization of clustered data", fontweight="bold")
    ax.set_aspect("equal")
    plt.show()

    # Run the Kmeans algorithm and get the index of data points clusters
    sse = []
    list_k = list(range(1, 10))

    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(X_std)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, "-o")
    plt.xlabel(r"Number of clusters *k*")
    plt.ylabel("Sum of squared distance")
    plt.show()


# Exercise 12
def ex12():
    # Load libraries
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import matplotlib.pyplot as plt

    import pandas as pd

    # Load data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    iris["feature_names"]
    df = pd.DataFrame(X, columns=iris["feature_names"])
    print(df)

    # Plot the data
    plt.figure(figsize=(6, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.xlabel("Iris X axis")
    plt.ylabel("Iris Y axis")
    plt.title("Visualization of raw data")
    plt.show()

    # Scale data.
    standard = StandardScaler()
    X_std = standard.fit_transform(X)

    # Run local implementation of kmeans
    km = KMeans(n_clusters=2, max_iter=100)
    km.fit(X_std)
    centroids = km.cluster_centers_

    data_scaled = pd.DataFrame(
        X_std,
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )
    print(data_scaled.head())

    # Plot the clustered data
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(
        X_std[km.labels_ == 0, 0],
        X_std[km.labels_ == 0, 1],
        c="green",
        label="cluster 1",
    )
    plt.scatter(
        X_std[km.labels_ == 1, 0],
        X_std[km.labels_ == 1, 1],
        c="blue",
        label="cluster 2",
    )
    plt.scatter(
        centroids[:, 0], centroids[:, 1], marker="*", s=300, c="r", label="centroid"
    )
    plt.legend()
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("Iris X axis")
    plt.ylabel("Iris Y axis")
    plt.title("Visualization of clustered data", fontweight="bold")
    ax.set_aspect("equal")
    plt.show()

    # Run the Kmeans algorithm and get the index of data points clusters
    sse = []
    list_k = list(range(1, 10))

    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(X_std)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, "-o")
    plt.xlabel(r"Number of clusters *k*")
    plt.ylabel("Sum of squared distance")
    plt.show()

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    nmi_result = normalized_mutual_info_score(y, km.labels_)
    ars_result = adjusted_rand_score(y, km.labels_)
    print("\n*** Normalized mutual information score: " + str(nmi_result))
    print("*** Adjusted rand score: " + str(ars_result))


# Exercise 13
def ex13():
    import pandas as pd
    import os

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Import data into a DataFrame.
    path = f"{os.environ['DATASET_DIRECTORY']}babysamp-98.txt"
    df = pd.read_table(path, delim_whitespace=True)
    numericColumns = [
        "MomAge",
        "DadAge",
        "MomEduc",
        "MomMarital",
        "numlive",
        "dobmm",
        "gestation",
        "weight",
        "prenatalstart",
    ]

    dfNumeric = df[numericColumns]

    # Show data types for each column.
    print("\n*** Before imputing")
    print(df.describe())
    print(dfNumeric.head(11))

    # Show summaries for objects like dates and strings.
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=5)
    dfNumeric = pd.DataFrame(
        imputer.fit_transform(dfNumeric), columns=dfNumeric.columns
    )

    # Show data types for each column.
    print("\n*** After imputing")
    print(dfNumeric.describe())
    print(dfNumeric.head(11))


# Exercise 14
def ex14():
    import pandas as pd
    import os

    # Import data into a DataFrame.
    df = pd.read_csv(
        f"{os.environ['DATASET_DIRECTORY']}titanic_training_data.csv",
        skiprows=1,  # Don't include header row as part of data.
        encoding="ISO-8859-1",
        sep=",",
        names=(
            "PassengerId",
            "Survived",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
        ),
    )
    dfNumeric = df[
        [
            "PassengerId",
            "Survived",
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
        ]
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Show data types for each column.
    print("\n*** Before imputing")
    print(df.describe())
    print(dfNumeric.head(11))

    # Show summaries for objects like dates and strings.
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=5)
    dfNumeric = pd.DataFrame(
        imputer.fit_transform(dfNumeric), columns=dfNumeric.columns
    )

    # Show data types for each column.
    print("\n*** After imputing")
    print(dfNumeric.describe())
    print(dfNumeric.head(11))


def main():
    # ex1()
    # ex5()
    # ex7()
    # ex8()
    # ex9()
    # ex10()
    # ex11()
    # ex12()
    # ex13()
    ex14()


if __name__ == "__main__":
    main()
