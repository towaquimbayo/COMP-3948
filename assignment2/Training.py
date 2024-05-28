from enum import Enum

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import statsmodels.api as sm
import numpy as np
from imblearn.over_sampling import SMOTE
import pickle


# TODO: Please change the path to the CustomerChurn.csv file.
def get_data():
    return pd.read_csv(
        "/Users/elber/Documents/Data Sets/CustomerChurn.csv",
        skiprows=1,
        encoding="ISO-8859-1",
        sep=",",
        names=(
            "AccountAge",
            "MonthlyCharges",
            "TotalCharges",
            "SubscriptionType",
            "PaymentMethod",
            "PaperlessBilling",
            "ContentType",
            "MultiDeviceAccess",
            "DeviceRegistered",
            "ViewingHoursPerWeek",
            "AverageViewingDuration",
            "ContentDownloadsPerMonth",
            "GenrePreference",
            "UserRating",
            "SupportTicketsPerMonth",
            "Gender",
            "WatchlistSize",
            "ParentalControl",
            "SubtitlesEnabled",
            "CustomerID",
            "Churn",
        ),
    )


def display_boxplot(df, feature_name):
    # Grouped box plots with a 0 target box showing the feature range  when the target is 0.
    # Then beside it show the 1 target box which shows the feature range when the target is 1.
    sns.boxplot(
        x="Churn",
        y=feature_name,
        data=df,
        palette="Set2",
    )
    plt.title(f"Grouped Box Plot of {feature_name} vs Churn")
    plt.show()


def display_freq_dist(df, feature_name):
    # Frequency histogram of feature.
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(
        x=feature_name,
        hue="Churn",
        data=df,
        palette="Set2",
        element="step",
        stat="count",
        common_norm=False,
    )

    plt.title(f"Histogram of {feature_name} for Churn 0 and 1", fontsize=18)
    plt.xlabel(feature_name, fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    legend_labels = [
        mpatches.Patch(color=sns.color_palette("Set2")[0], label="Churn 0"),
        mpatches.Patch(color=sns.color_palette("Set2")[1], label="Churn 1"),
    ]
    ax.legend(handles=legend_labels, fontsize=16)

    plt.tight_layout()
    plt.show()


def display_histograms(df):
    # Plot histogram of all columns.
    df.hist(bins=50, figsize=(20, 15))
    plt.show()


def display_scatter_matrix(df):
    # Scatter plot of all columns.
    scatter_matrix(df, figsize=(20, 15))
    plt.title("Scatter Matrix of Key Features vs Churn")
    plt.tight_layout()
    plt.show()


def display_heatmap(df):
    # Single Column Heatmap for Churn column.
    plt.figure(figsize=(12, 7))
    sns.heatmap(
        df.corr()[["Churn"]],
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Churn Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def convert_na_cells_to_num(col_name, df, measure_type):
    # Create two new column names based on original column name.
    indicator_col_name = "m_" + col_name  # Tracks whether imputed.
    imputed_col_name = "imp_" + col_name  # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputed_value = 0
    if measure_type == "median":
        imputed_value = df[col_name].median()
    elif measure_type == "mode":
        imputed_value = float(df[col_name].mode())
    else:
        imputed_value = df[col_name].mean()

    # Populate new columns with data.
    imputed_column = []
    indicator_column = []
    for i in range(len(df)):
        is_imputed = False

        # mi_OriginalName column stores imputed & original data.
        if np.isnan(df.loc[i][col_name]):
            is_imputed = True
            imputed_column.append(imputed_value)
        else:
            imputed_column.append(df.loc[i][col_name])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if is_imputed:
            indicator_column.append(1)
        else:
            indicator_column.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicator_col_name] = indicator_column
    df[imputed_col_name] = imputed_column
    return df


# Cross fold validation to get average accuracy, precision, recall, and F1-score.
def cross_fold_validation(x, y, solver, threshold):
    k_fold = KFold(n_splits=5, shuffle=True)
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list_adjusted = []
    precision_list_adjusted = []
    recall_list_adjusted = []
    f1_list_adjusted = []
    for train_index, test_index in k_fold.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LogisticRegression(
            fit_intercept=True, solver=solver, C=1, max_iter=2000
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        predictions_adjusted = (model.predict_proba(x_test)[:, 1] >= threshold).astype(
            int
        )
        accuracy_list.append(accuracy_score(y_test, predictions))
        precision_list.append(precision_score(y_test, predictions))
        recall_list.append(recall_score(y_test, predictions))
        f1_list.append(f1_score(y_test, predictions))
        accuracy_list_adjusted.append(accuracy_score(y_test, predictions_adjusted))
        precision_list_adjusted.append(precision_score(y_test, predictions_adjusted))
        recall_list_adjusted.append(recall_score(y_test, predictions_adjusted))
        f1_list_adjusted.append(f1_score(y_test, predictions_adjusted))

    print(f"Average Accuracy: {np.mean(accuracy_list)}")
    print(f"Standard Deviation of Accuracy: {np.std(accuracy_list)}")
    print(f"Average Precision: {np.mean(precision_list)}")
    print(f"Standard Deviation of Precision: {np.std(precision_list)}")
    print(f"Average Recall: {np.mean(recall_list)}")
    print(f"Standard Deviation of Recall: {np.std(recall_list)}")
    print(f"Average F1-score: {np.mean(f1_list)}")
    print(f"Standard Deviation of F1-score: {np.std(f1_list)}")
    print(f"\nAdjusted Threshold ({threshold}):")
    print(f"Average Accuracy: {np.mean(accuracy_list_adjusted)}")
    print(f"Standard Deviation of Accuracy: {np.std(accuracy_list_adjusted)}")
    print(f"Average Precision: {np.mean(precision_list_adjusted)}")
    print(f"Standard Deviation of Precision: {np.std(precision_list_adjusted)}")
    print(f"Average Recall: {np.mean(recall_list_adjusted)}")
    print(f"Standard Deviation of Recall: {np.std(recall_list_adjusted)}")
    print(f"Average F1-score: {np.mean(f1_list_adjusted)}")
    print(f"Standard Deviation of F1-score: {np.std(f1_list_adjusted)}")


def chi_squared_feature_selection(x, y, num_features, print_results):
    test = SelectKBest(score_func=chi2, k=num_features)
    chi_scores = test.fit(x, y)
    np.set_printoptions(precision=3)
    cols = chi_scores.get_support(indices=True)
    feature_names = x.columns[cols].tolist()

    if print_results:
        print("Predictor Chi-Square Scores: " + str(chi_scores.scores_))
        print(cols)
        print(f"Top {num_features} Chi-Square Features:\n{', '.join(feature_names)}")
    return feature_names


def forward_feature_selection(x, y, num_features, print_results):
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1000)
    ffs = f_regression(x, y)
    features_df = pd.DataFrame()
    for i in range(0, len(x.columns)):
        features_df = features_df._append(
            {"feature": x.columns[i], "ffs": ffs[0][i]}, ignore_index=True
        )
    features_df = features_df.drop([0])  # Drop const
    features_df = features_df.sort_values(by=["ffs"], ascending=False)
    feature_names = features_df["feature"].head(num_features).tolist()

    if print_results:
        print(f"Predictor F-Regression Scores:\n{features_df}")
        print(
            f"\nTop {num_features} Forward Feature Selection Features:\n{', '.join(feature_names)}"
        )
    return feature_names


def rfe_feature_selection(x, y, num_features, print_results):
    model = LogisticRegression(fit_intercept=True, solver="liblinear")
    if "const" in x.columns:
        x = x.drop("const", axis=1)

    rfe = RFE(model, n_features_to_select=num_features, step=5)
    rfe = rfe.fit(x, y)
    feature_names = x.columns[rfe.support_].tolist()

    if print_results:
        print(f"Predictor RFE Scores:\n{rfe.ranking_}")
        print(f"\nTop {num_features} RFE Features:\n{', '.join(feature_names)}")
    return feature_names


class FeatureSelectionNames(Enum):
    CHI_SQUARED = "chi-squared"
    FORWARD = "forward"
    RFE = "rfe"


def main():
    df = get_data()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # TODO: Change feature selection name to chi-squared, forward, or rfe.
    feature_selection_name = FeatureSelectionNames.FORWARD.value
    if feature_selection_name not in [
        FeatureSelectionNames.CHI_SQUARED.value,
        FeatureSelectionNames.FORWARD.value,
        FeatureSelectionNames.RFE.value,
    ]:
        print("Invalid feature selection name.")
        return

    # Get first 5000 rows of data.
    # df = df.head(5000)

    print(df)
    print(df.describe().round(2))
    print(df.info())

    # Check which columns have NA values.
    na_columns_list = []
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if col == "CustomerID":
                continue
            na_columns_list.append(col)

    # Imputation by converting NA cells to median, mode, or mean.
    for col in na_columns_list:
        print(f"Imputing Column: {col}")
        if col in [
            "AccountAge",
            "MonthlyCharges",
            "ViewingHoursPerWeek",
            "AverageViewingDuration",
            "UserRating",
            "ContentDownloadsPerMonth",
            "SupportTicketsPerMonth",
            "WatchlistSize",
        ]:
            df = convert_na_cells_to_num(col, df, "mean")
        elif col in ["TotalCharges"]:
            df = convert_na_cells_to_num(col, df, "median")
        else:
            df = convert_na_cells_to_num(col, df, "mode")

    # Generate dummy variables for categorical features.
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "SubscriptionType",
                        "PaymentMethod",
                        "PaperlessBilling",
                        "ContentType",
                        "MultiDeviceAccess",
                        "DeviceRegistered",
                        "GenrePreference",
                        "Gender",
                        "ParentalControl",
                        "SubtitlesEnabled",
                    ]
                ],
                columns=[
                    "SubscriptionType",
                    "PaymentMethod",
                    "PaperlessBilling",
                    "ContentType",
                    "MultiDeviceAccess",
                    "DeviceRegistered",
                    "GenrePreference",
                    "Gender",
                    "ParentalControl",
                    "SubtitlesEnabled",
                ],
            ).astype(int),
        ],
        axis=1,
    )  # Join dummy df with original df

    # Create bins
    df["AccountAgeBin"] = pd.cut(
        x=df["imp_AccountAge"],
        bins=[0, 30, 60, 90, 120],
    )
    df["MonthlyChargesBin"] = pd.cut(
        x=df["MonthlyCharges"],
        bins=[0, 5, 10, 15, 20, 25],
    )
    df["TotalChargesBin"] = pd.cut(
        x=df["TotalCharges"],
        bins=[0, 500, 1000, 1500, 2000, 2500],
    )
    df["ViewingHoursPerWeekBin"] = pd.cut(
        x=df["imp_ViewingHoursPerWeek"],
        bins=[0, 10, 20, 30, 40, 50],
    )
    df["AverageViewingDurationBin"] = pd.cut(
        x=df["imp_AverageViewingDuration"],
        bins=[0, 50, 100, 150, 200],
    )
    df["ContentDownloadsPerMonthBin"] = pd.cut(
        x=df["ContentDownloadsPerMonth"],
        bins=[0, 10, 20, 30, 40, 50],
    )
    df["UserRatingBin"] = pd.cut(
        x=df["UserRating"],
        bins=[0, 2, 4, 6],
    )
    df["SupportTicketsPerMonthBin"] = pd.cut(
        x=df["SupportTicketsPerMonth"],
        bins=[0, 2, 4, 6, 8, 10],
    )
    df["WatchlistSizeBin"] = pd.cut(
        x=df["WatchlistSize"],
        bins=[0, 5, 10, 15, 20, 25],
    )

    # Generate dummy variables for binned features.
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "AccountAgeBin",
                        "MonthlyChargesBin",
                        "TotalChargesBin",
                        "ViewingHoursPerWeekBin",
                        "AverageViewingDurationBin",
                        "ContentDownloadsPerMonthBin",
                        "UserRatingBin",
                        "SupportTicketsPerMonthBin",
                        "WatchlistSizeBin",
                    ]
                ],
                columns=[
                    "AccountAgeBin",
                    "MonthlyChargesBin",
                    "TotalChargesBin",
                    "ViewingHoursPerWeekBin",
                    "AverageViewingDurationBin",
                    "ContentDownloadsPerMonthBin",
                    "UserRatingBin",
                    "SupportTicketsPerMonthBin",
                    "WatchlistSizeBin",
                ],
            ).astype(int),
        ],
        axis=1,
    )  # Join dummies with original df

    print("\nAfter imputation:")
    print(df)
    print(df.describe().round(2))
    print(df.info())

    # Used for testing only. Contains all features.
    x = df[
        [
            # "AccountAge",  # imputed
            "MonthlyCharges",  # binned
            "TotalCharges",  # binned
            # "SubscriptionType",  # dummy
            # "PaymentMethod",  # dummy
            # "PaperlessBilling",  # dummy
            # "ContentType",  # dummy
            # "MultiDeviceAccess",  # dummy
            # "DeviceRegistered",  # dummy
            # "ViewingHoursPerWeek",  # imputed
            # "AverageViewingDuration",  # imputed
            "ContentDownloadsPerMonth",  # binned
            # "GenrePreference",  # dummy
            "UserRating",  # binned
            "SupportTicketsPerMonth",  # binned
            # "Gender",  # dummy
            "WatchlistSize",  # binned
            # "ParentalControl",  # dummy
            # "SubtitlesEnabled",  # dummy
            # "CustomerID",  # insignificant
            "m_AccountAge",
            "imp_AccountAge",  # binned
            "m_ViewingHoursPerWeek",
            "imp_ViewingHoursPerWeek",  # binned
            "m_AverageViewingDuration",
            "imp_AverageViewingDuration",  # binned
            "SubscriptionType_Basic",
            "SubscriptionType_Premium",
            "SubscriptionType_Standard",
            "PaymentMethod_Bank transfer",
            "PaymentMethod_Credit card",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check",
            "PaperlessBilling_No",
            "PaperlessBilling_Yes",
            "ContentType_Both",
            "ContentType_Movies",
            "ContentType_TV Shows",
            "MultiDeviceAccess_No",
            "MultiDeviceAccess_Yes",
            "DeviceRegistered_Computer",
            "DeviceRegistered_Mobile",
            "DeviceRegistered_TV",
            "DeviceRegistered_Tablet",
            "GenrePreference_Action",
            "GenrePreference_Comedy",
            "GenrePreference_Drama",
            "GenrePreference_Fantasy",
            "GenrePreference_Sci-Fi",
            "Gender_Female",
            "Gender_Male",
            "ParentalControl_No",
            "ParentalControl_Yes",
            "SubtitlesEnabled_No",
            "SubtitlesEnabled_Yes",
            "AccountAgeBin_(0, 30]",
            "AccountAgeBin_(30, 60]",
            "AccountAgeBin_(60, 90]",
            "AccountAgeBin_(90, 120]",
            "MonthlyChargesBin_(0, 5]",
            "MonthlyChargesBin_(5, 10]",
            "MonthlyChargesBin_(10, 15]",
            "MonthlyChargesBin_(15, 20]",
            "MonthlyChargesBin_(20, 25]",
            "TotalChargesBin_(0, 500]",
            "TotalChargesBin_(500, 1000]",
            "TotalChargesBin_(1000, 1500]",
            "TotalChargesBin_(1500, 2000]",
            "TotalChargesBin_(2000, 2500]",
            "ViewingHoursPerWeekBin_(0, 10]",
            "ViewingHoursPerWeekBin_(10, 20]",
            "ViewingHoursPerWeekBin_(20, 30]",
            "ViewingHoursPerWeekBin_(30, 40]",
            "ViewingHoursPerWeekBin_(40, 50]",
            "AverageViewingDurationBin_(0, 50]",
            "AverageViewingDurationBin_(50, 100]",
            "AverageViewingDurationBin_(100, 150]",
            "AverageViewingDurationBin_(150, 200]",
            "ContentDownloadsPerMonthBin_(0, 10]",
            "ContentDownloadsPerMonthBin_(10, 20]",
            "ContentDownloadsPerMonthBin_(20, 30]",
            "ContentDownloadsPerMonthBin_(30, 40]",
            "ContentDownloadsPerMonthBin_(40, 50]",
            "UserRatingBin_(0, 2]",
            "UserRatingBin_(2, 4]",
            "UserRatingBin_(4, 6]",
            "SupportTicketsPerMonthBin_(0, 2]",
            "SupportTicketsPerMonthBin_(2, 4]",
            "SupportTicketsPerMonthBin_(4, 6]",
            "SupportTicketsPerMonthBin_(6, 8]",
            "SupportTicketsPerMonthBin_(8, 10]",
            "WatchlistSizeBin_(0, 5]",
            "WatchlistSizeBin_(5, 10]",
            "WatchlistSizeBin_(10, 15]",
            "WatchlistSizeBin_(15, 20]",
            "WatchlistSizeBin_(20, 25]",
        ]
    ]

    # The top 10 best features from forward selection used for production.
    x = df[
        [
            "imp_AccountAge",
            "SubscriptionType_Premium",
            "PaymentMethod_Credit card",
            "SubtitlesEnabled_Yes",
            "SupportTicketsPerMonthBin_(0, 2]",
            "UserRatingBin_(0, 2]",
            "Gender_Female",
            "ContentType_TV Shows",
            "ContentType_Movies",
            "ParentalControl_Yes",
        ]
    ]
    x = sm.add_constant(x)
    y = df["Churn"]

    # Display box plots and frequency distributions for features.
    # for feature in x.columns:
    #     if feature == "const":
    #         continue
    #     display_boxplot(df, feature)
    #     display_freq_dist(df, feature)
    # display_histograms(x)
    # display_scatter_matrix(
    #     df[
    #         [
    #             "imp_AccountAge",
    #             "SubscriptionType_Premium",
    #             "PaymentMethod_Credit card",
    #             "SubtitlesEnabled_Yes",
    #             "SupportTicketsPerMonthBin_(0, 2]",
    #             "UserRatingBin_(0, 2]",
    #             "Gender_Female",
    #             "ContentType_TV Shows",
    #             "ContentType_Movies",
    #             "ParentalControl_Yes",
    #             "Churn",
    #         ]
    #     ]
    # )
    # display_heatmap(
    #     df[
    #         [
    #             "imp_AccountAge",
    #             "SubscriptionType_Premium",
    #             "PaymentMethod_Credit card",
    #             "SubtitlesEnabled_Yes",
    #             "SupportTicketsPerMonthBin_(0, 2]",
    #             "UserRatingBin_(0, 2]",
    #             "Gender_Female",
    #             "ContentType_TV Shows",
    #             "ContentType_Movies",
    #             "ParentalControl_Yes",
    #             "Churn",
    #         ]
    #     ]
    # )

    # SMOTE to balance classes
    smote = SMOTE()
    x, y = smote.fit_resample(x, y)

    # Select features using chi-squared, forward selection, or RFE by uncommenting.
    features = []
    if feature_selection_name == FeatureSelectionNames.CHI_SQUARED.value:
        features = chi_squared_feature_selection(x, y, 10, True)
    elif feature_selection_name == FeatureSelectionNames.FORWARD.value:
        features = forward_feature_selection(x, y, 10, True)
    elif feature_selection_name == FeatureSelectionNames.RFE.value:
        features = rfe_feature_selection(x, y, 10, True)
    x = x[features]

    # Cross fold validation to get average accuracy, precision, recall, and F1-score.
    print("\nCross fold validation:")
    if feature_selection_name == FeatureSelectionNames.CHI_SQUARED.value:
        cross_fold_validation(x, y, "liblinear", 0.4)
    elif feature_selection_name == FeatureSelectionNames.FORWARD.value:
        cross_fold_validation(x, y, "liblinear", 0.4)
    elif feature_selection_name == FeatureSelectionNames.RFE.value:
        cross_fold_validation(x, y, "sag", 0.25)

    # Create logistic regression model.
    # use liblinear for chi-squared, sag for forward selection and RFE
    solver = "liblinear"
    if feature_selection_name == FeatureSelectionNames.RFE.value:
        solver = "sag"
    model = LogisticRegression(fit_intercept=True, solver=solver, C=1, max_iter=2000)

    # Create training set with 80% of data and test set with 20% of data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    # Train model and make predictions.
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()

    print("\nModel:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"True Positive:\t{tp}")
    print(f"False Positive:\t{fp}")
    print(f"True Negative:\t{tn}")
    print(f"False Negative:\t{fn}")
    print(f"True Positive Rate:\t{tp / (tp + fn)}")
    print(f"False Positive Rate:\t{fp / (fp + tn)}")

    print("\nMetrics:")
    print(f"Accuracy:\t{accuracy_score(y_test, predictions)}")
    print(f"Precision:\t{precision_score(y_test, predictions)}")
    print(f"Recall:\t{recall_score(y_test, predictions)}")
    print(f"F1-score:\t{f1_score(y_test, predictions)}")

    # Save model as pickle file.
    file_handler = open("best_model.pkl", "wb")
    pickle.dump(model, file_handler)
    file_handler.close()

    # Load model from pickle file.
    file_handler = open("best_model.pkl", "rb")
    loaded_model = pickle.load(file_handler)
    pickle_predictions = loaded_model.predict(x_test)
    print(f"\nPickled Predictions:\n{pickle_predictions}")
    file_handler.close()


if __name__ == "__main__":
    main()
