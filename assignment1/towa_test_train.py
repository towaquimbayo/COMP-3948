import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import statsmodels.api as sm
import numpy as np


def get_data():
    return pd.read_csv(
        "/Users/elber/Documents/Data Sets/loan_v2.csv",
        skiprows=1,
        encoding="ISO-8859-1",
        sep=",",
        names=(
            "Gender",
            "Age",
            "Income (USD)",
            "Income Stability",
            "Profession",
            "Type of Employment",
            "Location",
            "Loan Amount Request (USD)",
            "Current Loan Expenses (USD)",
            "Expense Type 1",
            "Expense Type 2",
            "Dependents",
            "Credit Score",
            "No. of Defaults",
            "Has Active Credit Card",
            "Property ID",
            "Property Age",
            "Property Type",
            "Property Location",
            "Co-Applicant",
            "Property Price",
            "Loan Sanction Amount (USD)",
        ),
    )


def display_histograms(df):
    # Plot histogram of all columns.
    df.hist(bins=50, figsize=(20, 15))
    plt.show()


def display_scatter_matrix(df):
    # Scatter plot of all columns.
    scatter_matrix(df, figsize=(20, 15))
    plt.show()


def display_heatmap(df):
    # Heatmap of all columns.
    plt.figure(figsize=(20, 15))
    corr = df.corr()
    sns.heatmap(
        corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True,
    )
    plt.show()


def plot_prediction_vs_actual(title, y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted (Y) vs. Actual (X): " + title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")


def plot_residuals_vs_actual(title, y_test, predictions):
    residuals = y_test - predictions
    plt.scatter(y_test, residuals, label="Residuals vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title("Error Residuals (Y) vs. Actual (X): " + title)
    plt.plot([y_test.min(), y_test.max()], [0, 0], "k--")


def plot_residual_histogram(title, y_test, predictions, bins):
    residuals = y_test - predictions
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.hist(residuals, label="Residuals vs Actual", bins=bins)
    plt.title("Error Residual Frequency: " + title)


def draw_validation_plots(title, bins, y_test, predictions):
    # Define number of rows and columns for graph display.
    plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    plt.subplot(1, 3, 1)  # Specify total rows, columns and image #
    plot_prediction_vs_actual(title, y_test, predictions)

    plt.subplot(1, 3, 2)  # Specify total rows, columns and image #
    plot_residuals_vs_actual(title, y_test, predictions)

    plt.subplot(1, 3, 3)  # Specify total rows, columns and image #
    plot_residual_histogram(title, y_test, predictions, bins)
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


def view_and_get_outliers(df, col_name, threshold):
    # Show basic statistics.
    df_sub = df[[col_name]]
    print("*** Statistics for " + col_name)
    print(df_sub.describe())

    # Show boxplot.
    df_sub.boxplot(column=[col_name])
    plt.title(col_name)
    plt.show()

    # Note this is absolute 'abs' so it gets both high and low values.
    z = np.abs(stats.zscore(df_sub))
    row_column_array = np.where(z > threshold)
    row_indices = row_column_array[0]

    # Show outlier rows.
    print("\nOutlier row indexes for " + col_name + ":")
    print(row_indices)
    print("")

    # Show filtered and sorted DataFrame with outliers.
    df_sub = df.iloc[row_indices]
    print("\nDataFrame rows containing outliers for " + col_name + ":")
    print(df_sub.sort_values([col_name], ascending=[True]))
    return row_indices


def cross_fold_validation(x, y):
    k_fold = KFold(n_splits=5, shuffle=True)
    rmse_list = []
    for train_index, test_index in k_fold.split(x):
        # use index lists to isolate rows for train and test sets.
        # Get rows filtered by index and all columns.
        # X.loc[row number array, all columns]
        x_train = x.loc[x.index.intersection(train_index), :]
        x_test = x.loc[x.index.intersection(test_index), :]
        y_train = y.loc[y.index.intersection(train_index), :]
        y_test = y.loc[y.index.intersection(test_index), :]

        # Build linear regression model.
        model = sm.OLS(y_train, x_train).fit()
        predictions = model.predict(x_test)  # make the predictions by the model
        rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
        print(model.summary(title="Model C: Loan Sanction Amount (USD)"))
        print(f"\nRoot Mean Squared Error: {str(rmse)}")
        rmse_list.append(rmse)

    print("Average rmse: " + str(np.mean(rmse_list)))


def plot_target_vs_avg_continuous_data(df, feature):
    df = df[[feature, "Loan Sanction Amount (USD)"]]
    # Generate grouping column by percentile
    df["group"] = pd.qcut(df["Loan Sanction Amount (USD)"], q=3)
    df = pd.get_dummies(df, columns=["group"])
    print(df)

    dfLow = df[df["group_(-999.001, 17637.567]"] == 1]
    dfMid = df[df["group_(17637.567, 59188.64]"] == 1]
    dfHigh = df[df["group_(59188.64, 481907.32]"] == 1]

    avgTargetLow = dfLow["Loan Sanction Amount (USD)"].mean()
    avgTargetMid = dfMid["Loan Sanction Amount (USD)"].mean()
    avgTargetHigh = dfHigh["Loan Sanction Amount (USD)"].mean()

    avgFeatureLow = dfLow[feature].mean()
    avgFeatureMid = dfMid[feature].mean()
    avgFeatureHigh = dfHigh[feature].mean()

    avgTargetHigh = avgTargetHigh.round(2)
    avgTargetMid = avgTargetMid.round(2)
    avgTargetHigh = avgTargetHigh.round(2)

    plt.bar(
        [avgTargetLow, avgTargetMid, avgTargetHigh],
        [avgFeatureLow, avgFeatureMid, avgFeatureHigh],
    )
    plt.title(f"{feature} vs. Loan Sanction Amount (USD)")
    plt.ylabel(f"Feature Avg ({feature})")
    plt.xticks(
        ticks=[avgTargetLow, avgTargetMid, avgTargetHigh],
        labels=[avgTargetLow, avgTargetMid, avgTargetHigh],
        rotation=70,
    )
    plt.xlabel("Target %tile group averages for Loan Sanction Amount (USD)")
    plt.tight_layout()
    plt.show()


# imputation by converting NA cells to median, mode, or mean
def model_a():
    df = get_data()

    print(df)
    print(df.describe().round(2))
    print(df.info())

    # Imputation by converting NA cells to median, mode, or mean.
    df = convert_na_cells_to_num("Income (USD)", df, "mean")
    df = convert_na_cells_to_num("Current Loan Expenses (USD)", df, "median")
    df = convert_na_cells_to_num("Dependents", df, "mode")
    df = convert_na_cells_to_num("Credit Score", df, "mean")
    df = convert_na_cells_to_num("Property Age", df, "median")

    print("\nAfter imputation:")
    print(df)
    print(df.describe().round(2))
    print(df.info())

    x = df[
        [
            # "Gender",  # non-numeric
            "Age",
            # "Income (USD)",  # imputed
            # "Income Stability",  # non-numeric
            # "Profession",  # non-numeric
            # "Type of Employment",  # non-numeric
            # "Location",  # non-numeric
            "Loan Amount Request (USD)",
            # "Current Loan Expenses (USD)",  # imputed
            # "Expense Type 1",  # non-numeric
            # "Expense Type 2",  # non-numeric
            # "Dependents",  # imputed
            # "Credit Score",  # imputed
            # "No. of Defaults",  # insignificant
            # "Has Active Credit Card",  # non-numeric
            # "Property ID",  # insignificant
            # "Property Age",  # imputed
            # "Property Type",  # insignificant
            # "Property Location",  # non-numeric
            # "Co-Applicant",  # insignificant
            # "Property Price",  # insignificant
            # "m_Income (USD)",  # insignificant
            # "imp_Income (USD)",  # insignificant
            # "m_Current Loan Expenses (USD)",  # insignificant
            # "imp_Current Loan Expenses (USD)",  # insignificant
            "m_Dependents",
            # "imp_Dependents",  # insignificant
            "m_Credit Score",
            "imp_Credit Score",
            "m_Property Age",
            # "imp_Property Age",  # insignificant
        ]
    ]
    x = sm.add_constant(x)
    y = df[["Loan Sanction Amount (USD)"]]

    # Cross fold validation
    print("\nCross fold validation:")
    cross_fold_validation(x, y)

    y = df["Loan Sanction Amount (USD)"]  # reset y

    # Create training set with 80% of data and test set with 20% of data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    # Create linear regression model.
    model = sm.OLS(y_train, x_train).fit()
    predictions = model.predict(x_test)  # make the predictions by the model

    print(model.summary(title="Model A: Loan Sanction Amount (USD)"))
    print(
        "Root Mean Squared Error:",
        np.sqrt(metrics.mean_squared_error(y_test, predictions)),
    )

    # Draw validation plots on the best model.
    draw_validation_plots(
        title="Loan Sanction Amount (USD)",
        bins=8,
        y_test=y_test,
        predictions=predictions,
    )


# imputation by converting NA cells to median, mode, or mean
# includes binning and dummy variables
def model_b():
    df = get_data()

    print(df)
    print(df.describe().round(2))
    print(df.info())

    # Imputation by converting NA cells to median, mode, or mean.
    df = convert_na_cells_to_num("Income (USD)", df, "mean")
    df = convert_na_cells_to_num("Current Loan Expenses (USD)", df, "median")
    df = convert_na_cells_to_num("Dependents", df, "mode")
    df = convert_na_cells_to_num("Credit Score", df, "mean")
    df = convert_na_cells_to_num("Property Age", df, "median")

    # Generate dummies variables
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "Gender",
                        "Income Stability",
                        "Profession",
                        "Type of Employment",
                        "Location",
                        "Expense Type 1",
                        "Expense Type 2",
                        "No. of Defaults",
                        "Has Active Credit Card",
                        "Property Type",
                        "Property Location",
                    ]
                ],
                columns=[
                    "Gender",
                    "Income Stability",
                    "Profession",
                    "Type of Employment",
                    "Location",
                    "Expense Type 1",
                    "Expense Type 2",
                    "No. of Defaults",
                    "Has Active Credit Card",
                    "Property Type",
                    "Property Location",
                ],
            ).astype(int),
        ],
        axis=1,
    )  # Join dummy df with original df

    # Create bins.
    df["ageBin"] = pd.cut(
        x=df["Age"],
        bins=[10, 20, 30, 40, 50, 60, 70],
    )
    df["incomeBin"] = pd.cut(
        x=df["imp_Income (USD)"],
        bins=[0, 1000, 2000, 3000, 4000, 5000, 1778000],
    )
    df["loanAmountRequestBin"] = pd.cut(
        x=df["Loan Amount Request (USD)"],
        bins=[0, 100000, 200000, 300000, 400000, 500000, 600000, 700000],
    )
    df["currentLoanExpensesBin"] = pd.cut(
        x=df["imp_Current Loan Expenses (USD)"],
        bins=[-1000, 0, 1000, 2000, 3000, 4000],
    )
    df["creditScoreBin"] = pd.cut(
        x=df["imp_Credit Score"],
        bins=[500, 600, 700, 800, 900],
    )
    df["propertyIdBin"] = pd.cut(
        x=df["Property ID"],
        bins=[0, 250, 500, 750, 1000],
    )
    df["propertyAgeBin"] = pd.cut(
        x=df["imp_Property Age"],
        bins=[0, 1000, 2000, 3000, 4000, 5000, 1778000],
    )
    df["propertyPriceBin"] = pd.cut(
        x=df["Property Price"],
        bins=[0, 100000, 200000, 300000, 400000, 500000, 1030000],
    )
    # Join dummies with original df
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "ageBin",
                        "incomeBin",
                        "loanAmountRequestBin",
                        "currentLoanExpensesBin",
                        "creditScoreBin",
                        "propertyIdBin",
                        "propertyAgeBin",
                        "propertyPriceBin",
                    ]
                ],
                columns=[
                    "ageBin",
                    "incomeBin",
                    "loanAmountRequestBin",
                    "currentLoanExpensesBin",
                    "creditScoreBin",
                    "propertyIdBin",
                    "propertyAgeBin",
                    "propertyPriceBin",
                ],
            ).astype(int),
        ],
        axis=1,
    )

    print("\nAfter imputation and dummy variables:")
    print(df)
    print(df.describe().round(2))
    print(df.info())

    x = df[
        [
            # "Gender",  # non-numeric, dummy
            # "Age",  # binned, insignificant
            # "Income (USD)",  # imputed
            # "Income Stability",  # non-numeric, dummy
            # "Profession",  # non-numeric, dummy
            # "Type of Employment",  # non-numeric, dummy
            # "Location",  # non-numeric, dummy
            "Loan Amount Request (USD)",
            # "Current Loan Expenses (USD)",  # imputed
            # "Expense Type 1",  # non-numeric, dummy
            # "Expense Type 2",  # non-numeric, dummy
            # "Dependents",  # imputed
            # "Credit Score",  # imputed
            # "No. of Defaults",  # dummy, insignificant
            # "Has Active Credit Card",  # non-numeric, dummy
            # "Property ID",  # insignificant
            # "Property Age",  # imputed
            # "Property Type",  # dummy, insignificant
            # "Property Location",  # non-numeric, dummy
            # "Co-Applicant",  # insignificant
            # "Property Price",  # insignificant
            # "m_Income (USD)",  # insignificant
            # "imp_Income (USD)",  # insignificant
            # "m_Current Loan Expenses (USD)",  # insignificant
            # "imp_Current Loan Expenses (USD)",  # insignificant
            "m_Dependents",
            # "imp_Dependents",  # insignificant
            "m_Credit Score",
            "imp_Credit Score",
            "m_Property Age",
            # "imp_Property Age",  # insignificant
            # "Gender_F",  # insignificant
            # "Gender_M",  # insignificant
            "Income Stability_High",
            "Income Stability_Low",
            # "Profession_Businessman",  # insignificant
            # "Profession_Commercial associate",  # insignificant
            # "Profession_Pensioner",  # insignificant
            # "Profession_State servant",  # insignificant
            # "Profession_Working",  # insignificant
            # "Type of Employment_Accountants",  # insignificant
            # "Type of Employment_Cleaning staff",  # insignificant
            # "Type of Employment_Cooking staff",  # insignificant
            # "Type of Employment_Core staff",  # insignificant
            # "Type of Employment_Drivers",  # insignificant
            # "Type of Employment_HR staff",  # insignificant
            # "Type of Employment_High skill tech staff",  # insignificant
            # "Type of Employment_IT staff",  # insignificant
            # "Type of Employment_Laborers",  # insignificant
            # "Type of Employment_Low-skill Laborers",  # insignificant
            "Type of Employment_Managers",
            # "Type of Employment_Medicine staff",  # insignificant
            # "Type of Employment_Private service staff",  # insignificant
            # "Type of Employment_Realty agents",  # insignificant
            # "Type of Employment_Sales staff",  # insignificant
            # "Type of Employment_Secretaries",  # insignificant
            # "Type of Employment_Security staff",  # insignificant
            # "Type of Employment_Waiters/barmen staff",  # insignificant
            # "Location_Rural",  # insignificant
            # "Location_Semi-Urban",  # insignificant
            # "Location_Urban",  # insignificant
            # "Expense Type 1_N",  # insignificant
            # "Expense Type 1_Y",  # insignificant
            # "Expense Type 2_N",  # insignificant
            # "Expense Type 2_Y",  # insignificant
            # "No. of Defaults_0",  # insignificant
            # "No. of Defaults_1",  # insignificant
            # "Has Active Credit Card_Active",  # insignificant
            # "Has Active Credit Card_Inactive",  # insignificant
            # "Has Active Credit Card_Unpossessed",  # insignificant
            # "Property Type_1",  # insignificant
            # "Property Type_2",  # insignificant
            # "Property Type_3",  # insignificant
            # "Property Type_4",  # insignificant
            # "Property Location_Rural",  # insignificant
            # "Property Location_Semi-Urban",  # insignificant
            # "Property Location_Urban",  # insignificant
            # "ageBin_(10, 20]",  # insignificant
            # "ageBin_(20, 30]",  # insignificant
            # "ageBin_(30, 40]",  # insignificant
            # "ageBin_(40, 50]",  # insignificant
            # "ageBin_(50, 60]",  # insignificant
            # "ageBin_(60, 70]",  # insignificant
            # "incomeBin_(0, 1000]",  # insignificant
            # "incomeBin_(1000, 2000]",  # insignificant
            # "incomeBin_(2000, 3000]",  # insignificant
            # "incomeBin_(3000, 4000]",  # insignificant
            # "incomeBin_(4000, 5000]",  # insignificant
            # "incomeBin_(5000, 1778000]",  # insignificant
            # "loanAmountRequestBin_(0, 100000]",  # insignificant
            # "loanAmountRequestBin_(100000, 200000]",  # insignificant
            "loanAmountRequestBin_(200000, 300000]",
            # "loanAmountRequestBin_(300000, 400000]",  # insignificant
            # "loanAmountRequestBin_(400000, 500000]",  # insignificant
            # "loanAmountRequestBin_(500000, 600000]",  # insignificant
            # "loanAmountRequestBin_(600000, 700000]",  # insignificant
            "currentLoanExpensesBin_(-1000, 0]",
            "currentLoanExpensesBin_(0, 1000]",
            "currentLoanExpensesBin_(1000, 2000]",
            # "currentLoanExpensesBin_(2000, 3000]",  # insignificant
            # "currentLoanExpensesBin_(3000, 4000]",  # insignificant
            # "creditScoreBin_(500, 600]",  # insignificant
            "creditScoreBin_(600, 700]",
            # "creditScoreBin_(700, 800]",  # insignificant
            "creditScoreBin_(800, 900]",
            # "propertyIdBin_(0, 250]",  # insignificant
            # "propertyIdBin_(250, 500]",  # insignificant
            # "propertyIdBin_(500, 750]",  # insignificant
            # "propertyIdBin_(750, 1000]",  # insignificant
            # "propertyAgeBin_(0, 1000]",  # insignificant
            # "propertyAgeBin_(1000, 2000]",  # insignificant
            # "propertyAgeBin_(2000, 3000]",  # insignificant
            # "propertyAgeBin_(3000, 4000]",  # insignificant
            # "propertyAgeBin_(4000, 5000]",  # insignificant
            # "propertyAgeBin_(5000, 1778000]",  # insignificant
            # "propertyPriceBin_(0, 100000]",  # insignificant
            # "propertyPriceBin_(100000, 200000]",  # insignificant
            # "propertyPriceBin_(200000, 300000]",  # insignificant
            # "propertyPriceBin_(300000, 400000]",  # insignificant
            # "propertyPriceBin_(400000, 500000]",  # insignificant
            # "propertyPriceBin_(500000, 1030000]",  # insignificant
        ]
    ]
    x = sm.add_constant(x)
    y = df[["Loan Sanction Amount (USD)"]]

    # Cross fold validation
    print("\nCross fold validation:")
    cross_fold_validation(x, y)

    y = df["Loan Sanction Amount (USD)"]  # reset y

    # Create training set with 80% of data and test set with 20% of data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    # Create linear regression model.
    model = sm.OLS(y_train, x_train).fit()
    predictions = model.predict(x_test)  # make the predictions by the model

    print(model.summary(title="Model B: Loan Sanction Amount (USD)"))
    print(
        "Root Mean Squared Error:",
        np.sqrt(metrics.mean_squared_error(y_test, predictions)),
    )

    # Draw validation plots on the best model.
    draw_validation_plots(
        title="Loan Sanction Amount (USD)",
        bins=8,
        y_test=y_test,
        predictions=predictions,
    )


# imputation by converting NA cells to median, mode, or mean
# includes binning and dummy variables
# also includes outlier treatment
def model_c():
    df = get_data()

    print(df)
    print(df.describe().round(2))
    print(df.info())

    # Imputation by converting NA cells to median, mode, or mean.
    df = convert_na_cells_to_num("Income (USD)", df, "mean")
    df = convert_na_cells_to_num("Current Loan Expenses (USD)", df, "median")
    df = convert_na_cells_to_num("Dependents", df, "mode")
    df = convert_na_cells_to_num("Credit Score", df, "mean")
    df = convert_na_cells_to_num("Property Age", df, "median")

    # Clip outliers by trimming the column to fixed boundaries and adding the adjusted column to the current data frame.
    # Logically cannot be less than 0
    df["Co-ApplicantAdjusted"] = df["Co-Applicant"].clip(0, 1)
    # Logically cannot be less than 0, retain max value
    df["PropertyPriceAdjusted"] = df["Property Price"].clip(0, 1028083)
    # Logically cannot be less than 0, retain max value
    df["imp_CurrentLoanExpensesAdjusted"] = df["imp_Current Loan Expenses (USD)"].clip(
        0, 3840.88
    )
    # If we are treating the values as days, then it cannot be less than 0 and more than current year (2023) in days
    # Maximum value found in data that is within range is 122966.28 which is roughly 337 years
    df["imp_PropertyAgeAdjusted"] = df["imp_Property Age"].clip(0, 123000)
    # view_and_get_outliers(df, "imp_Property Age", 3)

    # Generate dummies variables
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "Gender",
                        "Income Stability",
                        "Profession",
                        "Type of Employment",
                        "Location",
                        "Expense Type 1",
                        "Expense Type 2",
                        "No. of Defaults",
                        "Has Active Credit Card",
                        "Property Type",
                        "Property Location",
                    ]
                ],
                columns=[
                    "Gender",
                    "Income Stability",
                    "Profession",
                    "Type of Employment",
                    "Location",
                    "Expense Type 1",
                    "Expense Type 2",
                    "No. of Defaults",
                    "Has Active Credit Card",
                    "Property Type",
                    "Property Location",
                ],
            ).astype(int),
        ],
        axis=1,
    )  # Join dummy df with original df

    # Create bins.
    df["ageBin"] = pd.cut(
        x=df["Age"],
        bins=[10, 20, 30, 40, 50, 60, 70],
    )
    df["incomeBin"] = pd.cut(
        x=df["imp_Income (USD)"],
        bins=[0, 1000, 2000, 3000, 4000, 5000, 1778000],
    )
    df["loanAmountRequestBin"] = pd.cut(
        x=df["Loan Amount Request (USD)"],
        bins=[0, 100000, 200000, 300000, 400000, 500000, 600000, 700000],
    )
    df["currentLoanExpensesBin"] = pd.cut(
        x=df["imp_CurrentLoanExpensesAdjusted"],
        bins=[0, 1000, 2000, 3000, 4000],
    )
    df["creditScoreBin"] = pd.cut(
        x=df["imp_Credit Score"],
        bins=[500, 600, 700, 800, 900],
    )
    df["propertyIdBin"] = pd.cut(
        x=df["Property ID"],
        bins=[0, 250, 500, 750, 1000],
    )
    df["propertyAgeBin"] = pd.cut(
        x=df["imp_PropertyAgeAdjusted"],
        bins=[0, 1000, 2000, 3000, 123000],
    )
    df["propertyPriceBin"] = pd.cut(
        x=df["PropertyPriceAdjusted"],
        bins=[0, 50000, 100000, 150000, 200000, 1030000],
    )
    # Join dummies with original df
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "ageBin",
                        "incomeBin",
                        "loanAmountRequestBin",
                        "currentLoanExpensesBin",
                        "creditScoreBin",
                        "propertyIdBin",
                        "propertyAgeBin",
                        "propertyPriceBin",
                    ]
                ],
                columns=[
                    "ageBin",
                    "incomeBin",
                    "loanAmountRequestBin",
                    "currentLoanExpensesBin",
                    "creditScoreBin",
                    "propertyIdBin",
                    "propertyAgeBin",
                    "propertyPriceBin",
                ],
            ).astype(int),
        ],
        axis=1,
    )

    print("\nAfter imputation, dummy variables, and outlier treatment:")
    print(df)
    print(df.describe().round(2))
    print(df.info())

    # Visual representation of the average continuous data vs target ranges
    # plot_target_vs_avg_continuous_data(df.copy(), "Loan Amount Request (USD)")

    x = df[
        [
            # "Gender",  # non-numeric, dummy
            # "Age",  # binned, insignificant
            # "Income (USD)",  # imputed
            # "Income Stability",  # non-numeric, dummy
            # "Profession",  # non-numeric, dummy
            # "Type of Employment",  # non-numeric, dummy
            # "Location",  # non-numeric, dummy
            "Loan Amount Request (USD)",
            # "Current Loan Expenses (USD)",  # imputed
            # "Expense Type 1",  # non-numeric, dummy
            # "Expense Type 2",  # non-numeric, dummy
            # "Dependents",  # imputed
            # "Credit Score",  # imputed
            # "No. of Defaults",  # dummy, did not contribute to model
            # "Has Active Credit Card",  # non-numeric, dummy
            # "Property ID",  # insignificant
            # "Property Age",  # imputed
            # "Property Type",  # dummy, did not contribute to model
            # "Property Location",  # non-numeric, dummy
            # "Co-Applicant",  # clipped outliers
            # "Property Price",  # clipped outliers
            # "m_Income (USD)",  # insignificant
            # "imp_Income (USD)",  # insignificant
            # "m_Current Loan Expenses (USD)",  # insignificant
            # "imp_Current Loan Expenses (USD)",  # clipped outliers
            "m_Dependents",
            # "imp_Dependents",  # insignificant
            "m_Credit Score",
            "imp_Credit Score",
            "m_Property Age",
            # "imp_Property Age",  # clipped outliers
            # "Gender_F",  # insignificant
            # "Gender_M",  # insignificant
            # "Income Stability_High",  # insignificant
            "Income Stability_Low",
            # "Profession_Businessman",  # insignificant
            # "Profession_Commercial associate",  # insignificant
            # "Profession_Pensioner",  # insignificant
            # "Profession_State servant",  # insignificant
            # "Profession_Working",  # insignificant
            # "Type of Employment_Accountants",  # insignificant
            # "Type of Employment_Cleaning staff",  # insignificant
            # "Type of Employment_Cooking staff",  # insignificant
            # "Type of Employment_Core staff",  # insignificant
            # "Type of Employment_Drivers",  # insignificant
            # "Type of Employment_HR staff",  # insignificant
            # "Type of Employment_High skill tech staff",  # insignificant
            # "Type of Employment_IT staff",  # insignificant
            # "Type of Employment_Laborers",  # insignificant
            # "Type of Employment_Low-skill Laborers",  # insignificant
            # "Type of Employment_Managers",  # insignificant
            # "Type of Employment_Medicine staff",  # insignificant
            # "Type of Employment_Private service staff",  # insignificant
            # "Type of Employment_Realty agents",  # insignificant
            # "Type of Employment_Sales staff",  # insignificant
            # "Type of Employment_Secretaries",  # insignificant
            # "Type of Employment_Security staff",  # insignificant
            # "Type of Employment_Waiters/barmen staff",  # insignificant
            # "Location_Rural",  # insignificant
            # "Location_Semi-Urban",  # insignificant
            # "Location_Urban",  # insignificant
            # "Expense Type 1_N",  # did not contribute to model
            # "Expense Type 1_Y",  # did not contribute to model
            # "Expense Type 2_N",  # did not contribute to model
            # "Expense Type 2_Y",  # did not contribute to model
            # "No. of Defaults_0",  # did not contribute to model
            # "No. of Defaults_1",  # did not contribute to model
            # "Has Active Credit Card_Active",  # insignificant
            # "Has Active Credit Card_Inactive",  # insignificant
            # "Has Active Credit Card_Unpossessed",  # insignificant
            # "Property Type_1",  # did not contribute to model
            # "Property Type_2",  # did not contribute to model
            # "Property Type_3",  # did not contribute to model
            # "Property Type_4",  # did not contribute to model
            # "Property Location_Rural",  # insignificant
            # "Property Location_Semi-Urban",  # insignificant
            # "Property Location_Urban",  # insignificant
            # "ageBin_(10, 20]",  # insignificant, did not contribute to model
            # "ageBin_(20, 30]",  # insignificant, did not contribute to model
            # "ageBin_(30, 40]",  # insignificant, did not contribute to model
            # "ageBin_(40, 50]",  # insignificant, did not contribute to model
            # "ageBin_(50, 60]",  # insignificant, did not contribute to model
            # "ageBin_(60, 70]",  # insignificant, did not contribute to model
            # "incomeBin_(0, 1000]",  # insignificant
            # "incomeBin_(1000, 2000]",  # insignificant
            # "incomeBin_(2000, 3000]",  # insignificant
            # "incomeBin_(3000, 4000]",  # insignificant
            # "incomeBin_(4000, 5000]",  # insignificant
            # "incomeBin_(5000, 1778000]",  # insignificant
            # "loanAmountRequestBin_(0, 100000]",  # insignificant
            # "loanAmountRequestBin_(100000, 200000]",  # insignificant
            "loanAmountRequestBin_(200000, 300000]",
            # "loanAmountRequestBin_(300000, 400000]",  # insignificant
            # "loanAmountRequestBin_(400000, 500000]",  # insignificant
            # "loanAmountRequestBin_(500000, 600000]",  # insignificant
            # "loanAmountRequestBin_(600000, 700000]",  # insignificant
            # "currentLoanExpensesBin_(0, 1000]",  # insignificant
            # "currentLoanExpensesBin_(1000, 2000]",  # insignificant
            # "currentLoanExpensesBin_(2000, 3000]",  # insignificant
            # "currentLoanExpensesBin_(3000, 4000]",  # insignificant, did not contribute to model
            # "creditScoreBin_(500, 600]",  # insignificant
            # "creditScoreBin_(600, 700]",  # insignificant
            "creditScoreBin_(700, 800]",
            "creditScoreBin_(800, 900]",
            # "propertyIdBin_(0, 250]",  # insignificant
            # "propertyIdBin_(250, 500]",  # insignificant
            # "propertyIdBin_(500, 750]",  # insignificant
            # "propertyIdBin_(750, 1000]",  # insignificant
            # "propertyAgeBin_(0, 1000]",  # insignificant
            # "propertyAgeBin_(1000, 2000]",  # insignificant
            # "propertyAgeBin_(2000, 3000]",  # insignificant
            # "propertyAgeBin_(3000, 123000]",  # insignificant
            # "propertyPriceBin_(0, 50000]",  # insignificant
            # "propertyPriceBin_(50000, 100000]",  # insignificant
            # "propertyPriceBin_(100000, 150000]",  # insignificant
            # "propertyPriceBin_(150000, 200000]",  # insignificant
            # "propertyPriceBin_(200000, 1030000]",  # insignificant
            "Co-ApplicantAdjusted",
            # "PropertyPriceAdjusted",  # insignificant
            # "imp_CurrentLoanExpensesAdjusted",  # insignificant
            # "imp_PropertyAgeAdjusted",  # insignificant
        ]
    ]
    x = sm.add_constant(x)
    y = df[["Loan Sanction Amount (USD)"]]

    print("\nCross fold validation:")
    cross_fold_validation(x, y)

    y = df["Loan Sanction Amount (USD)"]  # reset y

    # Create training set with 80% of data and test set with 20% of data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    # Create linear regression model.
    model = sm.OLS(y_train, x_train).fit()
    predictions = model.predict(x_test)  # make the predictions by the model
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

    print("\nModel C: Loan Sanction Amount (USD)")
    print(model.summary(title="Model C: Loan Sanction Amount (USD)"))
    print("Root Mean Squared Error:", rmse)

    # Draw validation plots on the best model.
    draw_validation_plots(
        title="Loan Sanction Amount (USD)",
        bins=8,
        y_test=y_test,
        predictions=predictions,
    )


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # display_histograms(get_data())
    # display_scatter_matrix(get_data())
    # display_heatmap(
    #     get_data()[
    #         [
    #             "Age",
    #             "Income (USD)",
    #             "Loan Amount Request (USD)",
    #             "Current Loan Expenses (USD)",
    #             "Dependents",
    #             "Credit Score",
    #             "No. of Defaults",
    #             "Property Age",
    #             "Co-Applicant",
    #             "Property Price",
    #             "Loan Sanction Amount (USD)",
    #         ]
    #     ]
    # )

    # model_a()
    # model_b()
    model_c()


if __name__ == "__main__":
    main()
