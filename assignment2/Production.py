import pandas as pd
import numpy as np
import pickle


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


def load_data(df):
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
                        "SubtitlesEnabled",
                        "ContentType",
                        "Gender",
                        "ParentalControl",
                    ]
                ],
                columns=[
                    "SubscriptionType",
                    "PaymentMethod",
                    "SubtitlesEnabled",
                    "ContentType",
                    "Gender",
                    "ParentalControl",
                ],
            ).astype(int),
        ],
        axis=1,
    )  # Join dummy df with original df

    # Create bins
    df["UserRatingBin"] = pd.cut(
        x=df[
            "imp_UserRating" if "imp_UserRating" in df.columns.values else "UserRating"
        ],
        bins=[0, 2, 4, 6],
    )
    df["SupportTicketsPerMonthBin"] = pd.cut(
        x=df[
            "imp_SupportTicketsPerMonth"
            if "imp_SupportTicketsPerMonth" in df.columns.values
            else "SupportTicketsPerMonth"
        ],
        bins=[0, 2, 4, 6, 8, 10],
    )

    # Generate dummy variables for binned features.
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "UserRatingBin",
                        "SupportTicketsPerMonthBin",
                    ]
                ],
                columns=[
                    "UserRatingBin",
                    "SupportTicketsPerMonthBin",
                ],
            ).astype(int),
        ],
        axis=1,
    )  # Join dummies with original df

    print(df)
    print(df.describe().round(2))
    print(df.info())

    # Fill missing values in Account Age column with 0's
    if "imp_AccountAge" in df.columns:
        if df["imp_AccountAge"].isna().sum() > 0:
            df["imp_AccountAge"].fillna(0, inplace=True)
    else:
        df["imp_AccountAge"] = df["AccountAge"]

    # Create columns that may not have been created by get_dummies()
    # if there were no values for them.
    columns_to_create = [
        "SubscriptionType_Premium",
        "PaymentMethod_Credit card",
        "SubtitlesEnabled_Yes",
        "Gender_Female",
        "ContentType_TV Shows",
        "ContentType_Movies",
        "ParentalControl_Yes",
    ]

    for column_name in columns_to_create:
        if column_name not in df.columns:
            df[column_name] = 0

    return df[
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


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    try:
        df = pd.read_csv(
            "CustomerChurn_Mystery.csv",
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
            ),
        )
        x = load_data(df)
        print(x.describe())
        file = open("best_model.pkl", "rb")
        model = pickle.load(file)

        # Make predictions.
        predictions = model.predict(x)
        print(f"Predictions: {predictions}")

        # Store predictions in a dataframe and save to csv.
        df_predictions = pd.DataFrame(predictions, columns=["Churn"])
        df_predictions.to_csv("CustomerChurn_Predictions.csv", index=False, header=True)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
