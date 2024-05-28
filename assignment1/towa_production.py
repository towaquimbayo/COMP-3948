import csv
import pandas as pd
import numpy as np


def get_loan_sanction_amount(
    loan_amount_request,
    m_dependents,
    m_credit_score,
    imp_credit_score,
    m_property_age,
    income_stability_low,
    loan_amount_request_bin_200000_to_300000,
    credit_score_bin_700_to_800,
    credit_score_bin_800_to_900,
    co_applicant_adjusted,
):
    loan_sanction_amount = (
        -195100
        + 0.5234 * loan_amount_request
        + 6328.4394 * m_dependents
        + 3466.4609 * m_credit_score
        + 245.6046 * imp_credit_score
        + 7107.5291 * m_property_age
        - 7192.7163 * income_stability_low
        + 10640 * loan_amount_request_bin_200000_to_300000
        - 7700.8052 * credit_score_bin_700_to_800
        - 19250 * credit_score_bin_800_to_900
        + 30660 * co_applicant_adjusted
    )
    print(f"Predicted Loan Sanction Amount: {loan_sanction_amount}")
    return loan_sanction_amount


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
    # Imputation of missing values
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode(), inplace=True)
            else:
                imputation_method = (
                    "mode"
                    if df[col].dtype == "float64" and df[col].isna().sum() == 0
                    else "mean"
                )
                if col in ["Current Loan Expenses (USD)", "Property Age"]:
                    imputation_method = "median"
                elif col in ["Income (USD)", "Credit Score"]:
                    imputation_method = "mean"
                df = convert_na_cells_to_num(col, df, imputation_method)

    # Outlier treatment
    df["Co-ApplicantAdjusted"] = df["Co-Applicant"].clip(0, 1)

    # Binning of continuous variables
    df["loanAmountRequestBin"] = pd.cut(
        x=df[
            "imp_Loan Amount Request (USD)"
            if "imp_Loan Amount Request (USD)" in df.columns.values
            else "Loan Amount Request (USD)"
        ],
        bins=[0, 100000, 200000, 300000, 400000, 500000, 600000, 700000],
    )
    df["creditScoreBin"] = pd.cut(
        x=df[
            "imp_Credit Score"
            if "imp_Credit Score" in df.columns.values
            else "Credit Score"
        ],
        bins=[500, 600, 700, 800, 900],
    )

    # Join dummies with original df
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "Income Stability",
                        "loanAmountRequestBin",
                        "creditScoreBin",
                    ]
                ],
                columns=[
                    "Income Stability",
                    "loanAmountRequestBin",
                    "creditScoreBin",
                ],
            ).astype(int),
        ],
        axis=1,
    )

    # Fill missing values in Loan Amount Request (USD) column with 0's
    if (
        "imp_Loan Amount Request (USD)" in df.columns
        and df["imp_Loan Amount Request (USD)"].isna().sum() > 0
    ):
        df["imp_Loan Amount Request (USD)"].fillna(0, inplace=True)

    # Fill missing values in Co-ApplicantAdjusted column with 0's
    if df["Co-ApplicantAdjusted"].isna().sum() > 0:
        df["Co-ApplicantAdjusted"].fillna(0, inplace=True)

    # Create Income Stability_Low column filled with 0's if it doesn't exist
    if "Income Stability_Low" not in df.columns:
        df["Income Stability_Low"] = 0

    # Create m_Dependents column filled with 0's if it doesn't exist
    if "m_Dependents" not in df.columns:
        df["m_Dependents"] = 0

    # Create m_Credit Score column filled with 0's if it doesn't exist
    if "m_Credit Score" not in df.columns:
        df["m_Credit Score"] = 0

    # Create m_Property Age column filled with 0's if it doesn't exist
    if "m_Property Age" not in df.columns:
        df["m_Property Age"] = 0

    # Fill missing values in Credit Score column with 0's
    if "imp_Credit Score" not in df.columns:
        if df["Credit Score"].isna().sum() > 0:
            df["Credit Score"] = df["Credit Score"].fillna(0)
    else:
        df["imp_Credit Score"] = df["imp_Credit Score"].fillna(0)

    return df[
        [
            "imp_Loan Amount Request (USD)"
            if "imp_Loan Amount Request (USD)" in df.columns
            else "Loan Amount Request (USD)",
            "m_Dependents",
            "m_Credit Score",
            "imp_Credit Score" if "imp_Credit Score" in df.columns else "Credit Score",
            "m_Property Age",
            "Income Stability_Low",
            "loanAmountRequestBin_(200000, 300000]",
            "creditScoreBin_(700, 800]",
            "creditScoreBin_(800, 900]",
            "Co-ApplicantAdjusted",
        ]
    ]


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    predicted_amounts = []

    try:
        df = pd.read_csv(
            "loan_mystery.csv",
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
            ),
        )
        x = load_data(df)

        # Get loan sanction amount using the x values from the mystery file
        for i in range(len(x.values)):
            predicted_amounts.append(
                get_loan_sanction_amount(
                    x.values[i][0],
                    x.values[i][1],
                    x.values[i][2],
                    x.values[i][3],
                    x.values[i][4],
                    x.values[i][5],
                    x.values[i][6],
                    x.values[i][7],
                    x.values[i][8],
                    x.values[i][9],
                )
            )
    except Exception as e:
        print(e)

    try:
        with open("loan_predictions.csv", "w", encoding="UTF8", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Loan Sanction Amount (USD)"])
            for i in range(len(predicted_amounts)):
                csv_writer.writerow([float(predicted_amounts[i])])
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
