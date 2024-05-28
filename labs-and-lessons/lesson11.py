# Exercise 1
def ex1():
    NUM_HOUR_PER_WEEK = 40  # 8 hours * 5 days
    NUM_SIMULATIONS = 500

    CUSTOMER_EXPECTED = 23
    CUSTOMER_SD = 7

    # Generate random numbers from a normal distribution.
    from scipy.stats import norm

    def generateRandomNumbers(mean, standardDev):
        randomNums = norm.rvs(loc=mean, scale=standardDev, size=NUM_HOUR_PER_WEEK)
        return randomNums

    # Do 500 simulations.
    import pandas as pd

    df = pd.DataFrame(columns=["Customers Per Week"])
    for i in range(0, NUM_SIMULATIONS):
        customers_per_hour = generateRandomNumbers(CUSTOMER_EXPECTED, CUSTOMER_SD)
        sub_df = pd.DataFrame(columns=["Customers Per Hour"])
        for i in range(0, NUM_HOUR_PER_WEEK):
            dictionary = {
                "Customers Per Hour": round(customers_per_hour[i], 2),
            }
            sub_df = sub_df._append(dictionary, ignore_index=True)

        # Calculate the number of times when more than 30 customers per hour and append to the dataframe.
        df = df._append(
            {"Customers Per Week": sum(sub_df["Customers Per Hour"] > 30)},
            ignore_index=True,
        )

    # Calculate the risk of incurring a loss.
    print(df)
    dfUnusualCustomersPerHour = df["Customers Per Week"].mean()
    print(
        "Total number of times when more than 30 customers per hour: "
        + str(dfUnusualCustomersPerHour)
    )


# Exercise 2
def ex2():
    import numpy as np

    LOW = 0
    HIGH = 24
    SIZE = 100
    NUM_SIMULATIONS = 5

    bankrupt_list = []
    for i in range(1, NUM_SIMULATIONS + 1):
        # Randomize data
        x = np.random.uniform(LOW, HIGH, SIZE)
        bankrupt = 0
        for option in x:
            if option < 2:
                bankrupt += 1
        bankrupt_list.append(bankrupt)

    print("Average numbers of bankrupt contestants: " + str(np.mean(bankrupt_list)))


# Exercise 3
def ex3():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats
    from scipy.stats import kstest
    import os

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Load samples.
    PATH = os.environ["DATASET_DIRECTORY"]
    FILE = "drugSales.csv"
    df = pd.read_csv(PATH + FILE)
    samples = np.array(df[["value"]])

    # High p-values are preferred and low D scores (closer to 0)
    # are preferred. loc==mean / scale==sd
    def runKolmogorovSmirnovTest(dist, loc, arg, scale, samples):
        d, pvalue = kstest(
            samples.ravel(),
            lambda x: dist.cdf(x, loc=loc, scale=scale, *arg),
            alternative="two-sided",
        )
        print("D value: " + str(d))
        print("p value: " + str(pvalue))
        dict = {
            "dist": dist.name,
            "D value": d,
            "p value": pvalue,
            "loc": loc,
            "scale": scale,
            "arg": arg,
        }
        return dict

    def fit_and_plot(dist, samples, df):
        print("\n*** " + dist.name + " ***")

        # Fit the distribution as best as possible
        # to existing data.
        params = dist.fit(samples)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Generate 'x' values between 0 and 80.
        x = np.linspace(0, 80, 80)

        # Run test to see if generated data aligns properly
        # to the sample data.
        distDandP = runKolmogorovSmirnovTest(dist, loc, arg, scale, samples)
        df = df._append(distDandP, ignore_index=True)

        # Plot the test and actual values together.
        _, ax = plt.subplots(1, 1)
        plt.hist(samples, bins=80, range=(0, 80))
        ax2 = ax.twinx()
        # ax2.plot(x, dist.pdf(x, loc=loc, scale=scale, *arg), "-", color="r", lw=2)
        ax2.plot(x, dist.cdf(x, loc=loc, scale=scale, *arg), "-", color="r", lw=2)
        plt.title(dist.name)
        plt.show()
        return df

    distributions = [
        scipy.stats.norm,
        scipy.stats.gamma,
        scipy.stats.chi2,
        scipy.stats.wald,
        scipy.stats.uniform,
        scipy.stats.norm,
        scipy.stats.t,
        scipy.stats.chi,
        scipy.stats.dgamma,
        scipy.stats.ncx2,
        scipy.stats.burr,
        scipy.stats.bradford,
        scipy.stats.crystalball,
        scipy.stats.exponnorm,
        scipy.stats.pearson3,
        scipy.stats.exponpow,
        scipy.stats.invgauss,
        scipy.stats.argus,
        scipy.stats.gennorm,
        scipy.stats.expon,
        scipy.stats.gengamma,
        scipy.stats.genextreme,
        scipy.stats.genexpon,
        scipy.stats.genlogistic,
    ]

    dfDistribution = pd.DataFrame()
    # Grid search the continuous distributions.
    for i in range(0, len(distributions)):
        dfDistribution = fit_and_plot(distributions[i], samples, dfDistribution)

    dfDistribution = dfDistribution.sort_values(by=["D value"])
    print(dfDistribution.T)


def main():
    # ex1()
    # ex2()
    ex3()


if __name__ == "__main__":
    main()
