# Exercise 1
def ex1():
    from statsmodels.stats.power import TTestIndPower

    effect = 1.11  # Obtained from previous step.
    alpha = 0.05  # Enable 95% confidence for two tail test.
    power = 0.8  # One minus the probability of a type II error.
    # Limits possibility of type II error to 20%.
    analysis = TTestIndPower()
    numSamplesNeeded = analysis.solve_power(effect, power=power, alpha=alpha)
    print(numSamplesNeeded)


# Exercise 2
def ex2():
    from scipy import stats

    old_cases_count = [
        10,
        11,
        6,
        18,
        11,
        9,
        13,
        9,
        3,
        12,
        3,
        13,
        14,
        4,
        12,
        8,
        18,
        17,
        15,
        18,
        6,
        1,
        13,
        9,
        11,
        15,
        11,
        7,
        12,
        14,
    ]

    new_cases_count = [
        5,
        7,
        3,
        12,
        0,
        7,
        0,
        8,
        9,
        5,
        5,
        2,
        2,
        2,
        4,
        6,
        6,
        7,
        1,
        6,
        10,
        1,
        0,
        2,
        2,
        4,
        5,
        1,
        4,
        3,
    ]

    testResult = stats.ttest_ind(
        new_cases_count[:14], old_cases_count[:14], equal_var=False
    )

    import numpy as np

    print("Hypothesis test p-value: " + str(testResult))
    print("New sales mean: " + str(np.mean(new_cases_count)))
    print("New sales std: " + str(np.std(new_cases_count)))


def main():
    # ex1()
    ex2()


if __name__ == "__main__":
    main()
