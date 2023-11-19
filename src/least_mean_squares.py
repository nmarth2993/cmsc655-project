"""
Nicholas Marthinuss
11/12/2023

Least Mean Squares implementation
"""


import numpy as np
from matplotlib import pyplot as plt
import random
import statistics

# the maximum degree allowed to be used by LMS
FIT_DEGREE = 3

# adjust standard deviation to increase/decrease noise of generated data
STDDEV = 50


def gen_data(a: float, b: float, n: int = 10, weights=None, degree=None):
    """
    Generate `n` noisy data points on `[a, b]` based on a polynomial

    ## Parameters
    `a` : float
        the beginning of the interval on which to generate points
    `b` : float
        the end of the interval on which to generate points
    `n` : int, optional,
        the number of points to generate, defaults to 10
    `weights` : list, optional
        the polynomial weights to use when generating data
    `degree` : int, optional
        the desired degree polynomial to generate if no weights are provided

    ## Returns
    `xdata` : np.ndarray
        an array of x-values for the data generated
    `ydata` : np.ndarray
        an array of y-values for the data generated
    `weights` : list
        a list of weights that describe the polynomial used to generate the data

    ## Notes
    if both `weights` and `degree` are specified, the polynomial degree is based on `weights`
    and `degree` is ignored
    """

    # generate random weights if none are provided
    if weights is None:
        # if no degree is specified, pick a random degree in [1, 4]
        if degree is None:
            degree = random.randint(1, 4)

        # generate random coefficients on [-9, 9]
        weights = []
        for _ in range(degree + 1):
            weights.append(random.randint(-9, 9))

    xdata = np.linspace(a, b, n)

    # initialize y series as an np.array to easily add noise weights later
    ydata = np.zeros(n)
    for i, x in enumerate(xdata):
        ydata[i] = poly_eval(x, weights)

    # generate noise based on STDDEV
    noise = np.random.default_rng().normal(0, STDDEV, n)

    # add noise to the y-values
    ydata = ydata + noise

    return xdata, ydata, weights


def lms(data: list, degree: int) -> np.ndarray:
    """
    Least mean squares algorithm. Returns polynomial weights of the best fit to the data

    ## Parameters
    `data` : list
        a list of the datapoints to be fit
    `degree` : int
        the desired degree of the polynomial used to fit the data

    ## Returns
    `weights` : np.ndarray
        array of weights that describe the LMS polynomial

    """

    # create the design matrix
    # [1, x, x^2, x^3, ..., x^n]
    design_matrix = np.ones((len(data), degree + 1))
    for i in range(1, degree + 1):
        # write x^n for each x value
        for j in range(len(data)):
            design_matrix[j][i] = data[j][0] ** i

    # the result vector is b in the equation Ax=b
    result_vector = np.array([point[1] for point in data])

    # rename matrices to stay consistent with math equations below
    x, y = design_matrix, result_vector

    # calculate (XTX)^-1 and XTY
    xtx_inv = np.linalg.inv(np.matmul(x.transpose(), x))
    xty = np.matmul(x.transpose(), y)

    # multiply these matrices to get the weight vector
    weights = np.matmul(xtx_inv, xty)

    return weights


def poly_eval(x: float, weights: list) -> float:
    """
    Evaluate the polynomial described by `weights` at `x`

    ## Parameters
    `x` : float
        the point at which to evaluate the polynomial
    `weights` : np.ndarray
        the weights that describe the polynomial

    ## Returns
    `y_val` : float
        the y-value of the function output at `x`
    """

    y_val = 0

    for i, weight in enumerate(weights):
        y_val += x**i * weight

    return y_val


def poly_weights_str(weights: list) -> str:
    """
    Generate a string representation of given polynomial weights

    ## Parameters
    `weights` : list
        list of polynomial weights to be stringified

    ## Returns
    `poly_str` : str
        the string representation of the polynomial
    """
    if not weights:
        return "(null)"

    # generating the latex x^{i} so that it works with multi-digit powers
    poly_str = [
        ("+" if weight >= 0 else "")
        + str(weight)
        + "x"
        + (f"^{{{len(weights) - i - 1}}}" if i != len(weights) - 2 else "")
        for i, weight in enumerate(weights[:-1])
    ]

    # finally, add the constant without an x^0
    poly_str.append(("+" if weights[-1] >= 0 else "") + str(weights[-1]))

    poly_str = "".join(poly_str)

    # remove leading + sign if first coefficient is positive
    if poly_str[0] == "+":
        return poly_str[1:]
    return poly_str


def calc_r2(lms_weights: np.ndarray, dataset: list) -> float:
    """
    Calculate the R^2 value of the LMS fit line

    ## Parameters
    `lms_weights` : np.ndarray
        weights of the line of best fit polynomial
    `dataset` : list
        a list of the points in the dataset

    ## Returns
    `R^2` : float
        the r squared value
    """
    rss = 0
    tss = 0

    # calculating total sum of squares requires knowing the mean
    y_mean = statistics.mean([y for x, y in dataset])

    # calculate rss and tss
    for x, y in dataset:
        rss += (poly_eval(x, list(lms_weights)) - y) ** 2
        tss += (y - y_mean) ** 2

    return 1 - rss / tss


def plot_fit(
    a: float,
    b: float,
    dataset: list,
    lms_weights: np.ndarray,
    poly_weights: list = None,
) -> None:
    """
    Plots dataset and LMS line of best fit

    ## Parameters
    `a` : float
        start of interval to be plotted
    `b` : float
        end of interval to be plotted
    `dataset` : list
        data points to be scatter plotted
    `lms_weights`: np.ndarray
        weights for LMS line of best fit
    `poly_weights` : list, optional
        weights of true polynomial (if known)

    """
    # first, scatter plot the dataset
    # must be split into x component and y component to plot
    xdata = [point[0] for point in dataset]
    ydata = [point[1] for point in dataset]
    plt.scatter(xdata, ydata, marker=".")

    # next, plot the line of best fit
    xseries = np.linspace(a, b)
    yseries = []

    # NOTE: using poly_eval for true polynomial evaluation
    # for a performance improvement and the expense of some accuracy,
    # use horner's method through np.polyval()
    for x in xseries:
        yseries.append(poly_eval(x, lms_weights))

    # the label is set to the string representation of the polynomial
    # values are rounded to 2 decimal places and the weights are reversed
    # since weights[0] is the constant and weights[n] is the coefficient for degree n

    # add r squared value in legend
    r_squared = calc_r2(lms_weights, dataset)
    r_squared = np.round(r_squared, 3)

    # R^2 is added to the polynomial label. the text is made smaller by using the ^ latex format
    plt.plot(
        xseries,
        yseries,
        label=f"LMS ${poly_weights_str(list(np.around(lms_weights, 2))[::-1])}$\n$^{{R^2={r_squared}}}$",
        color="orange",
    )

    # set plot title
    # if true polynomial weights were provided, set the title to the polynomial
    if poly_weights is None:
        plt.title("LMS Data Fit")
    else:
        # reverse weights before getting string representation
        poly_str = poly_weights_str(poly_weights[::-1])
        plt.title(f"LMS Fit of ${poly_str}$ w/ Noise")

    # add legend, gridlines, axes labels
    plt.legend()
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    # set intervals
    A, B = -4, 4
    # set number of points to generate
    N = 100

    # generate some data to be fit
    x, y, weights = gen_data(A, B, N, degree=4)
    """
    # x, y, weights = [-4, -3, -2, -1, 1, 2, 3, 4], [16, 9, 4, 1, 1, 4, 9, 16], [0, 0, 1]
    x, y, weights = (
        [-4, -3, -2, -1, 1, 2, 3, 4],
        [15, 10, 3.25, 0.87, 1, 0.9, 5, 8, 17],
        [0, 0, 1],
    )
    """

    dataset = list(zip(x, y))

    # compute the LMS weights and plot the fit against the dataset
    weight_vector = lms(dataset, FIT_DEGREE)
    plot_fit(A, B, dataset, weight_vector, weights)
