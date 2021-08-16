from math import tanh


def heaviside_function(x, threshold=0, h0=0.5):
    x = round(x - threshold, 5)
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return 0.0
    elif x == 0.0:
        return h0


def rectified_linear_unit(x):
    return x if x > 0.0 else 0.0


def leaky_rectified_linear_unit(x, a=0.1):
    return parametric_rectified_linear_unit(x, a)


def parametric_rectified_linear_unit(x, a=0.1):
    return x if x > 0.0 else a * x


def tahn_activation_function(x):
    return tanh(x)
