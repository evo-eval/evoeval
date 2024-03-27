import math


# oracle for EvoEval/32 in subtle
def _poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])


def _check_poly(xs, solution):
    full_xs = [xs[0], xs[1]]
    for i in range(2, len(xs)):
        full_xs.extend([0, xs[i]])
    assert abs(_poly(full_xs, solution)) <= 1e-6
