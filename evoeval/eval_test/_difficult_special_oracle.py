import math


# oracle for EvoEval/10 in difficult
def _check_insensitive_palindrome(check_palindrome, string, gt_palindrome):
    assert len(check_palindrome) == len(gt_palindrome)
    assert check_palindrome.startswith(string)
    assert check_palindrome.lower() == check_palindrome[::-1].lower()


def _poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])


# oracle for EvoEval/32 in difficult
def _check_difficult_poly(xs, interval, solution, gt_solution):
    if gt_solution is None:
        assert solution is None
        return

    start, end = interval
    assert start <= solution <= end
    assert abs(_poly(xs, solution)) <= 2e-2
