import numpy as np
from scipy.optimize import minimize

from dca.adca.utils import robust_minimize
from dca.decline_curve_analysis import Arps, CurveLoss


def test_that_robust_optimization_returns_bounded_parameters():
    # Create a curve with some crazy, increasing data
    t = np.linspace(0, 25, num=2**5)
    y = 1e6 * (1 + np.linspace(0, 1, num=len(t)))

    def log_ARPS(t, theta1, theta2, theta3):
        return Arps(theta1, theta2, theta3).eval_log(t)

    def log_ARPS_grad(t, theta1, theta2, theta3):
        return Arps(theta1, theta2, theta3).eval_grad_log(t)

    mu = np.array([0, 0, 0])

    loss = CurveLoss(
        curve_func=log_ARPS,
        p=1.5,
        curve_func_grad=log_ARPS_grad,
        half_life=25,
        precision_matrix=0 * np.eye(3),  # No regularization
        mu=mu,
    )

    optimization_result = minimize(
        fun=loss,
        x0=mu,
        method="BFGS",
        jac=loss.grad,
        args=(t, np.log(y)),
    )

    robust_result = robust_minimize(
        fun=loss,
        x0=mu,
        jac=loss.grad,
        args=(t, np.log(y)),
    )

    # Smaller norm
    assert np.linalg.norm(robust_result.x) < np.linalg.norm(optimization_result.x)

    # But the curve is almost identical over the range
    opt_curve = Arps(*optimization_result.x)
    robust_curve = Arps(*robust_result.x)

    assert np.allclose(opt_curve.eval(t), robust_curve.eval(t), rtol=1e-4)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys", "-x"])
