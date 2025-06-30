r"""
Our code is based on the following:
https://github.com/pytorch/opacus/blob/v0.15.0/opacus/privacy_analysis.py
"""

import os, math, argparse
from typing import List, Tuple, Union

import numpy as np
from scipy import special
from scipy.optimize import brentq

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


########################
# LOG-SPACE ARITHMETIC #
########################
def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.
    Args:
        logx: First term in log space.
        logy: Second term in log space.
    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.
    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.
    Returns:
        Difference of numbers in log space.
    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _compute_log_a_for_int_alpha(q: float, sigma: float, alpha: int) -> float:
    r"""Computes :math:`log(A_\alpha)` for integer ``alpha``.
    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.
        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.
    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.
    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """

    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = (
                math.log(special.binom(alpha, i))
                + i * math.log(q)
                + (alpha - i) * math.log(1 - q)
        )

        s = log_coef_i + (i * i - i) / (2 * (sigma ** 2))
        log_a = _log_add(log_a, s)

    return float(log_a)


def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha``.
    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.
        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.
    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.
    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for any positive finite ``alpha``.
    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.
        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf
        for details.
    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.
    Returns:
        :math:`log(A_\alpha)` as defined in the paper mentioned above.
    """
    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)


def _log_erfc(x: float) -> float:
    r"""Computes :math:`log(erfc(x))` with high accuracy for large ``x``.
    Helper function used in computation of :math:`log(A_\alpha)`
    for a fractional alpha.
    Args:
        x: The input to the function
    Returns:
        :math:`log(erfc(x))`
    """
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)


def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    r"""Computes RDP of the Sampled Gaussian Mechanism at order ``alpha``.
    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.
    Returns:
        RDP at order ``alpha``; can be np.inf.
    """
    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    if q == 1.0:
        return alpha / (2 * sigma ** 2)

    if np.isinf(alpha):
        return np.inf

    return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(
        q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float]
) -> Union[List[float], float]:
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled Gaussian Mechanism (SGM) iterated ``steps`` times.
    Args:
        q: Sampling rate of SGM.
        noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added. Note that this is same as the standard
            deviation of the additive Gaussian noise when the L2-sensitivity
            of the function is 1.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.
    Returns:
        The RDP guarantees at all orders; can be ``np.inf``.
    """
    if isinstance(orders, float):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps


def get_privacy_spent(
        orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
            rdp_vec
            - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
            + np.log((orders_vec - 1) / orders_vec)
    )

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    return eps[idx_opt], orders_vec[idx_opt]


def compute_rdp_epsilon(sigma, sampling_prob, global_iterations, delta, verbose=False):
    rdp_1 = np.atleast_1d(compute_rdp(sampling_prob, sigma, global_iterations, DEFAULT_ALPHAS))
    eps, best_alpha = get_privacy_spent(DEFAULT_ALPHAS, rdp_1, delta)
    return float(eps)


def parse_args():
    parser = argparse.ArgumentParser(description="PFL-DocVQA: Centralized privacy calculator.")

    # Required
    parser.add_argument("--noise_multiplier", type=float, required=True, help="Noise multiplier.")
    parser.add_argument("--num_iterations", type=int, required=True, help="Number of steps.")
    parser.add_argument(
        "--providers_per_iteration", type=int, required=True, help="Number of groups (providers) sampled in each step."
    )
    parser.add_argument(
        "--num_total_providers_in_dataset", type=int, default=4149, help="Number of groups (providers) in the dataset."
    )
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta for privacy analysis")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sampling_prob = args.providers_per_iteration / args.num_total_providers_in_dataset

    epsilon = compute_rdp_epsilon(
        sigma=args.noise_multiplier, sampling_prob=sampling_prob, global_iterations=args.num_iterations, delta=args.delta
    )
    print(
        f"The privacy parameters result in a epsilon of {epsilon} at delta {args.delta} when using the RDP accountant."
    )
    print(
        f"(noise_multiplier {args.noise_multiplier}, num_steps {args.num_iterations}, num_providers {args.providers_per_iteration})"
    )
