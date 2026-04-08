"""
Statistical Arbitrage utilities based on Avellaneda and Lee (2008):
"Statistical Arbitrage in the U.S. Equities Market"

This module implements:
1. PCA-based factor model for stock returns
2. Ornstein-Uhlenbeck parameter estimation for residuals
3. S-score computation for mean-reversion trading signals
"""

from typing import Any

import numpy as np
import pandas as pd
from utilities.principal_component_analysis import principal_component_analysis


def compute_volume_adjusted_returns(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    trailing_window: int = 60,
) -> pd.DataFrame:
    """Compute volume-adjusted ("trading time") returns.

    The idea is that price moves on low-volume days carry more information
    than moves on high-volume days. The adjusted return is:

        R_bar_{i,t} = R_{i,t} * <delta_V_i> / V_{i,t}

    where <delta_V_i> is the trailing average daily volume and V_{i,t} is
    the actual volume on day t. This amplifies low-volume moves and dampens
    high-volume moves.

    Args:
        returns: Daily returns (T x N).
        volume: Daily trading volume (T x N), must share index and columns with returns.
        trailing_window: Number of days for the trailing average volume.

    Returns:
        Volume-adjusted returns (T x N). Rows with insufficient volume history are NaN.
    """
    # Align volume to returns index and columns
    common_cols = returns.columns.intersection(volume.columns)
    common_idx = returns.index.intersection(volume.index)
    vol = volume.loc[common_idx, common_cols]
    ret = returns.loc[common_idx, common_cols]

    # Trailing average volume
    avg_volume = None  # !!! COMPLETE AS APPROPRIATE !!!

    # Volume adjustment ratio: <δV> / V_t
    # Clip volume to avoid division by zero or extreme ratios
    vol_clipped = vol.clip(lower=1)
    adjustment = None  # !!! COMPLETE AS APPROPRIATE !!!

    # Cap extreme adjustments (e.g., when volume drops to near zero)
    adjustment = adjustment.clip(upper=10.0)

    return ret * adjustment


def estimate_factor_model(
    returns: pd.DataFrame,
    n_factors: int = 15,
) -> dict[str, Any]:
    """Estimate a PCA-based factor model for stock returns.

    The model decomposes returns as:
        R_i,t = alpha_i + sum_j(beta_i,j * F_j,t) + epsilon_i,t

    where F_j are the principal component factors extracted from the correlation matrix.

    Args:
        returns: Returns matrix (T x N), T = time periods, N = assets.
        n_factors: Number of principal component factors to use, default 15.

    Returns:
        Dictionary containing eigenvalues, eigenvectors, factors, betas, alphas,
        residuals, explained_variance, and n_factors.

    Raises:
        TypeError: If returns is not a DataFrame.
        ValueError: If returns is empty, has insufficient data, or invalid parameters.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a DataFrame, got {type(returns)}")

    if returns.empty:
        raise ValueError("returns DataFrame is empty")

    if returns.isna().all().any():
        nan_cols = returns.columns[returns.isna().all()].tolist()
        raise ValueError(f"Columns {nan_cols} contain only NaN values")

    T, N = returns.shape

    if T < N:
        raise ValueError(
            f"Insufficient data: {T} observations for {N} assets. "
            f"Need at least N observations for PCA."
        )

    if n_factors > N:
        raise ValueError(
            f"n_factors ({n_factors}) cannot exceed number of assets ({N})"
        )

    if n_factors < 1:
        raise ValueError(f"n_factors must be at least 1, got {n_factors}")

    # Compute correlation matrix
    cov_matrix = returns.cov().values

    # Eigendecomposition
    eigenvalues, eigenvectors = None  # !!! COMPLETE AS APPROPRIATE !!!

    # Select top n_factors
    eigenvalues_selected = None  # !!! COMPLETE AS APPROPRIATE !!!
    eigenvectors_selected = None  # !!! COMPLETE AS APPROPRIATE !!!

    # Factor returns
    factors = None  # !!! COMPLETE AS APPROPRIATE !!!
    factors_df = pd.DataFrame(
        factors, index=returns.index, columns=[f"PC{i + 1}" for i in range(n_factors)]
    )

    # For each asset, regress returns on factors to get betas and alpha
    residuals_df, betas_df, alphas_series = estimate_ou_window_residuals(
        returns,
        factors_df,
    ).values()

    # Explained variance ratio
    explained_variance = None  # !!! COMPLETE AS APPROPRIATE !!!

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "factors": factors_df,
        "betas": betas_df,
        "alphas": alphas_series,
        "residuals": residuals_df,
        "explained_variance": explained_variance,
        "n_factors": n_factors,
    }


def estimate_ou_window_residuals(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    ou_window: int | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run OLS on the O-U estimation window and return cumulative residuals.

    Args:
        returns: Returns from the full estimation window (T x N).
        factors: PCA factor returns from the same full window (T x K).
        ou_window: Number of days in the O-U estimation sub-window. If None, use the full window.

    Returns:
        Dictionary with residuals, betas, and alphas for the O-U sub-window.
    """
    if ou_window is None:
        ou_window = len(returns)
    returns_ou = returns.iloc[-ou_window:]
    factors_ou = factors.iloc[-ou_window:]
    T_ou, N = returns_ou.shape
    n_factors = factors_ou.shape[1]

    betas = np.zeros((N, n_factors))
    alphas = np.zeros(N)
    residuals = np.zeros((T_ou, N))

    # !!! COMPLETE AS APPROPRIATE !!!

    residuals_df = pd.DataFrame(
        residuals, index=returns_ou.index, columns=returns_ou.columns
    )
    betas_df = pd.DataFrame(
        betas, index=returns.columns, columns=[f"PC{i + 1}" for i in range(n_factors)]
    )
    alphas_series = pd.Series(alphas, index=returns.columns)

    return {
        "residuals": residuals_df,
        "betas": betas_df,
        "alphas": alphas_series,
    }


def estimate_ou_parameters(
    residuals: pd.Series,
    dt: float = 1 / 252,
) -> dict[str, float]:
    """Estimate Ornstein-Uhlenbeck process parameters from cumulative residuals.

    The O-U process is: dX_t = kappa * (m - X_t) * dt + sigma * dW_t

    Estimated via discrete AR(1): X_{t+1} = a + b * X_t + epsilon_t
    where b = exp(-kappa*dt), a = m*(1-b).

    The original parameters are recovered as:
        kappa = -log(b) / dt
        m = a / (1 - b)
        sigma_eq = sqrt(Var(epsilon) / (1 - b^2))

    Args:
        residuals: Cumulative residual time series X_t.
        dt: Time step in years, default 1/252 (daily).

    Returns:
        Dictionary with kappa, m, sigma, sigma_eq, half_life, a, b, var_epsilon.
    """
    X = residuals.values
    T = len(X)

    a = None  # !!! COMPLETE AS APPROPRIATE !!!
    b = None  # !!! COMPLETE AS APPROPRIATE !!!
    epsilon = None  # !!! COMPLETE AS APPROPRIATE !!!
    var_epsilon = None  # !!! COMPLETE AS APPROPRIATE !!!

    kappa = None  # !!! COMPLETE AS APPROPRIATE !!!
    m = None  # !!! COMPLETE AS APPROPRIATE !!!
    sigma = None  # !!! COMPLETE AS APPROPRIATE !!!

    # Equilibrium standard deviation
    sigma_eq = None  # !!! COMPLETE AS APPROPRIATE !!!

    # Half-life of mean reversion
    half_life = None  # !!! COMPLETE AS APPROPRIATE !!!

    return {
        "kappa": kappa,
        "m": m,
        "sigma": sigma,
        "sigma_eq": sigma_eq,
        "half_life": half_life,
        "a": a,
        "b": b,
        "var_epsilon": var_epsilon,
    }


def estimate_all_ou_parameters(
    cumulative_residuals: pd.DataFrame,
    dt: float = 1 / 252,
    center_ou_means: bool = True,
) -> dict[str, dict[str, float]]:
    """Estimate O-U parameters for all assets.

    Args:
        cumulative_residuals: Cumulative residuals (T x N).
        dt: Time step in years.
        center_ou_means: Whether to center the equilibrium means by subtracting the cross-sectional
            average.

    Returns:
        O-U parameters for each asset.
    """
    ou_params = {}
    for asset in cumulative_residuals.columns:
        ou_params[asset] = estimate_ou_parameters(
            cumulative_residuals[asset].dropna(), dt=dt
        )

    # Center O-U equilibrium means by subtracting the cross-sectional average
    if center_ou_means:
        pass  # !!! COMPLETE AS APPROPRIATE !!!

    return ou_params


def compute_s_score(
    cumulative_residuals: pd.DataFrame,
    ou_params: dict[str, dict[str, float]],
    modified: bool = False,
) -> pd.DataFrame:
    """Compute the s-score for each asset based on its deviation from equilibrium.

    The s-score is defined as:
        s_t = (X_t - m) / sigma_eq
    The modified version is:
        s_mod,t = s_t - alpha /(kappa * sigma_eq)

    Args:
        cumulative_residuals: Cumulative residuals (T x N).
        ou_params: O-U parameters for each asset.
        modified: Whether to compute the modified s-score that accounts for the residual drift.

    Returns:
        S-scores (T x N).
    """
    s_scores = pd.DataFrame(
        index=cumulative_residuals.index,
        columns=cumulative_residuals.columns,
        dtype=float,
    )

    # !!! COMPLETE AS APPROPRIATE !!!

    return s_scores


def update_positions(
    current_positions: dict[str, float],
    cur_s_scores: pd.Series,
    valid_assets: list[str],
    s_bo: float = 1.25,
    s_so: float = 1.25,
    s_bc: float = 0.50,
    s_sc: float = 0.50,
) -> dict[str, float]:
    """Update positions for a single day based on current s-scores and previous positions.

    Trading rules:
        - Buy to open if s < -s_bo
        - Sell to open if s > +s_so
        - Close long if s > -s_bc
        - Close short if s < +s_sc

    Args:
        current_positions: Previous positions {asset: +1/-1/0}.
        cur_s_scores: Current s-scores for all assets.
        valid_assets: Assets passing the O-U parameter filter.
        s_bo: Threshold to open long position.
        s_so: Threshold to open short position.
        s_bc: Threshold to close long position.
        s_sc: Threshold to close short position.

    Returns:
        Updated positions dict for today.
    """
    new_positions: dict[str, float] = {}

    for asset in cur_s_scores.index:
        s = cur_s_scores[asset]
        prev = current_positions.get(asset, 0.0)

        ### !!! COMPLETE AS APPROPRIATE !!!

    return new_positions


def compute_portfolio_weights(
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute portfolio weights from trading signals.

    Args:
        positions: Position signals for all assets (+1, -1, 0).

    Returns:
        Portfolio weights.
    """
    weights = positions.copy()

    # !!! COMPLETE AS APPROPRIATE !!!

    return weights


def compute_strategy_statistics(
    cumulative_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute performance statistics for the strategy.

    Args:
        cumulative_returns: Cumulative portfolio returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of trading periods per year.

    Returns:
        Performance statistics dict.
    """
    returns = cumulative_returns.pct_change().dropna()

    # Annualized return
    total_return = cumulative_returns.iloc[-1] / cumulative_returns.iloc[0] - 1
    n_years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Annualized volatility
    annualized_vol = returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio
    sharpe = (
        (annualized_return - risk_free_rate) / annualized_vol
        if annualized_vol > 0
        else 0
    )

    # Maximum drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }
