"""Synthetic market data, interpolated curves, volatility surfaces, and calibration.

Rates and volatilities are continuously compounded decimal values.  Curves use
linear interpolation in zero-rate space with flat extrapolation; surfaces use
bilinear interpolation with flat extrapolation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares


def _positive_grid(values: Iterable[float], name: str) -> np.ndarray:
    grid = np.asarray(tuple(values), dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional grid")
    if np.any(grid <= 0.0) or np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} must be positive and strictly increasing")
    return grid


@dataclass(frozen=True)
class ZeroCurve:
    """Continuously compounded zero curve."""

    tenors: np.ndarray
    zero_rates: np.ndarray

    def __post_init__(self) -> None:
        tenors = _positive_grid(self.tenors, "tenors")
        rates = np.asarray(self.zero_rates, dtype=float)
        if rates.shape != tenors.shape or not np.all(np.isfinite(rates)):
            raise ValueError("zero_rates must be finite and match tenors")
        object.__setattr__(self, "tenors", tenors)
        object.__setattr__(self, "zero_rates", rates)

    def zero_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        """Return the interpolated continuously compounded zero rate."""
        x = np.asarray(maturity, dtype=float)
        if np.any(x < 0.0):
            raise ValueError("maturity cannot be negative")
        result = np.interp(x, self.tenors, self.zero_rates)
        return float(result) if x.ndim == 0 else result

    def discount(self, maturity: float | np.ndarray) -> float | np.ndarray:
        """Return discount factors at the requested maturities."""
        x = np.asarray(maturity, dtype=float)
        result = np.exp(-np.asarray(self.zero_rate(x)) * x)
        return float(result) if x.ndim == 0 else result

    def forward_rate(self, start: float, end: float) -> float:
        """Return the continuously compounded forward rate over ``[start,end]``."""
        if start < 0.0 or end <= start:
            raise ValueError("require 0 <= start < end")
        log_ratio = np.log(float(self.discount(start)) / float(self.discount(end)))
        return float(log_ratio / (end - start))

    def bumped(self, parallel_bp: float = 0.0, twist_bp: float = 0.0) -> ZeroCurve:
        """Return a curve with a parallel and linear steepener bump in basis points."""
        pivot = float(np.mean(self.tenors))
        scale = max(float(np.ptp(self.tenors)), 1.0)
        bump = (parallel_bp + twist_bp * (self.tenors - pivot) / scale) * 1.0e-4
        return replace(self, zero_rates=self.zero_rates + bump)


@dataclass(frozen=True)
class VolSurface:
    """Implied-volatility surface indexed by expiry and log-moneyness."""

    expiries: np.ndarray
    log_moneyness: np.ndarray
    vols: np.ndarray

    def __post_init__(self) -> None:
        expiries = _positive_grid(self.expiries, "expiries")
        moneyness = np.asarray(self.log_moneyness, dtype=float)
        vols = np.asarray(self.vols, dtype=float)
        if (
            moneyness.ndim != 1
            or moneyness.size == 0
            or np.any(np.diff(moneyness) <= 0.0)
            or not np.all(np.isfinite(moneyness))
        ):
            raise ValueError("log_moneyness must be finite and strictly increasing")
        if vols.shape != (expiries.size, moneyness.size):
            raise ValueError("vols shape must be (len(expiries), len(log_moneyness))")
        if not np.all(np.isfinite(vols)) or np.any(vols <= 0.0):
            raise ValueError("volatilities must be finite and positive")
        object.__setattr__(self, "expiries", expiries)
        object.__setattr__(self, "log_moneyness", moneyness)
        object.__setattr__(self, "vols", vols)

    def volatility(
        self, expiry: float | np.ndarray, log_moneyness: float | np.ndarray
    ) -> float | np.ndarray:
        """Bilinearly interpolate volatility, using flat boundary extrapolation."""
        t, k = np.broadcast_arrays(
            np.asarray(expiry, dtype=float), np.asarray(log_moneyness, dtype=float)
        )
        if np.any(t < 0.0):
            raise ValueError("expiry cannot be negative")
        flat_t, flat_k = t.ravel(), k.ravel()
        by_expiry = np.vstack(
            [np.interp(flat_k, self.log_moneyness, row) for row in self.vols]
        )
        result = np.array(
            [np.interp(x, self.expiries, by_expiry[:, i]) for i, x in enumerate(flat_t)]
        ).reshape(t.shape)
        return float(result) if result.ndim == 0 else result

    def bumped(self, absolute_vol: float) -> VolSurface:
        """Return a surface shifted by an absolute volatility amount."""
        vols = self.vols + absolute_vol
        if np.any(vols <= 0.0):
            raise ValueError("volatility bump produces non-positive volatility")
        return replace(self, vols=vols)


@dataclass(frozen=True)
class MarketState:
    """Complete market state used by pricing and scenario engines."""

    spot: float
    rate_curve: ZeroCurve
    dividend_curve: ZeroCurve
    vol_surface: VolSurface
    valuation_time: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.spot) or self.spot <= 0.0:
            raise ValueError("spot must be finite and positive")
        if self.valuation_time < 0.0:
            raise ValueError("valuation_time cannot be negative")

    def volatility(self, strike: float, expiry: float) -> float:
        """Return implied volatility for a strike and expiry."""
        if strike <= 0.0:
            raise ValueError("strike must be positive")
        return float(self.vol_surface.volatility(expiry, np.log(strike / self.spot)))

    def bumped(
        self,
        *,
        spot_relative: float = 0.0,
        vol_absolute: float = 0.0,
        rate_bp: float = 0.0,
        twist_bp: float = 0.0,
        time_years: float = 0.0,
    ) -> MarketState:
        """Create a bumped immutable market state."""
        spot = self.spot * (1.0 + spot_relative)
        if spot <= 0.0:
            raise ValueError("spot bump produces non-positive spot")
        return replace(
            self,
            spot=spot,
            rate_curve=self.rate_curve.bumped(rate_bp, twist_bp),
            vol_surface=self.vol_surface.bumped(vol_absolute),
            valuation_time=self.valuation_time + time_years,
        )


@dataclass(frozen=True)
class SyntheticMarketModel:
    """Deterministic parametric generator for market curves and surfaces."""

    base_rate: float = 0.03
    rate_slope: float = 0.005
    dividend_yield: float = 0.015
    atm_vol: float = 0.20
    skew: float = -0.08
    term_vol: float = 0.015

    def build(
        self,
        spot: float = 100.0,
        tenors: Iterable[float] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
        log_moneyness: Iterable[float] = (-0.4, -0.2, 0.0, 0.2, 0.4),
    ) -> MarketState:
        """Build a synthetic market without random sampling."""
        t = _positive_grid(tenors, "tenors")
        k = np.asarray(tuple(log_moneyness), dtype=float)
        rates = self.base_rate + self.rate_slope * (1.0 - np.exp(-t / 5.0))
        dividends = np.full_like(t, self.dividend_yield)
        vols = (
            self.atm_vol + self.skew * k[None, :] + self.term_vol * np.sqrt(t[:, None])
        )
        return MarketState(
            spot,
            ZeroCurve(t, rates),
            ZeroCurve(t, dividends),
            VolSurface(t, k, np.maximum(vols, 0.01)),
        )


def calibrate_synthetic_market(
    tenors: Iterable[float],
    observed_zero_rates: Iterable[float],
    expiries: Iterable[float],
    log_moneyness: Iterable[float],
    observed_vols: np.ndarray,
    *,
    dividend_yield: float = 0.0,
) -> SyntheticMarketModel:
    """Least-squares calibrate the deterministic synthetic model.

    The fixed starting point and deterministic trust-region solver make repeated
    calibrations bitwise stable for identical NumPy/SciPy versions.
    """
    t = _positive_grid(tenors, "tenors")
    rates = np.asarray(tuple(observed_zero_rates), dtype=float)
    e = _positive_grid(expiries, "expiries")
    k = np.asarray(tuple(log_moneyness), dtype=float)
    vols = np.asarray(observed_vols, dtype=float)
    if rates.shape != t.shape or vols.shape != (e.size, k.size):
        raise ValueError("observations do not match their calibration grids")

    def residuals(x: np.ndarray) -> np.ndarray:
        rate_fit = x[0] + x[1] * (1.0 - np.exp(-t / 5.0))
        vol_fit = x[2] + x[3] * k[None, :] + x[4] * np.sqrt(e[:, None])
        return np.concatenate((rate_fit - rates, (vol_fit - vols).ravel()))

    result = least_squares(
        residuals,
        np.array([0.02, 0.01, 0.20, -0.05, 0.01]),
        bounds=([-0.20, -0.20, 0.001, -2.0, -1.0], [0.50, 0.50, 3.0, 2.0, 1.0]),
        method="trf",
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    )
    if not result.success:
        raise RuntimeError(f"market calibration failed: {result.message}")
    return SyntheticMarketModel(
        base_rate=float(result.x[0]),
        rate_slope=float(result.x[1]),
        dividend_yield=dividend_yield,
        atm_vol=float(result.x[2]),
        skew=float(result.x[3]),
        term_vol=float(result.x[4]),
    )
