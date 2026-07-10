"""Acoustic emission models for bubble dynamics.

Computes the radiated acoustic pressure at a field point from a solved
bubble trajectory.  Each model captures a different level of physical
fidelity (incompressible monopole → quasi-acoustic → fully compressible).

Usage::

    from jbubble.acoustics import IncompressibleMonopole

    emission = IncompressibleMonopole(rho_L=998.0)
    p_rad = emission(result, r=0.01)  # at 1 cm
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..simulation import SimulationResult


class EmissionModel(eqx.Module, abc.ABC):
    """Acoustic emission model: bubble trajectory → radiated pressure.

    Subclasses implement ``__call__`` which takes a solved
    :class:`~jbubble.simulation.SimulationResult` and a field-point
    distance *r* and returns the radiated pressure time series.

    Multiple field-point distances are handled naturally via
    ``jax.vmap``::

        distances = jnp.array([0.001, 0.005, 0.01])
        p_all = jax.vmap(lambda r: model(result, r))(distances)
        # shape (3, N)
    """

    @abc.abstractmethod
    def __call__(
        self,
        result: SimulationResult,
        r: ArrayLike,
        alpha: ArrayLike = None,
    ) -> jax.Array:
        """Compute radiated pressure at distance *r*.

        Parameters
        ----------
        result : SimulationResult
            Solved bubble trajectory (``state``, ``state_dot``, ``ts``).
        r : float or jax.Array
            Distance from the bubble centre to the field point [m].

        Returns
        -------
        jax.Array, shape (N,)
            Radiated pressure [Pa] at each saved time point.
        """
        ...


class IncompressibleMonopole(EmissionModel):
    """Incompressible monopole radiation.

    Assumes an incompressible surrounding liquid so that the radiated
    pressure at distance *r* is given by the time derivative of the
    volume flux:

    ::

        p_rad(r, t) = rho_L / r · d/dt(R² Ṙ)
                     = rho_L / r · (2 R Ṙ² + R² R̈)

    This is the simplest acoustic emission model and is accurate when
    the bubble-wall Mach number Ṁ = Ṙ / c_L ≪ 1 and the field point
    is in the geometric near-field (r ≪ c_L / f).

    Fields
    ------
    rho_L : float or jax.Array
        Liquid density [kg/m³].
    """

    rho_L: ArrayLike

    def __call__(
        self,
        result: SimulationResult,
        r: ArrayLike,
    ) -> jax.Array:
        R = result.state.R
        R_dot = result.state.R_dot
        R_ddot = result.state_dot.R_dot
        return (
            jnp.asarray(self.rho_L)
            / jnp.asarray(r)
            * (2.0 * R * R_dot**2 + R**2 * R_ddot)
        )


class QuasiAcoustic(EmissionModel):
    """Quasi-acoustic emission with retarded-time correction.

    Accounts for the finite speed of sound by evaluating the bubble-wall
    quantities at the retarded time t_ret = t − r / c_L:

    ::

        p_rad(r, t) = rho_L R²(t_ret) / r
                      · [R̈(t_ret) + 2 Ṙ²(t_ret) / R(t_ret)]

    Uses linear interpolation (``jnp.interp``) to evaluate the
    trajectory at retarded times.  For field points where
    t_ret < t_start the values are clamped to the initial (equilibrium)
    state — physically reasonable since the bubble is quiescent before
    excitation.

    Fields
    ------
    rho_L : float or jax.Array
        Liquid density [kg/m³].
    c_L : float or jax.Array
        Speed of sound in the liquid [m/s].
    """

    rho_L: ArrayLike
    c_L: ArrayLike

    def __call__(
        self,
        result: SimulationResult,
        r: ArrayLike,
    ) -> jax.Array:
        delay = r / self.c_L
        t_ret = result.ts - delay

        # Interpolate bubble-wall quantities at retarded times.
        R_ret = jnp.interp(t_ret, result.ts, result.state.R)
        R_dot_ret = jnp.interp(t_ret, result.ts, result.state.R_dot)
        R_ddot_ret = jnp.interp(t_ret, result.ts, result.state_dot.R_dot)


        return (
            jnp.asarray(self.rho_L)
            * R_ret**2
            / jnp.asarray(r)
            * (R_ddot_ret + 2.0 * R_dot_ret**2 / R_ret)
        )
    

class QuasiAcoustic_Attenuated(EmissionModel):
    """Quasi-acoustic emission with retarded-time correction.

    Accounts for the finite speed of sound by evaluating the bubble-wall
    quantities at the retarded time t_ret = t − r / c_L:

    ::

        p_rad(r, t) = rho_L R²(t_ret) / r
                      · [R̈(t_ret) + 2 Ṙ²(t_ret) / R(t_ret)]

    Uses linear interpolation (``jnp.interp``) to evaluate the
    trajectory at retarded times.  For field points where
    t_ret < t_start the values are clamped to the initial (equilibrium)
    state — physically reasonable since the bubble is quiescent before
    excitation.

    Fields
    ------
    rho_L : float or jax.Array
        Liquid density [kg/m³].
    c_L : float or jax.Array
        Speed of sound in the liquid [m/s].
    """

    rho_L: ArrayLike
    c_L: ArrayLike


    def attenuation_factor(self,
            freq_hz: jax.Array,
            r:ArrayLike,
            alpha: ArrayLike
            ) -> jax.Array:
        attenuation_db = alpha * (freq_hz / 1e6) * (r * 100)  #assume r in m
        return 10 ** (-attenuation_db / 20)

    def __call__(
        self,
        result: SimulationResult,
        r: ArrayLike,
        alpha: ArrayLike
        ) -> jax.Array:
        delay = r / self.c_L
        ts = result.ts
        t_ret = result.ts - delay

        # Interpolate bubble-wall quantities at retarded times.
        R_ret = jnp.interp(t_ret, result.ts, result.state.R)
        R_dot_ret = jnp.interp(t_ret, result.ts, result.state.R_dot)
        R_ddot_ret = jnp.interp(t_ret, result.ts, result.state_dot.R_dot)


        Pscat = (
            jnp.asarray(self.rho_L)
            * R_ret**2
            / jnp.asarray(r)
            * (R_ddot_ret + 2.0 * R_dot_ret**2 / R_ret)
        )

        Pscat_fft = jnp.fft.rfft(Pscat)
        freqs = jnp.fft.rfftfreq(Pscat.size, d=(result.ts[1] - result.ts[0]))
        attenuation = self.attenuation_factor(freq_hz = freqs, r = r, alpha = alpha)
        Pscat_fft_attenuated = Pscat_fft * attenuation
        Pscat_attenuated = jnp.fft.irfft(Pscat_fft_attenuated, n=Pscat.size)

        return Pscat_attenuated

