"""Shell / coating models.

Computes the total inward stress from the bubble shell, including
Laplace pressure from surface tension, viscous dissipation, and
elastic restoring forces.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from .property import Property, as_property
from .state import BubbleState

import scipy.io as sio
from interpax import Interpolator1D



class ShellModel(eqx.Module, abc.ABC):
    """Bubble shell / coating model.

    Computes the total inward stress from the shell, including:

    - Laplace pressure from surface tension:  2 sigma(R) / R
    - Shell viscous dissipation  (e.g. 4 kappa_s Rdot / R^2)
    - Shell elastic restoring forces  (for thick shells)

    Every ``ShellModel`` holds a ``Property`` as its ``sigma`` field.
    A plain float is accepted and auto-converted to a ``Property``
    in ``__post_init__``.

    """

    sigma: Property = eqx.field(converter=as_property) # input for shell model parameters can be a constant or a state-dependent property, but is always stored as a Property in the ShellModel

    def p_laplace(self, state: BubbleState) -> jax.Array:
        """Laplace pressure contribution from surface tension."""
        return 2.0 * self.sigma(state) / state.R

    @abc.abstractmethod
    def p_elastic(self, state: BubbleState) -> jax.Array:
        """Elastic contribution from the shell."""
        ...

    @abc.abstractmethod
    def p_viscous(self, state: BubbleState) -> jax.Array:
        """Viscous contribution from the shell."""
        ...

    def __call__(self, state: BubbleState) -> jax.Array:
        """Compute total shell pressure p_shell(state).

        Parameters
        ----------
        state : BubbleState
            Current bubble state.

        Returns
        -------
        scalar
            Total inward shell pressure.
        """
        return self.p_laplace(state) + self.p_elastic(state) + self.p_viscous(state)


class NoShell(ShellModel):
    """No shell coating — only Laplace pressure.

    p_shell = 2 sigma(R) / R

    Suitable for uncoated gas bubbles.  Accepts a plain float for
    ``sigma`` (e.g. 72e-3 for water).

    Fields
    ------
    sigma : float or Property
        Surface tension law.
    """

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0


class LipidShell(ShellModel):
    """Thin lipid shell with surface viscosity.

    p_shell = 2 sigma(R) / R  +  4 kappa_s Rdot / R^2

    This is the shell model used by Marmottant (2005) and most
    Gompertz-smoothed variants.

    Note on elastic contributions
    -----------------------------
    ``p_elastic`` returns zero for this model.  The shell elasticity is
    not absent — it is encoded entirely in the surface tension Property
    ``sigma``.  When ``sigma`` is state-dependent (e.g.
    ``MarmottantSurfaceTension``, ``GompertzSurfaceTension``), the
    area-elasticity term χ((R/Rb)² − 1) enters through ``p_laplace =
    2 σ(R) / R``, not through a separate ``p_elastic`` term.  This is
    consistent with how the Marmottant model is written in the
    literature: the elastic and ruptured regimes modify σ(R) rather than
    adding an independent stress contribution.

    Fields
    ------
    sigma : float or Property
        Surface tension law.
    kappa_s : float or Property
        Shell surface-dilatational viscosity  [N s/m].
    """

    kappa_s: Property = eqx.field(converter=as_property)

    def p_elastic(self, state: BubbleState) -> jax.Array:
        return state.R * 0.0

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.kappa_s(state) * state.R_dot / state.R**2


class ThickShell(ShellModel):
    """Church (1995) thick viscoelastic shell.

    In addition to Laplace pressure, this model includes thick-shell
    elastic and viscous contributions::

        p_elastic    = (4/3) G_s (d_s / R0) (1 - (R0/R)^3)
        p_shell_visc = 4 mu_s d_s Rdot / R^2

    Total shell pressure::

        p_shell = 2 sigma(R) / R  +  p_elastic  +  p_shell_visc

    Fields
    ------
    sigma : float or Property
        Surface tension law.
    d_s : float or Property
        Shell thickness  [m].
    G_s : float or Property
        Shell shear modulus  [Pa].  May be state-dependent (e.g. strain-
        stiffening / strain-softening).
    mu_s : float or Property
        Shell viscosity  [Pa s].  May be state-dependent (e.g. shear-
        thinning).
    """

    d_s: Property = eqx.field(converter=as_property)
    G_s: Property = eqx.field(converter=as_property)
    mu_s: Property = eqx.field(converter=as_property)

    def p_elastic(self, state: BubbleState) -> jax.Array:
        R = state.R
        return (
            (4.0 / 3.0)
            * self.G_s(state)
            * (self.d_s(state) / state.R0)
            * (1.0 - (state.R0 / R) ** 3)
        )

    def p_viscous(self, state: BubbleState) -> jax.Array:
        return 4.0 * self.mu_s(state) * self.d_s(state) * state.R_dot / state.R**2


class MarmottantSurfaceTension(Property):
    """Piecewise Marmottant surface tension law.

    Three regimes based on the ratio R / R_buckle::

        R <= R_buckle                :  sigma = 0               (buckled)
        R_buckle < R < R_rupture     :  sigma = chi ((R/R_b)^2 - 1)  (elastic)
        R >= R_rupture               :  sigma = sigma_rupture   (ruptured)

    where R_buckle = R_buckle_ratio * state.R0 and R_rupture is derived
    from continuity of sigma at the elastic-to-ruptured transition.

    Note: sigma(R) has discontinuous first derivatives at the regime
    boundaries.  For applications requiring smooth gradients (e.g.
    gradient-based optimisation), use ``GompertzSurfaceTension`` instead.

    Fields
    ------
    R_buckle_ratio : float
        Buckling radius as a fraction of R0  (dimensionless).
    chi : float or Property
        Shell elasticity  [N/m].
    sigma_rupture : float or Property
        Surface tension (post-rupture value)  [N/m].
    """

    R_buckle_ratio: float
    chi: float
    sigma_rupture: float

    def __call__(self, state: BubbleState) -> jax.Array:
        R, R0 = state.R, state.R0
        R_buckle = self.R_buckle_ratio * R0
        chi = self.chi
        sigma_rupture = self.sigma_rupture
        R_rupture = R_buckle * jnp.sqrt(sigma_rupture / chi + 1.0)
        sigma_elastic = chi * ((R / R_buckle) ** 2 - 1.0)
        in_elastic = (R_buckle < R) & (R_rupture > R)
        in_ruptured = R_rupture <= R
        return jnp.where(
            in_ruptured,
            sigma_rupture,
            jnp.where(
                in_elastic,
                sigma_elastic,
                0.0,
            ),
        )


class GompertzSurfaceTension(Property):
    """Smooth Gompertz surface tension law.

    A differentiable Gompertz function approximates the piecewise
    Marmottant surface tension, enabling robust automatic
    differentiation::

        sigma(R) = a exp(-b exp(c (1 - R / R_buckle)))

    where R_buckle = R_buckle_ratio * state.R0.

    The Gompertz parameters b and c are derived from chi and sigma_rupture
    such that sigma(R0) matches the elastic regime and sigma -> sigma_rupture
    as R -> infinity.  R0 is read from the state so this model stays
    consistent when R0 evolves (e.g. rectified diffusion).

    Well-posedness constraint
    -------------------------
    The Gompertz fit requires that the initial surface tension at R0 lies
    strictly below the rupture threshold::

        chi * ((1 / R_buckle_ratio)^2 - 1) < sigma_rupture

    Construction raises ``ValueError`` if this is violated.  A common
    mistake is setting R_buckle_ratio too small (e.g. 0.9), which inflates
    sigma(R0) above sigma_rupture.  Values around 0.95–0.99 are typical.

    Fields
    ------
    R_buckle_ratio : float
        Buckling radius as a fraction of R0  (dimensionless).
    chi : float or Property
        Shell elasticity  [N/m].
    sigma_rupture : float or Property
        Asymptotic (ruptured) surface tension  [N/m].
    """

    R_buckle_ratio: float
    chi: float
    sigma_rupture: float

    def __post_init__(self) -> None:
        sigma_at_R0 = self.chi * ((1.0 / self.R_buckle_ratio) ** 2 - 1.0)

        def _check(s_at_r0, s_rupture):
            if s_at_r0 >= s_rupture:
                raise ValueError(
                    f"GompertzSurfaceTension: sigma(R0) = {s_at_r0:.4g} N/m "
                    f">= sigma_rupture = {s_rupture:.4g} N/m.  "
                    f"The bubble starts in the ruptured regime and the Gompertz "
                    f"fit is ill-posed.  Increase R_buckle_ratio (try 0.98) or "
                    f"decrease chi."
                )

        jax.debug.callback(_check, sigma_at_R0, self.sigma_rupture)

    def __call__(self, state: BubbleState) -> jax.Array:
        R, R0 = state.R, state.R0
        R_buckle = self.R_buckle_ratio * R0
        chi = self.chi
        a = self.sigma_rupture
        c = (2.0 * chi / a) * jnp.sqrt(1.0 + a / (2.0 * chi))
        sigma_R0 = chi * ((R0 / R_buckle) ** 2 - 1.0)
        b = -jnp.log(sigma_R0 / a) / jnp.exp(c * (1.0 - R0 / R_buckle))
        return a * jnp.exp(-b * jnp.exp(c * (1.0 - R / R_buckle)))
    


# class SegersSurfaceTension(Property):




class SegersSurfaceTension(Property):
    mat_path: str
    sig_0: float
    sigma_rupture: float
    k: float = 1e4

    _Am1: float = eqx.field(init=False, static=True)
    _Am2: float = eqx.field(init=False, static=True)
    _Am0: float = eqx.field(init=False, static=True)
    _sigma_of_Am: Interpolator1D = eqx.field(init=False, static=True)

    def __post_init__(self) -> None:
        data = sio.loadmat(self.mat_path)
        Am_list = data["A_m"].flatten()
        sigma_list = data["sig"].flatten()

        object.__setattr__(self, "_Am1", float(Am_list.min()))
        object.__setattr__(self, "_Am2", float(Am_list.max()))
        object.__setattr__(self, "_Am0", float(jnp.interp(self.sig_0, jnp.array(sigma_list), jnp.array(Am_list))))
        object.__setattr__(self, "_sigma_of_Am", Interpolator1D(Am_list, sigma_list, method="monotonic"))

    def __call__(self, state: BubbleState) -> jax.Array:
        R, R0 = state.R, state.R0
        AN = 4.0 * jnp.pi * R0 ** 2 / self._Am0
        Am = 4.0 * jnp.pi * R ** 2 / AN

        # Clamp to valid interpolation range before evaluating
        Am_clamped = jnp.clip(Am, self._Am1, self._Am2)
        sig_interp = self._sigma_of_Am(Am_clamped)

        w_lower = jax.nn.sigmoid(self.k * (Am - self._Am1))
        w_upper = jax.nn.sigmoid(self.k * (Am - self._Am2))

        return w_lower * ((1.0 - w_upper) * sig_interp + w_upper * self.sigma_rupture)
    

class SegersSurfaceTensionPolyfit(Property):
    """Differentiable Segers surface tension law (polynomial fit).

    Evaluates the polynomial fit to the experimental sigma(A_m) curve from
    Segers et al., Soft Matter, 14, 2018, using double-precision coefficients
    obtained from Tim Segers.

    The domain boundaries Am1 and Am2 are derived analytically from the
    polynomial:
    - Am1: root of the polynomial closest to 0.92 (sigma crosses zero)
    - Am2: root of the derivative closest to 1.12 (maximum of the polynomial)

    Regime boundaries are blended with sigmoid functions for full
    differentiability.

    Fields
    ------
    mat_path : str
        Path to fit_SigmaR_04-08-2017.mat containing fit.coeff.
    sig_0 : float
        Reference surface tension at R0  [N/m].
    sigma_rupture : float
        Surface tension in the ruptured regime  [N/m].
    k : float
        Sigmoid sharpness for regime blending (default 1e4).
    """

    mat_path: str
    sig_0: float
    sigma_rupture: float
    k: float = 1e4

    _coeffs: jax.Array = eqx.field(init=False)
    _Am1: float = eqx.field(init=False, static=True)
    _Am2: float = eqx.field(init=False, static=True)
    _Am0: float = eqx.field(init=False, static=True)

    def __post_init__(self) -> None:
        data = sio.loadmat(self.mat_path, simplify_cells=True)
        coeffs = jnp.array(data["fit"]["coeff"]).flatten()

        # Am1: root of polynomial closest to 0.92 (sigma = 0 crossing)
        roots_poly = jnp.roots(coeffs, strip_zeros=False)
        roots_real = jnp.where(jnp.imag(roots_poly) == 0, jnp.real(roots_poly), jnp.nan)
        Am1 = float(roots_real[jnp.nanargmin(jnp.abs(roots_real - 0.92))])

        # Am2: root of derivative closest to 1.12 (maximum of polynomial)
        roots_deriv = jnp.roots(jnp.polyder(coeffs), strip_zeros=False)
        roots_deriv_real = jnp.where(jnp.imag(roots_deriv) == 0, jnp.real(roots_deriv), jnp.nan)
        Am2 = float(roots_deriv_real[jnp.nanargmin(jnp.abs(roots_deriv_real - 1.12))])

        # Am0: shift polynomial down by sig_0 and find root in [Am1, Am2]
        coeffs_shifted = coeffs.at[-1].add(-self.sig_0)
        roots_shifted = jnp.roots(coeffs_shifted)
        roots_shifted_real = jnp.real(roots_shifted)
        in_domain = (jnp.imag(roots_shifted) == 0) & (roots_shifted_real > Am1) & (roots_shifted_real < Am2)
        roots_in_domain = jnp.where(in_domain, roots_shifted_real, jnp.nan)
        n_solutions = int(jnp.sum(in_domain))
        if n_solutions != 1:
            raise ValueError(
                f"SegersSurfaceTension: expected 1 solution for Am0 at "
                f"sig_0={self.sig_0:.4g} N/m, found {n_solutions}. "
                f"Check that sig_0 is within the elastic regime."
            )
        Am0 = float(jnp.nanmin(roots_in_domain))  # only one non-nan value

        object.__setattr__(self, "_coeffs", coeffs)
        object.__setattr__(self, "_Am1", Am1)
        object.__setattr__(self, "_Am2", Am2)
        object.__setattr__(self, "_Am0", Am0)

    def __call__(self, state: BubbleState) -> jax.Array:
        R, R0 = state.R, state.R0

        AN = 4.0 * jnp.pi * R0**2 / self._Am0
        Am = 4.0 * jnp.pi * R**2 / AN

        # Clamp before polynomial evaluation to avoid blow-up outside domain
        Am_clamped = jnp.clip(Am, self._Am1, self._Am2)
        sig_interp = jnp.polyval(self._coeffs, Am_clamped)

        w_lower = jax.nn.sigmoid(self.k * (Am - self._Am1))
        w_upper = jax.nn.sigmoid(self.k * (Am - self._Am2))

        return w_lower * ((1.0 - w_upper) * sig_interp + w_upper * self.sigma_rupture)