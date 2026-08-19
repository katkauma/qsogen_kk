#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Object-oriented IGM absorption models.

The main interface is:

    model = Inoue2014()
    tau = model.calculate_tau(z, wavelength)
    transmission = model.transmission(z, wavelength)

Individual optical-depth components are accessible through:

    tau.laf_lines
    tau.laf_continuum
    tau.dla_lines
    tau.dla_continuum

and components can be excluded from the transmission calculation.
"""
import os
import numpy as np
from mpmath import gammainc
from scipy.special import factorial
from abc import ABC, abstractmethod
from dataclasses import dataclass

install_path = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Constants
# ============================================================

C = 2.998e5          # speed of light, km/s
H0 = 70.
OMEGA_M = 0.3

# Lyman-series line data: order n, rest wavelength (A), oscillator strength
ns, linewvs, osc_strs =  np.genfromtxt(f'{install_path}/data/lyman_series_info.txt', skip_header=1, unpack=True)
maxlines = len(ns)


def _gammainc_reg(z, a=0, b=np.inf):
    """Regularized incomplete gamma function, scalar version."""
    return gammainc(z, a, b)


# Vectorized wrapper so mpmath's gammainc can be applied to arrays.
gammainc_vec = np.frompyfunc(_gammainc_reg, 3, 1)


# ============================================================
# Low-level physics functions
# ============================================================

def triple_powerlaw(z, z1, z2, C1, C2, C3, a1, a2, a3):
    return np.piecewise(
        z,
        [z < z1, (z >= z1) & (z < z2), z >= z2],
        [lambda z: C1 * (1 + z) ** a1,
         lambda z: C2 * (1 + z) ** a2,
         lambda z: C3 * (1 + z) ** a3],
    )


def double_powerlaw(z, z1, C1, C2, a1, a2):
    return np.piecewise(
        z,
        [z < z1, z >= z1],
        [lambda z: C1 * (1 + z) ** a1,
         lambda z: C2 * (1 + z) ** a2],
    )


def single_powerlaw(z, C1, a1):
    return C1 * (1 + z) ** a1


# ============================================================
# Lyman series calculation - sum tau_n = R_n * tau_alpha
# ============================================================

def calculate_series_tau(obswav, z, line_wavelengths, line_ratios, tau_lya_fn, num_lines):

    line_wavelengths = line_wavelengths[:num_lines]
    line_ratios = line_ratios[:num_lines]

    z_look = (obswav[None, :] / line_wavelengths[:, None]) - 1.0

    tau_lya_values = np.zeros_like(z_look)
    mask = z_look < z  # only compute tau where lookback redshift is below the source redshift
    tau_lya_values[mask] = tau_lya_fn(z_look[mask])

    tau = tau_lya_values * line_ratios[:, None]

    return np.sum(tau, axis=0)


# ============================================================
# Optical-depth container
# ============================================================

@dataclass
class TauComponents:
    """
    Stores the individual optical-depth contributions.

    All quantities should have the same shape as the input
    wavelength array.
    """

    laf_lines: np.ndarray
    laf_continuum: np.ndarray
    dla_lines: np.ndarray
    dla_continuum: np.ndarray

    @property
    def total(self):
        """Total optical depth."""
        return (
            self.laf_lines
            + self.laf_continuum
            + self.dla_lines
            + self.dla_continuum
        )

    @property
    def line(self):
        """All line absorption."""
        return self.laf_lines + self.dla_lines

    @property
    def continuum(self):
        """All continuum absorption."""
        return self.laf_continuum + self.dla_continuum

    def transmission(self, exclude=None):
        """
        Convert optical depth into transmission.

        Parameters
        ----------
        exclude : list of str, optional
            Components to leave out.

        Examples
        --------
        tau.transmission()

        tau.transmission(
            exclude=["dla_lines"]
        )

        tau.transmission(
            exclude=["laf_continuum", "dla_lines"]
        )
        """

        exclude = set(exclude or [])

        components = {
            "laf_lines": self.laf_lines,
            "laf_continuum": self.laf_continuum,
            "dla_lines": self.dla_lines,
            "dla_continuum": self.dla_continuum,
        }

        tau = np.zeros_like(self.total)
        for name, component in components.items():
            if name not in exclude:
                tau += component

        return np.exp(-tau)


# ============================================================
# Base IGM model
# ============================================================

class IGMModel(ABC):
    """
    Base class for all IGM models.

    Subclasses implement the individual physical components.
    """
    model_name = ""
    model_pub = ""
    model_doi = ""

    LINE_CROSS_SECTION_B = 28.  # Doppler b-parameter (km/s) used for line-centre cross sections

    @property
    @abstractmethod
    def num_lines_max(self):
        """Maximum number of Lyman-series lines supported by this model."""
        pass

    @property
    def line_wavelengths(self):
        """Wavelengths of the Lyman-series lines in Angstroms."""
        return linewvs[:self.num_lines_max]

    @property
    def line_oscillator_strengths(self):
        """Oscillator strengths of the Lyman-series lines."""
        return osc_strs[:self.num_lines_max]

    @property
    def line_cross_sections(self):
        """Line-centre cross sections of the Lyman-series lines in cm^2."""
        K = 0.0149743642  # sqrt(pi) * e^2 / (m_e * c)
        return K * self.line_oscillator_strengths * self.line_wavelengths / self.LINE_CROSS_SECTION_B * 1e-13

    def calculate_tau(self, z, wavelength, num_lines=None):
        """Calculate all optical-depth components."""
        num_lines = self._validate_num_lines(num_lines)

        return TauComponents(
            laf_lines=self.tau_laf_lines(z, wavelength, num_lines=num_lines),
            laf_continuum=self.tau_laf_continuum(z, wavelength),
            dla_lines=self.tau_dla_lines(z, wavelength, num_lines=num_lines),
            dla_continuum=self.tau_dla_continuum(z, wavelength),
        )

    def transmission(self, z, wavelength, exclude=None, num_lines=None):
        """
        Calculate transmission.

        Parameters
        ----------
        wavelength : array-like
            Observed wavelength in Angstroms.

        z : float
            Source redshift.

        exclude : list of str, optional
            Optical-depth components to exclude.
        """

        tau = self.calculate_tau(z, wavelength, num_lines=num_lines)

        return tau.transmission(exclude=exclude)

    def calculate_transmission_grid(self, z_grid, wavelength, exclude=None, num_lines=None):
        """
        Calculate transmission on a grid of redshifts and wavelengths.

        Internally this just calls `transmission()` once per redshift in
        `z_grid`, since the tau_* component functions are written for a
        scalar source redshift $z$. It's meant for building a lookup table
        of $T(z, \\lambda)$ once, up front, so that e.g. photometric-redshift
        fitting can interpolate the grid instead of paying for a fresh IGM
        calculation at every trial redshift.

        Parameters
        ----------
        z_grid : array-like, shape (n_z,)
            Source redshifts at which to evaluate the transmission.
        wavelength : array-like, shape (n_wave,)
            Observed-frame wavelength grid in Angstroms, shared by every
            redshift in `z_grid`.
        exclude : list of str, optional
            Optical-depth components to exclude, passed through to
            `transmission`.
        num_lines : int, optional
            Number of Lyman-series lines to include, passed through to
            `transmission`.

        Returns
        -------
        transmission_grid : ndarray, shape (n_z, n_wave)
            $T(z_i, \\lambda_j)$ = ``transmission_grid[i, j]``.
        """

        z_grid = np.atleast_1d(z_grid)
        wavelength = np.atleast_1d(wavelength)

        transmission_grid = np.empty((len(z_grid), len(wavelength)))

        for i, z in enumerate(z_grid):
            transmission_grid[i] = self.transmission(
                z, wavelength, exclude=exclude, num_lines=num_lines
            )

        return transmission_grid

    def transmission_interpolator(self, z_grid, wavelength, exclude=None, num_lines=None, **interp_kwargs):
        """
        Build a `scipy.interpolate.RegularGridInterpolator` over $T(z, \\lambda)$.

        Convenience wrapper around `calculate_transmission_grid` for the
        common case of wanting an interpolator object directly (e.g. for
        photo-z fitting) rather than the raw grid.

        Parameters
        ----------
        z_grid, wavelength, exclude, num_lines :
            See `calculate_transmission_grid`.
        **interp_kwargs :
            Passed through to `RegularGridInterpolator`, e.g.
            ``bounds_error=False, fill_value=None`` to linearly extrapolate,
            or ``method='cubic'`` for smoother interpolation.

        Returns
        -------
        interpolator : scipy.interpolate.RegularGridInterpolator
            Call as ``interpolator((z, wavelength))`` — see scipy docs for
            the expected input shape when querying multiple points at once.
        """

        from scipy.interpolate import RegularGridInterpolator

        z_grid = np.atleast_1d(z_grid)
        wavelength = np.atleast_1d(wavelength)

        transmission_grid = self.calculate_transmission_grid(
            z_grid, wavelength, exclude=exclude, num_lines=num_lines
        )

        return RegularGridInterpolator((z_grid, wavelength), transmission_grid, **interp_kwargs)

    def _validate_num_lines(self, num_lines):

        if num_lines is None:
            return self.num_lines_max

        if not isinstance(num_lines, (int, np.integer)):
            raise TypeError("num_lines must be an integer or None")

        if num_lines < 1:
            raise ValueError("num_lines must be >= 1")

        if num_lines > self.num_lines_max:
            raise ValueError(
                f"num_lines={num_lines} exceeds the maximum "
                f"of {self.num_lines_max} for {type(self).__name__}"
            )

        return num_lines

    # --------------------------------------------------------
    # Components - all return zero unless specified when subclassing,
    # except tau_laf_lines and tau_lya_laf which every subclass must define.
    # --------------------------------------------------------

    def tau_laf_lines(self, z, wavelength, num_lines):
        return calculate_series_tau(
            wavelength, z,
            self.line_wavelengths, self.laf_line_ratios,
            self.tau_lya_laf, num_lines,
        )

    def tau_dla_lines(self, z, wavelength, num_lines):
        return np.zeros_like(wavelength)

    @abstractmethod
    def tau_lya_laf(self, z):
        pass

    def tau_lya_dla(self, z):
        pass

    def tau_laf_continuum(self, z, wavelength):
        """Lyman-alpha forest continuum absorption."""
        return np.zeros_like(wavelength)

    def tau_dla_continuum(self, z, wavelength):
        """DLA continuum absorption."""
        return np.zeros_like(wavelength)


##################################
# subclasses
##################################

# Inoue2014 model
class Inoue2014(IGMModel):
    model_name = "Inoue2014"
    model_pub = "Inoue et al. (2014)"
    model_doi = ""

    laf_line_ratios = np.array([
        1.000e+00, 2.776e-01, 1.325e-01, 7.805e-02, 5.152e-02, 3.656e-02,
        2.727e-02, 2.112e-02, 1.682e-02, 1.372e-02, 1.138e-02, 9.598e-03,
        8.195e-03, 7.077e-03, 6.172e-03, 5.428e-03, 4.809e-03, 4.291e-03,
        3.849e-03, 3.472e-03, 3.147e-03, 2.866e-03, 2.620e-03, 2.404e-03,
        2.212e-03, 2.044e-03, 1.893e-03, 1.758e-03, 1.637e-03, 1.528e-03,
        1.429e-03, 1.339e-03, 1.258e-03, 1.183e-03, 1.115e-03, 1.053e-03,
        9.953e-04, 9.426e-04, 8.935e-04,
    ])

    dla_line_ratios = np.array([
        1.0000, 0.9555, 0.9264, 0.9029, 0.8837, 0.8670, 0.8516, 0.8380,
        0.8256, 0.8139, 0.8027, 0.7922, 0.7823, 0.7730, 0.7644, 0.7557,
        0.7477, 0.7403, 0.7328, 0.7254, 0.7186, 0.7118, 0.7050, 0.6988,
        0.6926, 0.6865, 0.6809, 0.6747, 0.6691, 0.6636, 0.6586, 0.6531,
        0.6481, 0.6432, 0.6382, 0.6333, 0.6289, 0.6240, 0.6197,
    ])

    def __init__(self, num_lines_max=39, name=None):
        self._num_lines_max = num_lines_max
        self.tau_lya_laf_params = dict(
            z1=1.2, z2=4.7, C1=1.690e-2, C2=2.354e-3, C3=1.026e-4, a1=1.2, a2=3.7, a3=5.5
        )
        self.tau_lya_dla_params = dict(z1=2., C1=1.617e-4, C2=5.390e-5, a1=2.0, a2=3.0)
        if name is not None:
            self.model_name = name

    @property
    def num_lines_max(self):
        return self._num_lines_max

    def tau_lya_laf(self, z):
        return triple_powerlaw(z, **self.tau_lya_laf_params)

    def tau_lya_dla(self, z):
        return double_powerlaw(z, **self.tau_lya_dla_params)

    def tau_dla_lines(self, z, wavelength, num_lines):
        return calculate_series_tau(
            wavelength, z,
            self.line_wavelengths, self.dla_line_ratios,
            self.tau_lya_dla, num_lines,
        )

    def tau_laf_continuum(self, z, wavelength):
        zs_p1 = 1. + z
        lratio = wavelength / 912.

        if z < 1.2:
            zs_term = zs_p1 ** (-0.9)
            tau_laf = 0.325 * (lratio ** 1.2 - zs_term * lratio ** 2.1)

        elif z < 4.7:
            zs_term = zs_p1 ** 1.6
            tau_laf = np.piecewise(
                lratio, [lratio < 2.2, lratio >= 2.2],
                [lambda lratio: 2.55e-2 * zs_term * lratio ** 2.1 + 0.325 * lratio ** 1.2 - 0.25 * lratio ** 2.1,
                 lambda lratio: 2.55e-2 * (zs_term * lratio ** 2.1 - lratio ** 3.7)],
            )

        else:
            zs_term = zs_p1 ** 3.4
            tau_laf = np.piecewise(
                lratio, [lratio < 2.2, (lratio >= 2.2) & (lratio < 5.7), lratio >= 5.7],
                [lambda lratio: 5.22e-4 * zs_term * lratio ** 2.1 + 0.325 * lratio ** 1.2 - 3.14e-2 * lratio ** 2.1,
                 lambda lratio: 5.22e-4 * zs_term * lratio ** 2.1 + 0.218 * lratio ** 2.1 - 2.55e-2 * lratio ** 3.7,
                 lambda lratio: 5.22e-4 * (zs_term * lratio ** 2.1 - lratio ** 5.5)],
            )

        return np.where(lratio <= zs_p1, tau_laf, 0.)

    def tau_dla_continuum(self, z, wavelength):
        zs_p1 = 1. + z
        lratio = wavelength / 912.

        if z < 2.0:
            tau_dla = 0.211 * zs_p1 ** 2.0 - 7.66e-2 * zs_p1 ** 2.3 * lratio ** (-0.3) - 0.135 * lratio ** 2.
        else:
            zs_term1 = zs_p1 ** 3.0
            zs_term2 = zs_p1 ** 3.3
            tau_dla = np.piecewise(
                lratio, [lratio < 3., lratio >= 3.],
                [lambda lratio: 0.634 + 4.7e-2 * zs_term1 - 1.78e-2 * zs_term2 * lratio ** (-0.3)
                 - 0.135 * lratio ** 2. - 0.291 * lratio ** (-0.3),
                 lambda lratio: 4.7e-2 * zs_term1 - 1.78e-2 * zs_term2 * lratio ** (-0.3) - 2.92e-2 * lratio ** 3.],
            )

        return np.where(lratio <= zs_p1, tau_dla, 0.)


####################
# Madau1995 model
class Madau1995(IGMModel):
    model_name = "Madau1995"
    model_pub = "Madau (1995)"
    model_doi = ""


    laf_line_ratios = np.array([
        0.0036, 0.0017, 0.0011846, 0.0009410, 0.0007960, 0.0006967,
        0.0006236, 0.0005665, 0.0005200, 0.0004817, 0.0004487, 0.0004200,
        0.0003947, 0.000372, 0.000352, 0.0003334, 0.00031644,
    ]) / 0.0036

    def __init__(self, name=None):
        self._num_lines_max = 17
        self.tau_lya_laf_params = dict(C1=0.0036, a1=3.46)
        if name is not None:
            self.model_name = name

    @property
    def num_lines_max(self):
        return self._num_lines_max

    def tau_lya_laf(self, z):
        return single_powerlaw(z, **self.tau_lya_laf_params)

    def tau_laf_continuum(self, z, wavelength):
        zs_p1 = 1. + z
        lratio = wavelength / 912
        tau = (
            0.25 * lratio ** 3. * (zs_p1 ** 0.46 - lratio ** 0.46)
            + 9.4 * lratio ** 1.5 * (zs_p1 ** 0.18 - lratio ** 0.18)
            - 0.7 * lratio ** 3. * (lratio ** (-1.32) - zs_p1 ** (-1.32))
            - 0.023 * (zs_p1 ** 1.68 - lratio ** 1.68)
        )
        return np.where(lratio <= zs_p1, tau, 0.)


####################
# Meiksin2006 model
class Meiksin2006(IGMModel):
    model_name = "Meiksin2006"
    model_pub = "Meiksin (2006)"
    model_doi = ""

    laf_line_ratios = np.array([
        1., 0.348, 0.179, 0.109, 0.0722, 0.0508, 0.0373, 0.0283,  # n=2 to n=9
        0.02058182, 0.01543636, 0.01187413, 0.00932967,
        0.00746374, 0.00606429, 0.00499412, 0.00416176, 0.00350464,  # n=10-31; t/tau_alpha = 20.376/(n(n^2-1))
        0.00297895, 0.00255338, 0.00220519, 0.00191756, 0.00167787,
        0.00147652, 0.00130615, 0.00116103, 0.00103663, 0.00092939,
        0.00083645, 0.00075551, 0.00068468,
    ])

    def __init__(self, name=None):
        self._num_lines_max = 30
        self.tau_lya_laf_params = dict(C1=0.00211, C2=0.00058, a1=3.7, a2=4.5, z1=4.)
        self.tau_lya_dla_params = None
        if name is not None:
            self.model_name = name

    @property
    def num_lines_max(self):
        return self._num_lines_max

    def tau_lya_laf(self, z):
        return double_powerlaw(z, **self.tau_lya_laf_params)

    def tau_laf_continuum(self, z, wavelength):
        lratio = wavelength / 912.
        zs_p1 = 1. + z
        return np.where(lratio <= zs_p1, 0.805 * lratio ** 3. * (1. / lratio - 1. / zs_p1), 0.)

    def tau_dla_continuum(self, z, wavelength):
        lratio = wavelength / 912.
        zs_p1 = 1. + z
        gamma = 0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
        n0 = 0.25
        n = np.arange(9)  # first 10 terms cause convergence

        term1 = gamma - np.exp(-1.)
        term2 = np.sum(np.power(-1., n) / (factorial(n) * (2. * n - 1.)))
        term3 = zs_p1 * lratio ** 1.5 - lratio ** 2.5

        n_ext = np.arange(1, 10)
        term4 = np.sum(
            (2. * np.power(-1., n_ext) / (factorial(n_ext) * ((6. * n_ext - 5.) * (2. * n_ext - 1))))[:, None]
            * (zs_p1 ** (2.5 - 3. * n_ext)[:, None] * lratio[None, :] ** (3. * n_ext)[:, None] - lratio[None, :] ** 2.5),
            axis=0,
        )

        return np.where(lratio <= zs_p1, n0 * ((term1 - term2) * term3 - term4), 0.)

####################
# Temple2021 model
class Temple2021(IGMModel):
    model_name = "Temple2021"
    model_pub = "Temple et al. (2021)"
    model_doi = ""
    notes = "Used in the original version of QSOGEN code."

    laf_line_ratios = np.array([1.,0.16, 0.056])

    def __init__(self, name=None):

        self._num_lines_max = 3
        if name is not None:
            self.model_name = name
        
    @property
    def num_lines_max(self):
        return self._num_lines_max

    def tau_lya_laf(self, z):
        return 0.751*((1+z)/(4.5))**2.90 - 0.132

    def tau_laf_continuum(self, z, wavelength):
        ll_rest = 912.
        ll_zs = ll_rest*(1+z)
        return np.where(wavelength<ll_zs, np.inf, 0.)



####################
# Kauma_flex model
class Kauma_flex(IGMModel):
    """
    Flexible IGM model with adjustable Lyman-alpha forest and DLA parameters.

    Most quantities that only depend on the model parameters (and not on
    z or wavelength) are pre-computed once here in __init__ and cached as
    instance attributes, rather than being recomputed on every call to
    tau_laf_continuum / tau_dla_continuum.
    """
    model_name = "Kauma"
    model_pub = "Kauma et al. (in prep.)"
    model_doi = ""

    def __init__(self,
                 tau_lya_laf_params=dict(A=0.05, z1=1.30, z2=5.02, g1=1.44, g2=3.65, g3=6.77),
                 laf_gN_params=dict(beta=1.69, logNmin=12, logNmax=17.5),
                 laf_mfp_params=dict(A=16.51, H0=70, h70=1, Om=0.3, z0=4.93, eta1=4.27, eta2=17.73),
                 dla_nz_params=dict(n0=0.138, g=1.7, d=5., b=28.),
                 dla_gN_params=dict(beta=1.01, logNb=21.16, logNmin=17.5),
                 ksum=5,
                 LL_alpha=2.75,
                 num_lines_max=39,
                 name = None):

        if name is not None:
            self.model_name = name
        self.tau_lya_laf_params = tau_lya_laf_params
        self.laf_gN_params = laf_gN_params
        self.laf_mfp_params = laf_mfp_params
        self.dla_nz_params = dla_nz_params
        self.dla_gN_params = dla_gN_params

        self._num_lines_max = num_lines_max
        self.LL_alpha = LL_alpha
        self.ksum = ksum

        # z-dependence coefficients for the tau_lya(z) power laws
        self.tau_lya_laf_coeffs = self._calc_tau_lya_laf_coeffs(**tau_lya_laf_params)
        self.tau_lya_dla_coeffs = self._calc_tau_lya_dla_coeffs(**dla_nz_params)

        # coefficients for the LAF/DLA continuum terms (z, wavelength independent)
        self._mfp_coeffs = self._calc_mfp_coeffs(**laf_mfp_params, alpha=LL_alpha)
        self._dla_continuum_coeffs = self._calc_dla_continuum_coeffs(
            **dla_gN_params, ksum=ksum, alpha=LL_alpha, g=dla_nz_params['g'],
        )

        # per-line ratios R_n, also independent of z and wavelength
        self.laf_line_ratios = self._analytic_Rn_LAF(self.line_cross_sections, **laf_gN_params)
        self.dla_line_ratios = self._analytic_Rn_DLA(self.line_cross_sections, **dla_gN_params)

    @property
    def num_lines_max(self):
        return self._num_lines_max

    # --------------------------------------------------------
    # Lyman-alpha optical depth vs. redshift
    # --------------------------------------------------------

    def _calc_tau_lya_laf_coeffs(self, A, z1, z2, g1, g2, g3):
        C1 = A / (1 + z1) ** g1
        C2 = A / (1 + z1) ** g2
        C3 = A / (1 + z2) ** g3 * ((1 + z2) / (1 + z1)) ** g2
        return dict(z1=z1, z2=z2, C1=C1, C2=C2, C3=C3, a1=g1, a2=g2, a3=g3)

    def _calc_tau_lya_dla_coeffs(self, n0, b, d, g):
        return dict(C1=n0 * d * b / C, a1=g + 1.)

    def tau_lya_laf(self, z):
        return triple_powerlaw(z, **self.tau_lya_laf_coeffs)

    def tau_lya_dla(self, z):
        return single_powerlaw(z, **self.tau_lya_dla_coeffs)

    def tau_dla_lines(self, z, wavelength, num_lines):
        return calculate_series_tau(
            wavelength, z,
            self.line_wavelengths, self.dla_line_ratios,
            self.tau_lya_dla, num_lines,
        )

    # --------------------------------------------------------
    # Line-ratio (column-density-integrated) factors R_n
    # --------------------------------------------------------

    def _analytic_Rn_LAF(self, xs, beta, logNmin, logNmax):
        norm = (xs ** (beta - 1) * (beta - 1)) / (10 ** (logNmin * (1 - beta)) - 10 ** (logNmax * (1 - beta)))
        integral = np.array(
            gammainc_vec(1 - beta, 0, xs * 10 ** logNmax) - gammainc_vec(1 - beta, 0, xs * 10 ** logNmin),
            dtype=np.float64,
        )
        Is = 1 - norm * integral
        return Is / Is[0]

    def _analytic_Rn_DLA(self, xs, beta, logNb, logNmin):
        norm = (10 ** logNb * xs) ** (beta - 1) / np.array(
            gammainc_vec(1 - beta, 10 ** (logNmin - logNb), np.inf), dtype=np.float64,
        )
        integral = np.array(gammainc_vec(1 - beta, xs * 10 ** logNmin, np.inf), dtype=np.float64)
        Is = 1 - norm * integral
        return Is / Is[0]

    # --------------------------------------------------------
    # LAF continuum (mean-free-path formulation)
    # --------------------------------------------------------

    def _calc_mfp_coeffs(self, A, H0, h70, Om, z0, eta1, eta2, alpha):
        K = C / (A * H0 / h70 * np.sqrt(Om))

        g1 = eta1 - alpha - 3. / 2.
        g2 = eta2 - alpha - 3. / 2.

        c1 = (1 + z0) ** (-eta1) / g1
        c2 = (1 + z0) ** (-eta2) / g2

        return dict(K=K, g1=g1, g2=g2, c1=c1, c2=c2, z0=z0, alpha=alpha)

    def tau_laf_continuum(self, z, wavelength):
        K, g1, g2, c1, c2, z0, alpha = (
            self._mfp_coeffs[k] for k in ('K', 'g1', 'g2', 'c1', 'c2', 'z0', 'alpha')
        )

        zlook = wavelength / 912. - 1.
        below_source = zlook <= z

        sum1 = c1 * np.where(below_source, (1 + min(z, z0)) ** g1 - (1 + np.minimum(zlook, z0)) ** g1, 0)
        sum2 = c2 * np.where(below_source, (1 + max(z, z0)) ** g2 - (1 + np.maximum(zlook, z0)) ** g2, 0)

        return np.where(below_source, K * (1 + zlook) ** alpha * (sum1 + sum2), 0.)

    # --------------------------------------------------------
    # DLA continuum
    # --------------------------------------------------------

    def _calc_dla_continuum_coeffs(self, beta, logNb, logNmin, ksum, alpha, g):
        """
        Pre-compute everything in the DLA continuum expression that only
        depends on the fixed model parameters (beta, logNb, logNmin, ksum,
        alpha, g) so tau_dla_continuum() only has to evaluate the
        z/wavelength-dependent terms on each call.
        """
        c1 = float((10 ** logNb * 6.3e-18) ** (beta - 1) / gammainc(1 - beta, 10 ** (logNmin - logNb), np.inf))

        taul = 10 ** logNmin * 6.3e-18
        k = np.arange(ksum)
        c2 = taul ** (1 - beta) * (-taul) ** k / (factorial(k) * (k + 1 - beta))

        g1mb = float(gammainc(1 - beta))

        # exponents used in the three summation terms of tau_dla_continuum;
        # these depend only on (g, alpha, beta, k) and are therefore constant
        g1 = g + 1
        g2 = g + 1 - alpha * (beta - 1)
        g3 = g + 1 - alpha * k  # shape (ksum,)

        return dict(c1=c1, c2=c2, g1mb=g1mb, k=k, g1=g1, g2=g2, g3=g3)

    def tau_dla_continuum(self, z, wavelength):
        coeffs = self._dla_continuum_coeffs
        c1, c2, g1mb, k, g1, g2, g3 = (
            coeffs[key] for key in ('c1', 'c2', 'g1mb', 'k', 'g1', 'g2', 'g3')
        )
        alpha = self.LL_alpha
        beta = self.dla_gN_params['beta']
        n0 = self.dla_nz_params['n0']

        lratio = wavelength / 912.
        zs_p1 = 1 + z

        sum1 = (zs_p1 ** g1 - lratio ** g1) / g1

        sum2 = c1 * g1mb * lratio ** (alpha * (beta - 1)) * (zs_p1 ** g2 - lratio ** g2) / g2

        k = k[:, None]
        g3 = g3[:, None]
        sum3 = c1 * (c2[:, None] * lratio[None, :] ** (alpha * k) * (zs_p1 ** g3 - lratio[None, :] ** g3) / g3).sum(axis=0)

        return np.where(lratio - 1 < z, n0 * (sum1 - sum2 + sum3), 0.)