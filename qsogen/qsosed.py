#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to produce model quasar SEDs.
See accompanying README file for further details.

Need accompanying config.py file and three additional input files:
qsosed_emlines_20210625.dat    Emission line templates
S0_template_norm.sed   Host galaxy template
pl_ext_comp_03.sph    Quasar extinction curve

@author: Matthew Temple

This version first created 2019 Feb 07; last updated 2021 Mar 13.
"""
import numpy as np
from scipy.integrate import quad
from astropy.convolution import Gaussian1DKernel, convolve

_c_ = 299792458.0   # speed of light in m/s

def four_pi_dL_sq(z):
    """Compute luminosity distance for flux-luminosity conversion."""

    def integrand(z):
        return (0.27*(1+z)**3 + 0.73)**(-0.5)

    value, err = quad(integrand, 0, z)
    Log_d_L = (np.log10(1.303e+28) + np.log10(1+z) + np.log10(value))
    # this is log10(c/H0)*(1+z)*(integral) in cgs units with
    # Omega_m=0.27, Omega_lambda=0.73, Omega_k=0, H_0=71 km/s/Mpc

    return (np.log10(12.5663706144) + 2*Log_d_L)  # multiply by 4pi


def pl(wavlen, plslp, const):
    """Define power-law in flux density per unit frequency."""
    return const*wavlen**plslp


def bb(tbb, wav):
    """Blackbody shape in flux per unit frequency.

    Parameters
    ----------
    tbb
        Temperature in Kelvin.
    wav : float or ndarray of floats
        Wavelength in Angstroms.

    Returns
    -------
    Flux : float or ndarray of floats
        (Non-normalised) Blackbody flux density per unit frequency.

    Notes
    -----
    h*c/k_b = 1.43877735e8 KelvinAngstrom
    """
    return (wav**(-3))/(np.exp(1.43877735e8 / (tbb*wav)) - 1.0)

#######
# different tau models

def tau_eff_dpl(z):
    #Ly alpha optical depth from Becker et al. 2013MNRAS.430.2067B.
    # using the coefficient values from bosman et al 2022
    '''
    tau0=0.3
    beta=13.7
    C=1.35
    z0=4.8'''

    #tau_eff = tau0*((1+z)/(1+z0))**beta+C
    #tau_eff = 0.751*((1 + z) / (1 + 3.5))**2.90 - 0.132
    
    #my smoothly varying double power law
    
       
    A = 2.13369912
    z_b = 5.17124349
    a_1 = -2.32643103
    a_2 = -7.01644977
    delta = 0.05886805
    c = -0.09805639
    
    tau_eff = A*(z/z_b)**(-a_1) * (0.5*(1+(z/z_b)**(1/delta)))**((a_1-a_2)*delta)+c
	

    return np.where(tau_eff < 0, 0., tau_eff)

def tau_eff_becker2013(z):
        #Ly alpha optical depth from Becker et al. 2013MNRAS.430.2067B.
    # using the coefficient values from bosman et al 2022
    
    tau0=0.3
    beta=13.7
    C=1.35
    z0=4.8

    tau_eff = tau0*((1+z)/(1+z0))**beta+C
    tau_eff = 0.751*((1 + z) / (1 + 3.5))**2.90 - 0.132
    
    return np.where(tau_eff < 0, 0., tau_eff)

absmodels = {'dpl':tau_eff_dpl,
			 'becker+2013':tau_eff_becker2013}


class Quasar_sed:
    """Construct an instance of the quasar SED model.

    Attributes
    ----------
    flux : ndarray
        Flux per unit wavelength from total SED, i.e. quasar plus host galaxy.
    host_galaxy_flux : ndarray
        Flux p.u.w. from host galaxy component of the model SED.
    wavlen : ndarray
        Wavelength array in the rest frame.
    wavred : ndarray
        Wavelength array in the observed frame.

    Examples
    --------
    Create and plot quasar models using default params at redshifts z=2 and z=4
    >>> Quasar2 = Quasar_sed(z=2)
    >>> Quasar4 = Quasar_sed(z=4)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(Quasar2.wavred, Quasar2.flux, label='$z=2$ quasar model')
    >>> plt.plot(Quasar4.wavred, Quasar4.flux, label='$z=4$ quasar model')

    """
    def __init__(self,
                 z=2,
                 LogL3000=46,
                 wavlen=np.logspace(2.95, 4.48, num=20001, endpoint=True),
                 ebv=0.,
                 params=None,
                 **kwargs):
        """Initialises an instance of the Quasar SED model.

        Parameters
        ----------
        z : float, optional
            Redshift. If `z` is less than 0.005 then 0.005 is used instead.
        LogL3000 : float, optional
            Monochromatic luminosity at 3000A of (unreddened) quasar model,
            used to scale model flux such that synthetic magnitudes can be
            computed.
        wavlen : ndarray, optional
            Rest-frame wavelength array. Default is log-spaced array covering
            ~890 to 30000 Angstroms. `wavlen` must be monotonically increasing,
            and if gflag==True, `wavlen` must cover 4000-5000A to allow the
            host galaxy component to be properly normalised.
        ebv : float, optional
            Extinction E(B-V) applied to quasar model. Not applied to galaxy
            component. Default is zero.
        zlum_lumval : array, optional
            Redshift-luminosity relation used to control galaxy and emission-
            line contributions. `zlum_lumval[0]` is an array of redshifts, and
            `zlum_lumval[1]` is an array of the corresponding absolute i-band
            magnitudes M_i. Default is the median M_i from SDSS DR16Q in the
            apparent magnitude range 18.6<i<19.1.
        M_i :float, optional
            Absolute i-band magnitude (at z=2), as reported in SDSS DR16Q, used
            to control scaling of emission-line and host-galaxy contributions.
            Default is to use the relevant luminosity from `zlum_lumval`, which
            gives a smooth scaling with redshift `z`.
        params : dict, optional
            Dictionary of additional parameters, including emission-line and
            host-galaxy template SEDs, reddening curve. Default is to read in
            from config.py file.

        Other Parameters
        ----------------
        tbb : float, optional
            Temperature of hot dust blackbody in Kelvin.
        bbnorm : float, optional
            Normalisation, relative to power-law continuum at 2 micron, of the
            hot dust blackbody.
        scal_emline : float, optional
            Overall scaling of emission line template. Negative values preserve
            relative equivalent widths while positive values preserve relative
            line fluxes. Default is -1.
        emline_type : float, optional
            Type of emission line template. Minimum allowed value is -2,
            corresponding to weak, highly blueshifed lines. Maximum allowed is
            +3, corresponding to strong, symmetric lines. Zero correspondes to
            the average emission line template at z=2, and -1 and +1 map to the
            high blueshift and high EW extrema observed at z=2. Default is
            None, which uses `beslope` to scale `emline_type` as a smooth
            function of `M_i`.
        scal_halpha, scal_lya, scal_nlr : float, optional
            Additional scalings for the H-alpha, Ly-alpha, and for the narrow
            optical lines. Default is 1.
        beslope : float, optional
            Baldwin effect slope, which controls the relationship between
            `emline_type` and luminosity `M_i`.
        bcnorm : float, optional
            Balmer continuum normalisation. Default is zero as default emission
            line templates already include the Balmer Continuum.
        lyForest : bool, optional
            Flag to include Lyman absorption from IGM. Default is True.
        lylim : float, optional
            Wavelength of Lyman-limit system, below which all flux is
            suppressed. Default is 912A.
        gflag : bool, optional
            Flag to include host-galaxy emission. Default is True.
        fragal : float, optional
            Fractional contribution of the host galaxy to the rest-frame 4000-
            5000A region of the total SED, for a quasar with M_i = -23.
        gplind : float, optional
            Power-law index dependence of galaxy luminosity on M_i.
        emline_template : array, optional
            Emission line templates. Array must have structure
            [wavelength, average lines, reference continuum,
            high-EW lines, high-blueshift lines, narrow lines]
        reddening_curve : array, optional
            Quasar reddening law.
            Array must have structure [wavelength lambda, E(lambda-V)/E(B-V)]
        galaxy_template : array, optional
            Host-galaxy SED template.
            Array must have structure [lambda, f_lambda].
            Default is an S0 galaxy template from the SWIRE library.

        """
        if params is None:
            from qsogen.config import params
        _params = params.copy()  # avoid overwriting params dict with kwargs
        for key, value in kwargs.items():
            if key not in _params.keys():
                print('Warning: "{}" not recognised as a kwarg'.format(key))
            _params[key] = value
        self.params = _params

        self.z = max(float(z), 0.005)
        # avoid crazy flux normalisation at zero redshift

        self.wavlen = wavlen
        if np.any(self.wavlen[:-1] > self.wavlen[1:]):
            raise Exception('wavlen must be monotonic')
        self.flux = np.zeros_like(self.wavlen)
        self.host_galaxy_flux = np.zeros_like(self.wavlen)

        self.ebv = ebv
        
        self.absmod = _params['absmod']
        
        self.plslp1 = _params['plslp1']
        self.plslp2 = _params['plslp2']
        self.plstep = _params['plstep']
        self.tbb = _params['tbb']
        self.plbrk1 = _params['plbrk1']
        self.plbrk3 = _params['plbrk3']
        self.bbnorm = _params['bbnorm']
        self.scal_emline = _params['scal_emline']
        self.emline_type = _params['emline_type']
        self.scal_halpha = _params['scal_halpha']
        self.scal_lya = _params['scal_lya']
        self.scal_nlr = _params['scal_nlr']

        self.emline_template = _params['emline_template']
        self.reddening_curve = _params['reddening_curve']
        self.galaxy_template = _params['galaxy_template']

        self.beslope = _params['beslope']
        self.benorm = _params['benorm']
        self.bcnorm = _params['bcnorm']
        self.fragal = _params['fragal']
        self.gplind = _params['gplind']

        self.zlum = _params['zlum_lumval'][0]
        self.lumval = _params['zlum_lumval'][1]

        if _params['M_i'] is not None:
            self.M_i = _params['M_i']
        else:
            self.M_i = np.interp(self.z, self.zlum, self.lumval)

        #######################################################
        # READY, SET, GO!
        #######################################################
        self.wavred = (self.z + 1)*self.wavlen
        self.set_continuum()
        self.add_blackbody()
        if self.bcnorm:
            self.add_balmer_continuum()
        if LogL3000 is not None:
            self.f3000 = (10**(LogL3000 - four_pi_dL_sq(self.z))
                          / (3000*(1 + self.z)))
            self.convert_fnu_flambda(flxnrm=self.f3000, wavnrm=3000)
        else:
            self.convert_fnu_flambda()

        self.add_emission_lines()
        if _params['gflag']:
            self.host_galaxy()
        # creates self.host_galaxy_flux object
        # need to create this before reddening qso to get correct normalisation

        # redden spectrum if E(B-V) != 0
        if self.ebv:
            self.redden_spectrum()

        # add in host galaxy flux
        if _params['gflag']:
            self.flux += self.host_galaxy_flux

        # simulate the effect of a Lyman limit system at rest wavelength Lylim
        # by setting flux equal to zero at wavelengths < Lylim angstroms
        if _params['lyForest']:
            lylim = self.wav2num(_params['lylim'])
            self.flux[:lylim] = 0.0
            self.host_galaxy_flux[:lylim] = 0.0
            # Then add in Ly forest absorption at z>1.4
            self.lyman_forest()

        # redshift spectrum
        

    def wav2num(self, wav):
        """Convert a wavelength to an index."""
        return np.argmin(np.abs(self.wavlen - wav))

    def wav2flux(self, wav):
        """Convert a wavelength to a flux.

        Different from self.flux[wav2num(wav)], as wav2flux interpolates in an
        attempt to avoid problems when wavlen has gaps. This mitigation only
        works before the emission lines are added to the model, and so wav2flux
        should only be used with a reasonably dense wavelength array.
        """
        return np.interp(wav, self.wavlen, self.flux)

    def set_continuum(self, flxnrm=1.0, wavnrm=5500):
        """Set multi-powerlaw continuum in flux density per unit frequency."""
        # Flip signs of powerlaw slopes to enable calculation to be performed
        # as a function of wavelength rather than frequency
        sl1 = -self.plslp1
        sl2 = -self.plslp2
        wavbrk1 = self.plbrk1

        # Define normalisation constant to ensure continuity at wavbrk
        const2 = flxnrm/(wavnrm**sl2)
        const1 = const2*(wavbrk1**sl2)/(wavbrk1**sl1)

        # Define basic continuum using the specified normalisation fnorm at
        # wavnrm and the two slopes - sl1 (<wavbrk) sl2 (>wavbrk)
        fluxtemp = np.where(self.wavlen < wavbrk1,
                            pl(self.wavlen, sl1, const1),
                            pl(self.wavlen, sl2, const2))

        # Also add steeper power-law component for sub-Lyman-alpha wavelengths
        sl3 = sl1 - self.plstep
        wavbrk3 = self.plbrk3
        # Define normalisation constant to ensure continuity
        const3 = const1*(wavbrk3**sl1)/(wavbrk3**sl3)

        self.flux = np.where(self.wavlen < wavbrk3,
                             pl(self.wavlen, sl3, const3),
                             fluxtemp)

    def add_blackbody(self, wnorm=20000.):
        """Add basic blackbody spectrum to the flux distribution."""
        bbnorm = self.bbnorm  # blackbody normalisation at wavelength wnorm
        tbb = self.tbb

        if bbnorm > 0:

            bbval = bb(tbb, wnorm)
            cmult = bbnorm / bbval
            bb_flux = cmult*bb(tbb, self.wavlen)
            self.flux += bb_flux

    def add_balmer_continuum(self,
                             tbc=15000., taube=1., wavbe=3646.,
                             wnorm=3000., vfwhm=5000.):
        """Add Balmer continuum emission to the model.

        Prescription from Grandi 1982ApJ...255...25G.

        Parameters
        ----------
        tbc
            BC temperature in Kelvin.
        taube
            The optical depth at wavelength wavbe, the Balmer edge.
        bcnorm
            Normalisation of the BC at wavelength wnorm Angstroms.
        """
        fnorm = self.bcnorm

        flux_bc = np.zeros_like(self.flux)

        nuzero = _c_/(wavbe*1.0e-10)  # frequency of Balmer edge
        # calculate required normalisation constant at wavelength wnorm

        bbval = bb(tbc, wnorm)
        nu = _c_/(wnorm*1.0e-10)
        tau = taube * (nuzero/nu)**3    # tau is the optical depth at wnorm
        if tau < 50:
            bbval = bbval * (1.0 - np.exp(-tau))
        cmult = fnorm/bbval

        nu = _c_ / self.wavlen
        tau = taube * np.power(nuzero/nu, 3)
        scfact = np.ones(len(flux_bc), dtype=np.float64)
        scfact[tau <= 50.0] = 1.0 - np.exp(-tau[tau <= 50.0])
        bwav = tuple([self.wavlen < wavbe])
        flux_bc[bwav] = cmult * scfact[bwav] * bb(tbc, self.wavlen[bwav])

        # now broaden bc to simulate effect of bulk-velocity shifts
        vsigma = vfwhm / 2.35
        wsigma = wavbe * vsigma*1e3 / _c_  # change vsigma from km/s to m/s
        winc = (self.wavlen[self.wav2num(wnorm)]
                - self.wavlen[self.wav2num(wnorm) - 1])
        psigma = wsigma / winc     # winc is wavelength increment at wnorm
        gauss = Gaussian1DKernel(stddev=psigma)
        flux_bc = convolve(flux_bc, gauss)
        # Performs a Gaussian smooth with dispersion psigma pixels

        # Determine height of power-law continuum at wavelength wnorm to
        # allow correct scaling of Balmer continuum contribution
        self.flux += flux_bc*self.wav2flux(wnorm)

    def convert_fnu_flambda(self, flxnrm=1.0, wavnrm=5100):
        """Convert f_nu to f_lamda, using c/lambda^2 conversion.
        Normalise such that f_lambda(wavnrm) is equal to flxnrm.
        """
        self.flux = self.flux*self.wavlen**(-2)
        self.flux = self.flux*flxnrm/self.wav2flux(wavnrm)

    def add_emission_lines(self, wavnrm=5500, wmin=6000, wmax=7000):
        """Add emission lines to the model SED.

        Emission-lines are included via 4 emission-line templates, which are
        packaged with a reference continuum. One of these templates gives the
        average line emission for a M_i=-27 SDSS DR16 quasar at z~2. The narrow
        optical lines have been isolated in a separate template to allow them
        to be re-scaled if necesssary. Two templates represent the observed
        extrema of the high-ionisation UV lines, with self.emline_type
        controlling the balance between strong, peaky, systemic emission and
        weak, highly skewed emission. Default is to let this vary as a function
        of redshift using self.beslope, which represents the Baldwin effect.
        The template scaling is specified by self.scal_emline, with positive
        values producing a scaling by intensity, whereas negative values give a
        scaling that preserves the equivalent-width of the lines relative
        to the reference continuum template. The facility to scale the H-alpha
        line by a multiple of the overall emission-line scaling is included
        through the parameter scal_halpha, and the ability to rescale the
        narrow [OIII], Hbeta, etc emission is included through scal_nlr.
        """
        scalin = self.scal_emline
        scahal = self.scal_halpha
        scalya = self.scal_lya
        scanlr = self.scal_nlr
        beslp = self.beslope
        benrm = self.benorm

        if self.emline_type is None:
            if beslp:
                vallum = self.M_i
                self.emline_type = (vallum - benrm)*beslp
            else:
                self.emline_type = 0.  # default median emlines

        varlin = self.emline_type

        linwav, medval, conval, pkyval, wdyval, nlr = self.emline_template

        if varlin == 0.:
            # average emission line template for z~2 SDSS DR16Q-like things
            linval = medval + (scanlr-1.)*nlr
        elif varlin > 0:
            # high EW emission line template
            varlin = min(varlin, 3.)
            linval = varlin*pkyval + (1-varlin)*medval + (scanlr-1.)*nlr
        else:
            # highly blueshifted emission lines
            varlin = min(abs(varlin), 2.)
            linval = varlin*wdyval + (1-varlin)*medval + (scanlr-1.)*nlr

        # remove negative dips from extreme extrapolation (i.e. abs(varlin)>>1)
        linval[(linwav > 4930) & (linwav < 5030) & (linval < 0.)] = 0.
        linval[(linwav > 1150) & (linwav < 1200) & (linval < 0.)] = 0.

        linval = np.interp(self.wavlen, linwav, linval)
        conval = np.interp(self.wavlen, linwav, conval)

        imin = self.wav2num(wmin)
        imax = self.wav2num(wmax)
        _scatmp = abs(scalin)*np.ones(len(self.wavlen))
        _scatmp[imin:imax] = _scatmp[imin:imax]*abs(scahal)
        _scatmp[:self.wav2num(1350)] = _scatmp[:self.wav2num(1350)]*abs(scalya)

        # Intensity scaling
        if scalin >= 0:
            # Normalise such that continuum flux at wavnrm equal to that
            # of the reference continuum at wavnrm
            self.flux += (_scatmp * linval *
                          self.flux[self.wav2num(wavnrm)] /
                          conval[self.wav2num(wavnrm)])
            # Ensure that -ve portion of emission line spectrum hasn't
            # resulted in spectrum with -ve fluxes
            self.flux[self.flux < 0.0] = 0.0

        # EW scaling
        else:
            self.flux += _scatmp * linval * self.flux / conval
            # Ensure that -ve portion of emission line spectrum hasn't
            # resulted in spectrum with -ve fluxes
            self.flux[self.flux < 0.0] = 0.0

    def host_galaxy(self, gwnmin=4000.0, gwnmax=5000.0):
        """Correctly normalise the host galaxy contribution."""

        if min(self.wavlen) > gwnmin or max(self.wavlen) < gwnmax:
            raise Exception(
                    'wavlen must cover 4000-5000 A for galaxy normalisation'
                    + '\n Redshift is {}'.format(self.z))

        fragal = min(self.fragal, 0.99)
        fragal = max(fragal, 0.0)

        if self.galaxy_template is not None:
            wavgal, flxtmp = self.galaxy_template
        else:
            # galaxy SED input file
            f3 = 'Sb_template_norm.sed'
            wavgal, flxtmp = np.genfromtxt(f3, unpack=True)

        # Interpolate galaxy SED onto master wavlength array
        flxgal = np.interp(self.wavlen, wavgal, flxtmp)
        galcnt = np.sum(flxgal[self.wav2num(gwnmin):self.wav2num(gwnmax)])

        # Determine fraction of galaxy SED to add to unreddened quasar SED
        qsocnt = np.sum(self.flux[self.wav2num(gwnmin):self.wav2num(gwnmax)])
        # bring galaxy and quasar flux zero-points equal
        cscale = qsocnt / galcnt

        vallum = self.M_i
        galnrm = -23.   # this is value of M_i for gznorm~0.35
        # galnrm = np.interp(0.2, self.zlum, self.lumval)

        vallum = vallum - galnrm
        vallum = 10.0**(-0.4*vallum)
        tscale = vallum**(self.gplind-1)
        scagal = (fragal/(1-fragal))*tscale

        self.host_galaxy_flux = cscale * scagal * flxgal

    def redden_spectrum(self, R=3.1):
        """Redden quasar component of total SED. R=A_V/E(B-V)."""

        if self.reddening_curve is not None:
            wavtmp, flxtmp = self.reddening_curve
        else:
            # read extinction law from file
            f4 = 'pl_ext_comp_03.sph'
            wavtmp, flxtmp = np.genfromtxt(f4, unpack=True)

        extref = np.interp(self.wavlen, wavtmp, flxtmp)
        exttmp = self.ebv * (extref + R)
        self.flux = self.flux*10.0**(-exttmp/2.5)

    def lyman_forest(self):
        
        if self.absmod == 'dpl' or self.absmod == 'becker+2013':
            tau_eff = absmodels[absmod]
            lyseries ={0: [1215.67,1],
	                1: [1025.72,0.19005811214447021],
	                2: [972.537,0.06965703475200001]}
            for i in lyseries.keys():
	            scale = np.zeros_like(self.flux)
	            wlim = lyseries[i][0]
	            ratio = lyseries[i][1]
	            zlook = ((1.0+self.z) * self.wavlen)/wlim - 1.0
	            scale[self.wavlen < wlim] = tau_eff(zlook[self.wavlen < wlim])
	            scale = np.exp(-ratio*scale)
	            self.flux = scale * self.flux
	            self.host_galaxy_flux = scale * self.host_galaxy_flux
            
        elif self.absmod=='inoue+2014':
            n = 38
            lines = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150, 920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919, 914.576, 914.286, 914.039, 913.826, 913.641, 913.480, 913.339, 913.215, 913.104, 913.006, 912.918, 912.839, 912.768, 912.703, 912.645, 912.592, 912.543, 912.499, 912.458, 912.420, 912.385, 912.353, 912.324])[:n]
            A1 = np.array([1.690e-2,4.692e-3,2.239e-3,1.319e-3,8.707e-4,6.178e-4,4.609e-4,3.569e-4,2.843e-4,2.318e-4,1.923e-4,1.622e-4,1.385e-4,1.196e-4,1.043e-4,9.174e-5,8.128e-5,7.251e-5,6.505e-5,5.868e-5,5.319e-5,4.843e-5,4.427e-5,4.063e-5,3.738e-5,3.454e-5,3.199e-5,2.971e-5,2.766e-5,2.582e-5,2.415e-5,2.263e-5,2.126e-5,2.000e-5,1.885e-5,1.779e-5,1.682e-5,1.593e-5,1.510e-5])[:n,None]
            A2 = np.array([2.354e-3,6.536e-4,3.119e-4,1.837e-4,1.213e-4,8.606e-5,6.421e-5,4.971e-5,3.960e-5,3.229e-5,2.679e-5,2.259e-5,1.929e-5,1.666e-5,1.453e-5,1.278e-5,1.132e-5,1.010e-5,9.062e-6,8.174e-6,7.409e-6,6.746e-6,6.167e-6,5.660e-6,5.207e-6,4.811e-6,4.456e-6,4.139e-6,3.853e-6,3.596e-6,3.364e-6,3.153e-6,2.961e-6,2.785e-6,2.625e-6,2.479e-6,2.343e-6,2.219e-6,2.103e-6])[:n,None]
            A3 = np.array([1.026e-4,2.849e-5,1.360e-5,8.010e-6,5.287e-6,3.752e-6,2.799e-6,2.167e-6,1.726e-6,1.407e-6,1.168e-6,9.847e-7,8.410e-7,7.263e-7,6.334e-7,5.571e-7,4.936e-7,4.403e-7,3.950e-7,3.563e-7,3.230e-7,2.941e-7,2.689e-7,2.467e-7,2.270e-7,2.097e-7,1.943e-7,1.804e-7,1.680e-7,1.568e-7,1.466e-7,1.375e-7,1.291e-7,1.214e-7,1.145e-7,1.080e-7,1.022e-7,9.673e-8,9.169e-8])[:n,None]
            
            tau = np.outer(1/lines,self.wavred)
            #print(tau)

            c0 = (tau <1.) | (tau>(1.+self.z))
            c1 = (tau<2.2) 
            c2 = (tau<=5.7) & (tau>=2.2)
            c3 = (tau>5.7) 

            #tau[c1]=np.multiply(tau[c1]**1.2,A1[:,None])
            #tau[c2]=np.multiply(tau[c2]**3.7,A2[:,None])
            #tau[c3]=np.multiply(tau[c3]**5.5,A3[:,None])


            #np.select([c1, c2, c3], [A1 * (wv_array / np.array(lines))**1.2, A2 * (wv_array / np.array(lines))**3.7, A3 * (wv_array / np.array(lines))**5.5])
            #plt.plot(wvarray,np.sum(tau,axis=0))


            #np.putmask(tau,tau<1.,0)
            np.putmask(tau,c1,np.multiply(tau**1.2,A1))
            np.putmask(tau,c2 ,np.multiply(tau**3.7,A2))
            np.putmask(tau,c3 ,np.multiply(tau**5.5,A3))
            np.putmask(tau,c0,0.)

            scale = np.sum(tau,axis=0)
            
            self.flux = np.exp(-scale)*self.flux
            self.host_galaxy_flux = scale*self.host_galaxy_flux
          
        elif self.absmod=='madau+1995':
            mlines = np.array([1215.67, 1025.72, 972.537, 949.743])
            mA = np.array([0.0036,1.7e-3,1.2e-3,9.3e-4])[:,None]
            ratio = np.outer(1/mlines,self.wavred)
            tau = np.zeros_like(ratio)

            for i in range(len(ratio)):
                r = ratio[i]
                c0 =  (self.wavred<=mlines[i]*(1+self.z))
                tau[i][c0]=(r**3.46*mA[i])[c0]
           
            su = np.sum(tau,axis=0)
            su[self.wavred<912*(1+self.z)]=np.inf

            scale = np.exp(-su)
            
            self.flux = scale*(self.flux)
            self.host_galaxy_flux = scale*self.host_galaxy_flux
              

	     
		

if __name__ == '__main__':

    print(help(Quasar_sed))
