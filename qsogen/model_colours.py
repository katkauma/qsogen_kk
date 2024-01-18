#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020 November 10
Public release 2021 March 13
Edit 2021 November 17 to add DECam and HSC filters
Edit 2022 June 08 to also allow filters to be stored in ./filters/
Edit 2022 July 29 to add GALEX NUV and FUV filters

@author: Matthew Temple

# This research has made use of the SVO Filter Profile Service
#  (http://svo2.cab.inta-csic.es/theory/fps/) supported from
#  the Spanish MINECO through grant AyA2014-55216

Any filter of interest should be saved in the filters/ folder eg., filters/SDSS_u.filter

Default filters are SDSS AB, UKIDSS Vega, WISE Vega ugrizYJHKW12

Conventional to use AB magnitudes for SDSS, VISTA (KiDS), LSST, Euclid
 and Vega magnitudes for UKIDSS and WISE.
Note UKIDSS and non-KiDS VISTA calibrated to 2MASS which assumes Vega has
 zero mag; WISE also assumes Vega has zero mag.

We use the 2007 Vega spectrum to produce Vega zeropoints.

The code is structured to do the following things once, minimising i/o:
    Load filter response functions and associated wav arrays
    Load filter normalisations
    Concatenate and sort wav arrays
    Identify where sorted array corresponds to each filter wav array
And then run the SED model many times, using filter wav arrays as input wavlen:
    looping over z (or any other variable of interest)
    multiply relevant part of each flux array with filter response function
    convert to magnitude, return colours or mags as appropriate

"""

import numpy as np
from scipy.integrate import simps
from qsogen.qsosed import Quasar_sed
import os
import json

install_path = os.path.dirname(os.path.abspath(__file__))


# load the zeropoints and pivot wavelengths from filterinfo.json
with open(install_path+'/filterinfo.json','r') as file:
    filterinfo = json.load(file)

Vega_zeropoints=filterinfo['Vega_zeropoints']
AB_zeropoints = filterinfo['AB_zeropoints']
pivotwv = filterinfo['Pivot_wv']

zeropoints = {**Vega_zeropoints, **AB_zeropoints}



###########################################################

wavarrs, resparrs = dict(), dict()
for band in pivotwv.keys():
    try:
        wavarr, response = np.genfromtxt(install_path+'/filters/'+band+'.filter', unpack=True)
    except OSError:
        wavarr, response = np.genfromtxt(install_path+'/filters/'+band+'.filter', unpack=True)
    wavarrs[band] = wavarr
    resparrs[band] = response


# default zeropoints use AB for SDSS and Vega for UKIDSS and WISE
def get_colours(redshifts,
                filters=['SDSS_u_AB',
                         'SDSS_g_AB',
                         'SDSS_r_AB',
                         'SDSS_i_AB',
                         'SDSS_z_AB',
                         'UKIDSS_Y_Vega',
                         'UKIDSS_J_Vega',
                         'UKIDSS_H_Vega',
                         'UKIDSS_K_Vega',
                         'WISE_W1_Vega',
                         'WISE_W2_Vega'],
                **kwargs):
    """Get synthetic colours from quasar model.

    Parameters
    ----------
    redshifts : iterable
        List or array of redshifts
    filters : array, optional
        List of filter passbands to compute colours between.
    **kwargs
        Arguments to pass to Quasar_sed.

    Returns
    -------
    model_colours : ndarray
        Array with model colours for each redshift in redshifts.

    Notes
    -----
    Wavelength array in Quasar_sed must cover rest-frame 4000-5000 Angstroms,
    if gflag is set to True.
    get_colours concatenates and sorts the wavelength arrays of the filter
    response functions, and uses this as the input wavelength array to
    Quasar_sed. This is computationally much faster as it means the model is
    only evaluated at the wavelengths it's actually needed and avoids
    unnecessary interpolation. However, this can lead to errors if the host
    galaxy is turned on and an unusually sparse combination of filters is
    requested.
    """

    waves, responses = [], []
    for band in filters:
        band = band.replace('_AB', '')
        band = band.replace('_Vega', '')
        waves.append(wavarrs[band])
        responses.append(resparrs[band])

    obs_wavlen = np.concatenate(waves)
    isort = obs_wavlen.argsort().argsort()
    # indices to invert the sorting on the concatenated array
    split_indices = np.cumsum([len(wav) for wav in waves[:-1]])
    # indices to invert the concatenation

    model_colours = []

    try:
        for z in redshifts:
            rest_ordered_wav = np.sort(obs_wavlen/(1+z))
            # now in rest frame of qso
            ordered_flux = Quasar_sed(wavlen=rest_ordered_wav,
                                      z=z,
                                      **kwargs).flux
            # qsosed will produce redshifted flux

            # Create individual arrays with the flux in each observed passband
            fluxes = np.split(ordered_flux[isort], split_indices)

            model_colours.append(-np.diff(sed2mags(filters,
                                                   waves,
                                                   fluxes,
                                                   responses)))
    except TypeError:
        z = float(redshifts)
        rest_ordered_wav = np.sort(obs_wavlen/(1+z))
        # now in rest frame of qso
        ordered_flux = Quasar_sed(wavlen=rest_ordered_wav, z=z, **kwargs).flux
        # qsosed will produce redshifted flux

        # Create individual arrays with the flux in each observed passband
        fluxes = np.split(ordered_flux[isort], split_indices)

        model_colours.append(-np.diff(sed2mags(filters,
                                               waves,
                                               fluxes,
                                               responses)))

    return(np.array(model_colours))


def get_mags(redshifts,
             filters=['SDSS_u_AB',
                      'SDSS_g_AB',
                      'SDSS_r_AB',
                      'SDSS_i_AB',
                      'SDSS_z_AB',
                      'UKIDSS_Y_Vega',
                      'UKIDSS_J_Vega',
                      'UKIDSS_H_Vega',
                      'UKIDSS_K_Vega',
                      'WISE_W1_Vega',
                      'WISE_W2_Vega'],
             **kwargs):
    """Get synthetic magnitudes from quasar model.

    Parameters
    ----------
    redshifts : iterable
        List or array of redshifts
    filters : array, optional
        List of filter passbands to compute colours between.
    **kwargs
        Arguments to pass to Quasar_sed.

    Returns
    -------
    model_mags : ndarray
        Array with model magnitudes for each redshift in redshifts.

    Notes
    -----
    Wavelength array in Quasar_sed must cover rest-frame 4000-5000 Angstroms,
    if gflag is set to True.
    get_mags concatenates and sorts the wavelength arrays of the filter
    response functions, and uses this as the input wavelength array to
    Quasar_sed. This is computationally much faster as it means the model is
    only evaluated at the wavelengths it's actually needed and avoids
    unnecessary interpolation. However, this can lead to errors if the host
    galaxy is turned on and an unusually sparse combination of filters is
    requested.
    """

    waves, responses = [], []
    for band in filters:
        band = band.replace('_AB', '')
        band = band.replace('_Vega', '')
        waves.append(wavarrs[band])
        responses.append(resparrs[band])

    obs_wavlen = np.concatenate(waves)
    isort = obs_wavlen.argsort().argsort()
    # indices to invert the sorting on the concatenated array
    split_indices = np.cumsum([len(wav) for wav in waves[:-1]])
    # indices to invert the concatenation

    model_mags = []

    try:
        for z in redshifts:
            rest_ordered_wav = np.sort(obs_wavlen/(1+z))
            # now in rest frame of qso
            # note wavlength array must cover 4000-5000 Angstroms
            ordered_flux = Quasar_sed(wavlen=rest_ordered_wav,
                                      z=z,
                                      **kwargs).flux
            # qsosed will produce redshifted flux

            # Create individual arrays with the flux in each observed passband
            fluxes = np.split(ordered_flux[isort], split_indices)
            
            model_mags.append(sed2mags(filters, waves, fluxes, responses))

    except TypeError:
        z = float(redshifts)
        rest_ordered_wav = np.sort(obs_wavlen/(1+z))
        # now in rest frame of qso
        # note wavlength array must cover 4000-5000 Angstroms
        ordered_flux = Quasar_sed(wavlen=rest_ordered_wav, z=z, **kwargs).flux
        # qsosed will produce redshifted flux

        # Create individual arrays with the flux in each observed passband
        fluxes = np.split(ordered_flux[isort], split_indices)

        model_mags.append(sed2mags(filters, waves, fluxes, responses))
        
    #added by katherine, converts list to numpy array
    model_mags = np.array(model_mags)

    return(model_mags)


def sed2mags(filters, waves, fluxes, responses):

        mags = np.full(len(waves), np.nan)

        for i in range(len(waves)):
            flux = simps(waves[i]*responses[i]*fluxes[i], waves[i])
            mags[i] = -2.5*np.log10(flux/zeropoints[filters[i]])

        return(mags)
    
    
def get_fluxes(redshifts,
             filters=['SDSS_u_AB',
                      'SDSS_g_AB',
                      'SDSS_r_AB',
                      'SDSS_i_AB',
                      'SDSS_z_AB',
                      'UKIDSS_Y_Vega',
                      'UKIDSS_J_Vega',
                      'UKIDSS_H_Vega',
                      'UKIDSS_K_Vega',
                      'WISE_W1_Vega',
                      'WISE_W2_Vega'],
             units='fnu',
             **kwargs):
    """Get synthetic magnitudes from quasar model.

    Parameters
    ----------
    redshifts : iterable
        List or array of redshifts
    filters : array, optional
        List of filter passbands to compute colours between.
    units : string, optional
        Controls whether fluxes are returned in units of "flam" (ergs s^-1 cm^-1 A^-1) or "fnu" (Jy). Default is "flam".
    **kwargs
        Arguments to pass to Quasar_sed.

    Returns
    -------
    model_mags : ndarray
        Array with model magnitudes for each redshift in redshifts.

    Notes
    -----
    Wavelength array in Quasar_sed must cover rest-frame 4000-5000 Angstroms,
    if gflag is set to True.
    get_mags concatenates and sorts the wavelength arrays of the filter
    response functions, and uses this as the input wavelength array to
    Quasar_sed. This is computationally much faster as it means the model is
    only evaluated at the wavelengths it's actually needed and avoids
    unnecessary interpolation. However, this can lead to errors if the host
    galaxy is turned on and an unusually sparse combination of filters is
    requested.
    """
    
    mags = get_mags(redshifts,
             filters=filters,
             **kwargs)
    
    fnu = 3631*10**(-0.4*mags)
    
    if units=='fnu':
        return fnu
    
    elif units=='flam':
        pivlam2=np.array([pivotwv2[band.replace('_AB','').replace('_Vega','')] for band in filters])
        
        flam = 2.9982e-5*fnu/(pivlam*pivlam)
        return flam
            
    else:
        raise ValueError("'units' keyword must be 'flam' or 'fnu")

 