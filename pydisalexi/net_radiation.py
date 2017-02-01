# This file is part of pyTSEB for calculating the net radiation and its divergence
# Copyright 2016 Hector Nieto and contributors listed in the README.md file.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Created on Apr 6 2015
@author: Hector Nieto (hnieto@ias.csic.es).

Modified on Jan 27 2016
@author: Hector Nieto (hnieto@ias.csic.es).

DESCRIPTION
===========
This package contains functions for estimating the net shortwave and longwave radiation
for soil and canopy layers. Additional packages needed are.

* :doc:`meteoUtils` for the estimation of meteorological variables.
* `foursail` for the estimation net radiation using the 4SAIL Radiative Transfer Model.

PACKAGE CONTENTS
================
* :func:`calc_difuse_ratio` estimation of fraction of difuse shortwave radiation.
* :func:`calc_emiss_atm` Atmospheric emissivity.
* :func:`calc_K_be_Campbell` Beam extinction coefficient.
* :func:`calc_L_n_Kustas` Net longwave radiation for soil and canopy layers.
* :func:`calc_Rn_OSEB` Net radiation in a One Source Energy Balance model.
* :func:`calc_Rn_4SAIL` Soil and canopy net radiation using 4SAIL.
* :func:`calc_Sn_Campbell` Net shortwave radiation.
* :func:`calc_tau_below_Campbell` Radiation transmission through a canopy.
* :func:`calc_Sn_below_4SAIL Soil` net shortwave radiation using 4SAIL.
'''

#from FourSAIL import FourSAIL
import numpy as np

from .meteo_utils import calc_stephan_boltzmann

#==============================================================================
# List of constants used in the netRadiation Module
#==============================================================================
# Stephan Boltzmann constant (W m-2 K-4)
sb = 5.670373e-8


def calc_difuse_ratio(S_dn, sza, press=1013.25, SOLAR_CONSTANT=1320):
    '''Fraction of difuse shortwave radiation.

    Partitions the incoming solar radiation into PAR and non-PR and
    diffuse and direct beam component of the solar spectrum.

    Parameters
    ----------
    S_dn : float
        Incoming shortwave radiation (W m-2).
    sza : float
        Solar Zenith Angle (degrees).
    Wv : float, optional
        Total column precipitable water vapour (g cm-2), default 1 g cm-2.
    press : float, optional
        atmospheric pressure (mb), default at sea level (1013mb).

    Returns
    -------
    difvis : float
        diffuse fraction in the visible region.
    difnir : float
        diffuse fraction in the NIR region.
    fvis : float
        fration of total visible radiation.
    fnir : float
        fraction of total NIR radiation.

    References
    ----------
    .. [Weiss1985] Weiss and Norman (1985) Partitioning solar radiation into direct and diffuse,
        visible and near-infrared components, Agricultural and Forest Meteorology,
        Volume 34, Issue 2, Pages 205-213,
        http://dx.doi.org/10.1016/0168-1923(85)90020-6.
    '''

    # Convert input scalars to numpy arrays
    S_dn, sza, press = map(np.asarray, (S_dn, sza, press))
    difvis, difnir, fvis, fnir = [np.zeros(S_dn.shape) for i in range(4)]
    fvis = fvis + 0.6
    fnir = fnir + 0.4
    # Calculate potential (clear-sky) visible and NIR solar components
    # Weiss & Norman 1985
    Rdirvis, Rdifvis, Rdirnir, Rdifnir = calc_potential_irradiance_weiss(
        sza, press=press, SOLAR_CONSTANT=SOLAR_CONSTANT)
    # Potential total solar radiation
    potvis = np.asarray(Rdirvis + Rdifvis)
    potvis[potvis <= 0] = 1e-6
    potnir = np.asarray(Rdirnir + Rdifnir)
    potnir[potnir <= 0] = 1e-6
    fclear = S_dn / (potvis + potnir)
    fclear = np.minimum(1.0, fclear)
    # Partition S_dn into VIS and NIR
    fvis = potvis / (potvis + potnir)  # Eq. 7
    fnir = potnir / (potvis + potnir)  # Eq. 8
    fvis = np.maximum(0, fvis)
    fvis = np.minimum(1, fvis)
    fnir = 1.0 - fvis
    # Estimate direct beam and diffuse fractions in VIS and NIR wavebands
    ratiox = np.asarray(fclear)
    ratiox[fclear > 0.9] = 0.9
    dirvis = (Rdirvis / potvis) * (1. - ((.9 - ratiox) / .7)**.6667)  # Eq. 11
    ratiox = np.asarray(fclear)
    ratiox[fclear > 0.88] = 0.88
    dirnir = (Rdirnir / potnir) * \
        (1. - ((.88 - ratiox) / .68)**.6667)  # Eq. 12
    dirvis = np.maximum(0.0, dirvis)
    dirnir = np.maximum(0.0, dirnir)
    dirvis = np.minimum(1, dirvis)
    dirnir = np.minimum(1, dirnir)
    difvis = 1.0 - dirvis
    difnir = 1.0 - dirnir
    return np.asarray(difvis), np.asarray(
        difnir), np.asarray(fvis), np.asarray(fnir)


def calc_emiss_atm(ea, T_A_K):
    '''Atmospheric emissivity

    Estimates the effective atmospheric emissivity for clear sky.

    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    T_A_K : float
        air temperature (Kelvin).

    Returns
    -------
    emiss_air : float
        effective atmospheric emissivity.

    References
    ----------
    .. [Brutsaert1975] Brutsaert, W. (1975) On a derivable formula for long-wave radiation
        from clear skies, Water Resour. Res., 11(5), 742-744,
        htpp://dx.doi.org/10.1029/WR011i005p00742.'''

    emiss_air = 1.24 * (ea / T_A_K)**(1. / 7.)  # Eq. 11 in [Brutsaert1975]_
    return np.asarray(emiss_air)


def calc_K_be_Campbell(theta, x_LAD=1):
    ''' Beam extinction coefficient

    Calculates the beam extinction coefficient based on [Campbell1998]_ ellipsoidal
    leaf inclination distribution function.

    Parameters
    ----------
    theta : float
        incidence zenith angle (degrees).
    x_LAD : float, optional
        Chi parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_LAD=1 for a spherical LAD.

    Returns
    -------
    K_be : float
        beam extinction coefficient.
    x_LAD: float, optional
        x parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_LAD=1 for a spherical LAD.

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    '''

    theta = np.radians(theta)
    K_be = np.sqrt(x_LAD**2 + np.tan(theta)**2) / \
        (x_LAD + 1.774 * (x_LAD + 1.182)**-0.733)
    return np.asarray(K_be)


def calc_L_n_Kustas(T_C, T_S, L_dn, LAI, emisVeg, emisGrd, x_LAD=1):
    ''' Net longwave radiation for soil and canopy layers

    Estimates the net longwave radiation for soil and canopy layers unisg based on equation 2a
    from [Kustas1999]_ and incorporated the effect of the Leaf Angle Distribution based on [Campbell1998]_

    Parameters
    ----------
    T_C : float
        Canopy temperature (K).
    T_S : float
        Soil temperature (K).
    L_dn : float
        Downwelling atmospheric longwave radiation (w m-2).
    LAI : float
        Effective Leaf (Plant) Area Index.
    emisVeg : float
        Broadband emissivity of vegetation cover.
    emisGrd : float
        Broadband emissivity of soil.
    x_LAD: float, optional
        x parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_LAD=1 for a spherical LAD.

    Returns
    -------
    L_nC : float
        Net longwave radiation of canopy (W m-2).
    L_nS : float
        Net longwave radiation of soil (W m-2).

    References
    ----------
    .. [Kustas1999] Kustas and Norman (1999) Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''

    # Integrate to get the diffuse transmitance
    taud = 0
    for angle in range(0, 90, 5):
        akd = calc_K_be_Campbell(angle, x_LAD)  # Eq. 15.4
        taub = np.exp(-akd * LAI)
        taud = taud + taub * np.cos(np.radians(angle)) * \
            np.sin(np.radians(angle)) * np.radians(5)
    taud = 2.0 * taud
    # D I F F U S E   C O M P O N E N T S
    # Diffuse light canopy reflection coefficients  for a deep canopy
    akd = -np.log(taud) / LAI
    ameanl = np.asarray(emisVeg)
    taudl = np.exp(-np.sqrt(ameanl) * akd * LAI)  # Eq 15.6
    # calculate long wave emissions from canopy, soil and sky
    L_C = emisVeg * calc_stephan_boltzmann(T_C)
    L_S = emisGrd * calc_stephan_boltzmann(T_S)
    # calculate net longwave radiation divergence of the soil
    L_nS = taudl * L_dn + (1.0 - taudl) * L_C - L_S
    L_nC = (1.0 - taudl) * (L_dn + L_S - 2.0 * L_C)
    return np.asarray(L_nC), np.asarray(L_nS)


def calc_potential_irradiance_weiss(
        sza,
        press=1013.25,
        SOLAR_CONSTANT=1320,
        fnir_ini=0.5455):
    ''' Estimates the potential visible and NIR irradiance at the surface

    Parameters
    ----------
    sza : float
        Solar Zenith Angle (degrees)
    press : Optional[float]
        atmospheric pressure (mb)

    Returns
    -------
    Rdirvis : float
        Potential direct visible irradiance at the surface (W m-2)
    Rdifvis : float
        Potential diffuse visible irradiance at the surface (W m-2)
    Rdirnir : float
        Potential direct NIR irradiance at the surface (W m-2)
    Rdifnir : float
        Potential diffuse NIR irradiance at the surface (W m-2)

    based on Weiss & Normat 1985, following same strategy in Cupid's RADIN4 subroutine.
    '''

    # Convert input scalars to numpy arrays
    sza, press = map(np.asarray, (sza, press))

    # Set defaout ouput values
    Rdirvis, Rdifvis, Rdirnir, Rdifnir, w = [
        np.zeros(sza.shape) for i in range(5)]

    coszen = np.cos(np.radians(sza))
    # Calculate potential (clear-sky) visible and NIR solar components
    # Weiss & Norman 1985
    # Correct for curvature of atmos in airmas (Kasten and Young,1989)
    i = sza < 90
    airmas = 1.0 / coszen
    # Visible PAR/NIR direct beam radiation
    Sco_vis = SOLAR_CONSTANT * (1.0 - fnir_ini)
    Sco_nir = SOLAR_CONSTANT * fnir_ini
    # Directional trasnmissivity
    # Calculate water vapour absorbance (Wang et al 1976)
    # A=10**(-1.195+.4459*np.log10(1)-.0345*np.log10(1)**2)
    # opticalDepth=np.log(10.)*A
    # T=np.exp(-opticalDepth/coszen)
    # Asssume that most absortion of WV is at the NIR
    Rdirvis[i] = (Sco_vis * np.exp(-.185 * (press[i] / 1313.25) * airmas[i]) -
                  w[i]) * coszen[i]  # Modified Eq1 assuming water vapor absorption
    # Rdirvis=(Sco_vis*exp(-.185*(press/1313.25)*airmas))*coszen
    # #Eq. 1
    Rdirvis = np.maximum(0, Rdirvis)
    # Potential diffuse radiation
    # Eq 3                                      #Eq. 3
    Rdifvis[i] = 0.4 * (Sco_vis * coszen[i] - Rdirvis[i])
    Rdifvis = np.maximum(0, Rdifvis)

    # Same for NIR
    # w=SOLAR_CONSTANT*(1.0-T)
    w = SOLAR_CONSTANT * \
        10**(-1.195 + .4459 * np.log10(coszen[i]) - .0345 * np.log10(coszen[i])**2)  # Eq. .6
    Rdirnir[i] = (Sco_nir * np.exp(-0.06 * (press[i] / 1313.25)
                                   * airmas[i]) - w) * coszen[i]  # Eq. 4
    Rdirnir = np.maximum(0, Rdirnir)
    # Potential diffuse radiation
    Rdifnir[i] = 0.6 * (Sco_nir * coszen[i] - Rdirvis[i] - w)  # Eq. 5
    Rdifnir = np.maximum(0, Rdifnir)
    Rdirvis, Rdifvis, Rdirnir, Rdifnir = map(
        np.asarray, (Rdirvis, Rdifvis, Rdirnir, Rdifnir))
    return Rdirvis, Rdifvis, Rdirnir, Rdifnir


def calc_Rn_OSEB(S_dn, L_dn, T_R, emis, albedo):
    ''' Net radiation in a One Source Energy Balance model

    Estimates surface net radiation assuming a single `big leaf` layer.

    Parameters
    ----------
    S_dn : float
        incoming shortwave radiation (W m-2).
    L_dn : float
        Incoming longwave radiation (W m-2).
    T_R : float
        Radiometric surface temperature (K).
    emis : float
        Broadband emissivity.
    albedoVeg : float
        Broadband short wave albedo.

    Returns
    -------
    R_s : float
        Net shortwave radiation (W m-2).
    R_l : float
        Net longwave radiation (W m-2).
    '''

    # outgoing shortwave radiation
    R_sout = S_dn * albedo
    # outgoing long wave radiation
    R_lout = emis * sb * (T_R)**4 + L_dn * (1.0 - emis)
    R_s = S_dn - R_sout
    R_l = L_dn - R_lout
    return np.asarray(R_s), np.asarray(R_l)



def calc_Sn_Campbell(LAI, sza, S_dn_dir, S_dn_dif, fvis, fnir, rho_leaf_vis,
                   tau_leaf_vis, rho_leaf_nir, tau_leaf_nir, rsoilv, rsoiln,
                   x_LAD=1, LAI_eff=None):
    ''' Net shortwave radiation

    Estimate net shorwave radiation for soil and canopy below a canopy using the [Campbell1998]_
    Radiative Transfer Model, and implemented in [Kustas1999]_

    Parameters
    ----------
    LAI : float
        Effective Leaf (Plant) Area Index.
    sza : float
        Sun Zenith Angle (degrees).
    S_dn_dir : float
        Broadband incoming beam shortwave radiation (W m-2).
    S_dn_dif : float
        Broadband incoming diffuse shortwave radiation (W m-2).
    fvis : float
        fration of total visible radiation.
    fnir : float
        fraction of total NIR radiation.
    rho_leaf_vis : float
        Broadband leaf bihemispherical reflectance in the visible region (400-700nm).
    tau_leaf_vis : float
        Broadband leaf bihemispherical transmittance in the visible region (400-700nm).
    rho_leaf_nir : float
        Broadband leaf bihemispherical reflectance in the NIR region (700-2500nm).
    tau_leaf_nir : float
        Broadband leaf bihemispherical transmittance in the NIR region (700-2500nm).
    rsoilv : float
        Broadband soil bihemispherical reflectance in the visible region (400-700nm).
    rsoiln : float
        Broadband soil bihemispherical reflectance in the NIR region (700-2500nm).
    x_LAD : float,  optional
        x parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    LAI_eff : float or None, optional
        if set, its value is the directional effective LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies.

    Returns
    -------
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    .. [Kustas1999] Kustas and Norman (1999) Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''

    # calculate aborprtivity
    ameanv = 1.0 - rho_leaf_vis - tau_leaf_vis
    ameann = 1.0 - rho_leaf_nir - tau_leaf_nir
    # Calculate canopy beam extinction coefficient
    # Modification to include other LADs
    if isinstance(LAI_eff, type(None)):
        LAI_eff = np.asarray(LAI)
    else:
        LAI_eff = np.asarray(LAI_eff)
    # Integrate to get the diffuse transmitance
    taud = 0
    for angle in range(0, 90, 5):
        akd = calc_K_be_Campbell(angle, x_LAD)  # Eq. 15.4
        taub = np.exp(-akd * LAI)
        taud = taud + taub * np.cos(np.radians(angle)) * \
            np.sin(np.radians(angle)) * np.radians(5)
    taud = 2.0 * taud
    # D I F F U S E   C O M P O N E N T S
    # Diffuse light canopy reflection coefficients  for a deep canopy
    # akd=-0.0683*log(LAI)+0.804                  # Fit to Fig 15.4 for x=1
    akd = -np.log(taud) / LAI
    rcpyn = (1.0 - np.sqrt(ameann)) / (1.0 + np.sqrt(ameann))  # Eq 15.7
    rcpyv = (1.0 - np.sqrt(ameanv)) / (1.0 + np.sqrt(ameanv))
    rdcpyn = 2.0 * akd * rcpyn / (akd + 1.0)  # Eq 15.8
    rdcpyv = 2.0 * akd * rcpyv / (akd + 1.0)
    # Diffuse canopy transmission coeff (visible)
    expfac = np.sqrt(ameanv) * akd * LAI
    xnum = (rdcpyv * rdcpyv - 1.0) * np.exp(-expfac)
    xden = (rdcpyv * rsoilv - 1.0) + rdcpyv * \
        (rdcpyv - rsoilv) * np.exp(-2.0 * expfac)
    taudv = xnum / xden  # Eq 15.11
    # Diffuse canopy transmission coeff (NIR)
    expfac = np.sqrt(ameann) * akd * LAI
    xnum = (rdcpyn * rdcpyn - 1.0) * np.exp(-expfac)
    xden = (rdcpyn * rsoiln - 1.0) + rdcpyn * \
        (rdcpyn - rsoiln) * np.exp(-2.0 * expfac)
    taudn = xnum / xden  # Eq 15.11
    # Diffuse radiation surface albedo for a generic canopy
    fact = ((rdcpyn - rsoiln) / (rdcpyn * rsoiln - 1.0)) * \
        np.exp(-2.0 * np.sqrt(ameann) * akd * LAI)  # Eq 15.9
    albdn = (rdcpyn + fact) / (1.0 + rdcpyn * fact)
    fact = ((rdcpyv - rsoilv) / (rdcpyv * rsoilv - 1.0)) * \
        np.exp(-2.0 * np.sqrt(ameanv) * akd * LAI)  # Eq 15.9
    albdv = (rdcpyv + fact) / (1.0 + rdcpyv * fact)
    # B E A M   C O M P O N E N T S
    # Direct beam extinction coeff (spher. LAD)
    akb = calc_K_be_Campbell(sza, x_LAD)  # Eq. 15.4
    # Direct beam canopy reflection coefficients for a deep canopy
    rcpyn = (1.0 - np.sqrt(ameann)) / (1.0 + np.sqrt(ameann))  # Eq 15.7
    rcpyv = (1.0 - np.sqrt(ameanv)) / (1.0 + np.sqrt(ameanv))
    rbcpyn = 2.0 * akb * rcpyn / (akb + 1.0)  # Eq 15.8
    rbcpyv = 2.0 * akb * rcpyv / (akb + 1.0)
    fact = ((rbcpyn - rsoiln) / (rbcpyn * rsoiln - 1.0)) * \
        np.exp(-2.0 * np.sqrt(ameann) * akb * LAI_eff)  # Eq 15.9
    albbn = (rbcpyn + fact) / (1.0 + rbcpyn * fact)
    fact = ((rbcpyv - rsoilv) / (rbcpyv * rsoilv - 1.0)) * \
        np.exp(-2.0 * np.sqrt(ameanv) * akb * LAI_eff)  # Eq 15.9
    albbv = (rbcpyv + fact) / (1.0 + rbcpyv * fact)
    # Weighted average albedo
    albedo_dir = fvis * albbv + fnir * albbn
    albedo_dif = fvis * albdv + fnir * albdn
    # Direct beam+scattered canopy transmission coeff (visible)
    expfac = np.sqrt(ameanv) * akb * LAI_eff
    xnum = (rbcpyv * rbcpyv - 1.0) * np.exp(-expfac)
    xden = (rbcpyv * rsoilv - 1.0) + rbcpyv * \
        (rbcpyv - rsoilv) * np.exp(-2.0 * expfac)
    taubtv = xnum / xden  # Eq 15.11
    # Direct beam+scattered canopy transmission coeff (NIR)
    expfac = np.sqrt(ameann) * akb * LAI_eff
    xnum = (rbcpyn * rbcpyn - 1.0) * np.exp(-expfac)
    xden = (rbcpyn * rsoiln - 1.0) + rbcpyn * \
        (rbcpyn - rsoiln) * np.exp(-2.0 * expfac)
    taubtn = xnum / xden  # Eq 15.11
    tau_dir = fvis * taubtv + fnir * taubtn
    tau_dif = fvis * taudv + fnir * taudn
    albedosoil = fvis * rsoilv + fnir * rsoiln
    Sn_C = (1.0 - tau_dir) * (1.0 - albedo_dir) * S_dn_dir + \
        (1.0 - tau_dif) * (1.0 - albedo_dif) * S_dn_dif
    Sn_S = tau_dir * (1.0 - albedosoil) * S_dn_dir + \
        tau_dif * (1.0 - albedosoil) * S_dn_dif
    return np.asarray(Sn_C), np.asarray(Sn_S)


def calc_tau_below_Campbell(LAI, sza, fvis, fnir, rho_leaf_vis, tau_leaf_vis,
                          rho_leaf_nir, tau_leaf_nir, rsoilv, rsoiln, x_LAD=1,
                          LAI_eff=None):
    ''' Radiation transmission through a canopy.

    Estimate transmitted shorwave (longwave) radiation through a canopy using the [Campbell1998]_
    Radiative Transfer Model.

    Parameters
    ----------
    LAI : float
        Effecive Leaf (Plant) Area Index.
    sza : float
        Sun Zenith Angle (degrees).
    fvis : float
        fration of total visible radiation.
    fnir : float
        fraction of total NIR radiation.
    rho_leaf_vis : float
        Broadband leaf bihemispherical reflectance in the visible region (400-700nm).
    tau_leaf_vis : float
        Broadband leaf bihemispherical transmittance in the visible region (400-700nm).
    rho_leaf_nir : float
        Broadband leaf bihemispherical reflectance in the NIR region (700-2500nm).
    tau_leaf_nir : float
        Broadband leaf bihemispherical transmittance in the NIR region (700-2500nm).
    rsoilv : float
        Broadband soil bihemispherical reflectance in the visible region (400-700nm).
    rsoiln : float
        Broadband soil bihemispherical reflectance in the NIR region (700-2500nm).
    x_LAD : float, optional
        x parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    LAI_eff : float or None, optional
        if set, its value is the directional effective LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies

    Returns
    -------
    tau_dir : float
        Beam canopy transmittance.
    tau_dif : float
        Diffuse canopy transmittance.

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    '''

    # calculate aborprtivity
    ameanv = 1.0 - rho_leaf_vis - tau_leaf_vis
    ameann = 1.0 - rho_leaf_nir - tau_leaf_nir
    # Calculate canopy beam extinction coefficient
    # Modification to include other LADs
    if isinstance(LAI_eff, type(None)):
        LAI_eff = np.asarray(LAI)
    else:
        LAI_eff = np.asarray(LAI_eff)
    # Integrate to get the diffuse transmitance
    taud = 0
    for angle in range(0, 90, 5):
        akd = calc_K_be_Campbell(angle, x_LAD)  # Eq. 15.4
        taub = np.exp(-akd * LAI)
        taud = taud + taub * np.cos(np.radians(angle)) * \
            np.sin(np.radians(angle)) * np.radians(5)
    taud = 2.0 * taud
    # D I F F U S E   C O M P O N E N T S
    # Diffuse light canopy reflection coefficients  for a deep canopy [Campbell1998]_
    # akd=-0.0683*log(LAI)+0.804                  # Fit to Fig 15.4 for x=1
    # [Campbell1998]_
    akd = -np.log(taud) / LAI
    rcpyn = (1.0 - np.sqrt(ameann)) / \
        (1.0 + np.sqrt(ameann))  # Eq 15.7 [Campbell1998]_
    rcpyv = (1.0 - np.sqrt(ameanv)) / (1.0 + np.sqrt(ameanv))
    rdcpyn = 2.0 * akd * rcpyn / (akd + 1.0)  # Eq 15.8 [Campbell1998]_
    rdcpyv = 2.0 * akd * rcpyv / (akd + 1.0)
    # Diffuse canopy transmission coeff (visible)
    expfac = np.sqrt(ameanv) * akd * LAI
    xnum = (rdcpyv * rdcpyv - 1.0) * np.exp(-expfac)
    xden = (rdcpyv * rsoilv - 1.0) + rdcpyv * \
        (rdcpyv - rsoilv) * np.exp(-2.0 * expfac)
    taudv = xnum / xden  # Eq 15.11 [Campbell1998]_
    # Diffuse canopy transmission coeff (NIR)
    expfac = np.sqrt(ameann) * akd * LAI
    xnum = (rdcpyn * rdcpyn - 1.0) * np.exp(-expfac)
    xden = (rdcpyn * rsoiln - 1.0) + rdcpyn * \
        (rdcpyn - rsoiln) * np.exp(-2.0 * expfac)
    taudn = xnum / xden  # Eq 15.11 [Campbell1998]_
    # Diffuse radiation surface albedo for a generic canopy
    # B E A M   C O M P O N E N T S
    # Direct beam extinction coeff (spher. LAD)
    akb = calc_K_be_Campbell(sza, x_LAD)  # Eq. 15.4
    # Direct beam canopy reflection coefficients for a deep canopy
    rcpyn = (1.0 - np.sqrt(ameann)) / \
        (1.0 + np.sqrt(ameann))  # Eq 15.7 [Campbell1998]_
    rcpyv = (1.0 - np.sqrt(ameanv)) / (1.0 + np.sqrt(ameanv))
    rbcpyn = 2.0 * akb * rcpyn / (akb + 1.0)  # Eq 15.8[Campbell1998]_
    rbcpyv = 2.0 * akb * rcpyv / (akb + 1.0)
    # Weighted average albedo
    # Direct beam+scattered canopy transmission coeff (visible)
    expfac = np.sqrt(ameanv) * akb * LAI_eff
    xnum = (rbcpyv * rbcpyv - 1.0) * np.exp(-expfac)
    xden = (rbcpyv * rsoilv - 1.0) + rbcpyv * \
        (rbcpyv - rsoilv) * np.exp(-2.0 * expfac)
    taubtv = xnum / xden  # Eq 15.11 [Campbell1998]_
    # Direct beam+scattered canopy transmission coeff (NIR)
    expfac = np.sqrt(ameann) * akb * LAI_eff
    xnum = (rbcpyn * rbcpyn - 1.0) * np.exp(-expfac)
    xden = (rbcpyn * rsoiln - 1.0) + rbcpyn * \
        (rbcpyn - rsoiln) * np.exp(-2.0 * expfac)
    taubtn = xnum / xden  # Eq 15.11 [Campbell1998]_
    tau_dir = fvis * taubtv + fnir * taubtn
    tau_dif = fvis * taudv + fnir * taudn
    return np.asarray(tau_dir), np.asarray(tau_dif)


#def calc_Sn_below_4SAIL(
#        Es,
#        Ed,
#        LAI,
#        lidf,
#        hotspot,
#        sza,
#        rho_leaf,
#        tau_leaf,
#        rsoil):
#    ''' Soil net shortwave radiation using 4SAIL
#
#    Estimates the shortwave radiation for soil using the [Verhoef2007]_ 4SAIL
#    Radiative Transfer Model.
#
#    Parameters
#    ----------
#    Es: 1D array_like
#        Narrowband direct incoming shortwave radiation (W m-2 nm-1) (400-2500 @1nm).
#    Ed : 1D array_like
#        Narrowband difuse incoming shortwave radiation (W m-2 nm-1) (400-2500 @1nm).
#    LAI : float
#        Leaf (Plant) Area Index.
#    lidf : 1D array_like
#        Leaf Inclination Distribution Function, 5 degrees step by default.
#    hotspot : float
#        hotspot parameters, use 0 to ignore the hotspot effect (turbid medium).
#    sza : float
#        Sun Zenith Angle (degrees).
#    rho_leaf :: 1D array_like
#        Narrowband leaf bihemispherical reflectance. It should have the same size as Es and Ed
#    tau_leaf : 1D array_like
#        Narrowband leaf bihemispherical transmittance. It should have the same size as Es and Ed
#    rsoil : 1D array_like
#        Narrowband soil bihemispherical reflectance (400-2500 @1nm). It should have the same size as Es and Ed
#
#    Returns
#    -------
#    Es_1 : float
#        Direct downwelling sw radiance at the bottom layer (W m-2).
#    Ed_down_1 : float
#        Difuse downwelling sw radiance at the bottom layer (W m-2).
#
#    References
#    ----------
#    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
#        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
#        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
#        http://dx.doi.org/10.1109/TGRS.2007.895844.
#     '''
#
#    # Run 4SAIL, vza and psi are not relevant
#    [tss,
#     too,
#     tsstoo,
#     rdd,
#     tdd,
#     rsd,
#     tsd,
#     rdo,
#     tdo,
#     rso,
#     rsos,
#     rsod,
#     rddt,
#     rsdt,
#     rdot,
#     rsodt,
#     rsost,
#     rsot,
#     gammasdf,
#     gammasdb,
#     gammaso] = FourSAIL(LAI,
#                         hotspot,
#                         lidf,
#                         sza,
#                         0,
#                         0,
#                         rho_leaf,
#                         tau_leaf,
#                         rsoil)
#    # Downwelling solar beam radiation at ground level (beam transmissnion)
#    Es_1 = tss * Es
#    # Upwelling diffuse shortwave radiation at ground level
#    Ed_up_1 = (rsoil * (Es_1 + tsd * Es + tdd * Ed)) / (1. - rsoil * rdd)
#    # Downwelling diffuse shortwave radiation at ground level
#    Ed_down_1 = tsd * Es + tdd * Ed + rdd * Ed_up_1
#    # Downwelling shortwave (beam and diffuse) below the canopy
#    Es_1 = np.sum(Es_1)   # broadband net canopy sw radiation
#    Ed_down_1 = np.sum(Ed_down_1)
#    return np.asarray(Es_1), np.asarray(Ed_down_1)
