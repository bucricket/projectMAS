# This file is part of pyTSEB for calculating the canopy clumping index
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

"""
Created on Apr 6 2015
@author: Hector Nieto (hnieto@ias.csic.es)

Modified on Mar 28 2016
@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
Routines for calculating the clumping index for both randomly placed canopies and
structured row crops such as vineyards.

PACKAGE CONTENTS
================
* :func:`calc_omega0_Kustas` Nadir viewing clmping factor.
* :func:`calc_omega_Kustas` Clumping index at an incidence angle.
* :func:`calc_omega_rows` Clumping index at an incidence angle for row crops.
"""

import numpy as np


def calc_omega0_Kustas(LAI, f_C, x_LAD=1, isLAIeff=True):
    ''' Nadir viewing clmping factor

    Estimates the clumping factor forcing equal gap fraction between the real canopy
    and the homogeneous case, after [Kustas1999]_.

    Parameters
    ----------
    LAI : float
        Leaf Area Index, it can be either the effective LAI or the real LAI
        , default input LAI is effective.
    f_C : float
        Apparent fractional cover, estimated from large gaps, means that
        are still gaps within the canopy to be quantified.
    x_LAD : float, optional
        Chi parameter for the ellipsoildal Leaf Angle Distribution function of
        [Campbell1988]_ [default=1, spherical LIDF].
    isLAIeff :  bool, optional
        Defines whether the input LAI is effective or local.

    Returns
    -------
    omega0 : float
        clumping index at nadir.

    References
    ----------
    .. [Kustas1999] William P Kustas, John M Norman, Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
 '''

    # Convert input scalars to numpy array
    LAI, f_C, x_LAD = map(np.asarray, (LAI, f_C, x_LAD))
    theta = np.zeros(LAI.shape)
    # Estimate the beam extinction coefficient based on a ellipsoidal LAD function
    # Eq. 15.4 of Campbell and Norman (1998)
    K_be = np.sqrt(x_LAD**2 + np.tan(theta)**2) / \
        (x_LAD + 1.774 * (x_LAD + 1.182)**-0.733)
    if isLAIeff:
        F = LAI / f_C
    else:  # The input LAI is actually the real LAI
        F = np.array(LAI)
    # Calculate the gap fraction of our canopy
    trans = np.asarray(f_C * np.exp(-K_be * F) + (1.0 - f_C))
    trans[trans <= 0] = 1e-36
    # and then the nadir clumping factor
    omega0 = -np.log(trans) / (F * K_be)
    return omega0


def calc_omega_Kustas(omega0, theta, w_C=1):
    ''' Clumping index at an incidence angle.

    Estimates the clumping index for a given incidence angle assuming randomnly placed canopies.

    Parameters
    ----------
    omega0 : float
        clumping index at nadir, estimated for instance by :func:`calc_omega0_Kustas`.
    theta : float
        incidence angle (degrees).
    w_C :  float, optional
        canopy witdth to height ratio, [default = 1].

    Returns
    -------
    Omega : float
        Clumping index at an incidenc angle.

    References
    ----------
    .. [Kustas1999] William P Kustas, John M Norman, Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''

    w_C = 1.0 / w_C
    omega = omega0 / (omega0 + (1.0 - omega0) *
                      np.exp(-2.2 * (np.radians(theta))**(3.8 - 0.46 * w_C)))
    return omega


def calc_omega_rows(
        LAI,
        f_c0,
        theta=0,
        psi=0,
        D=1,
        x_LAD=1,
        isLAIeff=True,
        h_ratio=0):
    ''' Clumping index in row crops.

    Calculates the clumping index for a given incidence angle assuming structured row crops.

    Parameters
    ----------
    LAI : float
        Leaf Area Index, it can be either the effective LAI or the real LAI
        depending on isLAIeff, default input LAI is effective.
    f_c0 : float
        Apparent nadir fractional cover, estimated from large gaps, means that
        are still gaps within the canopy to be quantified.
    theta : float, optional
        Incidence angle (degrees), default nadir.
    psi : float, optional
        relative row-sun azimiuth angle
    D :  float, optional
        canopy witdth to height ratio, [default = 1].
    x_LAD : float, optional
        Chi parameter for the ellipsoildal Leaf Angle Distribution function of
        [Campbell1988]_ [default=1, spherical LIDF].
    isLAIeff :  bool, optional
        Defines whether the input LAI is effective or real. [default True]
    h_ratio : float, optional
        Canopy base height to canopy total height ratio (default=0, canopy from the ground).

    Returns
    -------
    omega : float
        clumping index at an incidence angle.
    '''

    # Convert input scalars in numpy arrays
    LAI, f_c0, theta, psi, D, x_LAD, h_ratio = map(
        np.asarray, (LAI, f_c0, theta, psi, D, x_LAD, h_ratio))
    omega = np.zeros(LAI.shape)
    # Calculate the zenith angle of incidence towards the normal of the row
    # direction
    tan_alpha_x = np.tan(np.radians(theta)) * abs(np.sin(np.radians(psi)))
    # Calculate the fraction that is trasmitted trough vegetation
    f_c = np.asarray(f_c0 * (1.0 + (tan_alpha_x / D) * (1.0 - h_ratio)))
    # Ignore overlapping shadows from other rows
    f_c[f_c0 * tan_alpha_x / D > 1.0] = f_c[f_c0 * tan_alpha_x / D > 1.0] + \
        (f_c0[f_c0 * tan_alpha_x / D > 1.0] * 
        tan_alpha_x[f_c0 * tan_alpha_x / D > 1.0] / 
        D[f_c0 * tan_alpha_x / D > 1.0] - 1.0)

    f_c = np.minimum(f_c, 1.0)
    # Estimate the beam extinction coefficient based on a elipsoidal LAD function
    # Eq. 15.4 of Campbell and Norman (1998)
    K_be = np.sqrt(x_LAD**2 + np.tan(np.radians(theta))**2) / \
        (x_LAD + 1.774 * (x_LAD + 1.182)**-0.733)
    if isLAIeff:
        F = LAI / f_c0
    else:
        F = np.asarray(LAI)
    # Calculate the real gap fraction of our canopy
    trans = f_c * np.exp(-K_be * F) + (1.0 - f_c)
    # and then the clumping factor
    omega[trans > 0] = -np.log(trans[trans > 0]) / \
        (F[trans > 0] * K_be[trans > 0])

    return omega
