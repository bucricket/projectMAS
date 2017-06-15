# This file is part of pyTSEB for running different TSEB models
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
@author: Hector Nieto (hnieto@ias.csic.es)
Modified on Jan 27 2016
@author: Hector Nieto (hnieto@ias.csic.es)
DESCRIPTION
===========
This package contains the main routines inherent of Two Source Energy Balance `TSEB` models.
Additional functions needed in TSEB, such as computing of net radiation or estimating the
resistances to heat and momentum transport are imported.
* :doc:`netRadiation` for the estimation of net radiation and radiation partitioning.
* :doc:`ClumpingIndex` for the estimatio of canopy clumping index.
* :doc:`meteoUtils` for the estimation of meteorological variables.
* :doc:`resistances` for the estimation of the resistances to heat and momentum transport.
* :doc:`MOsimilarity` for the estimation of the Monin-Obukhov length and MOST-related variables.
PACKAGE CONTENTS
================
TSEB models
-----------
* :func:`TSEB_PT` Priestley-Taylor TSEB using a single observation of composite radiometric temperature.
Ancillary functions
-------------------
* :func:`calc_F_theta_campbell`. Gap fraction estimation.
* :func:`calc_G_time_diff`. Santanello & Friedl (2003) [Santanello2003]_ soil heat flux model.
* :func:`calc_G_ratio`. Soil heat flux as a fixed fraction of net radiation [Choudhury1987]_.
* :func:`calc_H_C_PT`. Priestley- Taylor Canopy sensible heat flux.
* :func:`calc_T_C_series.` Canopy temperature from canopy sensible heat flux and resistance in series.
* :func:`calc_T_S`. Soil temperature from form composite radiometric temperature.
'''

import numpy as np
#from FourSAIL import FourSAIL

from .meteo_utils import calc_rho,calc_c_p,calc_delta_vapor_pressure,calc_psicr,calc_lambda
from .resistances import calc_z_0H,calc_R_A,calc_R_x_Norman,calc_R_S_Kustas,calc_R_x_Choudhury,calc_R_S_Choudhury,calc_R_x_McNaughton,calc_R_S_McNaughton
from .MO_similarity import calc_u_star,calc_u_C_star,calc_u_Goudriaan,calc_L,calc_A_Goudriaan
from .net_radiation import calc_L_n_Kustas,calc_K_be_Campbell
from .clumping_index import calc_omega0_Kustas
from .TSEB_utils_usda import compute_G0,compute_resistence,albedo_separation
from .TSEB_utils_usda import compute_Rn,temp_separation,compute_stability

#from cachetools import cached

#==============================================================================
# List of constants used in TSEB model and sub-routines
#==============================================================================
# Change threshold in  Monin-Obukhov lengh to stop the iterations
L_thres = 0.00001
# Change threshold in  friction velocity to stop the iterations
u_thres = 0.00001
# mimimun allowed friction velocity
u_friction_min = 0.01
# Maximum number of interations
ITERATIONS = 20
# kB coefficient
kB = 0.0
# Stephan Boltzmann constant (W m-2 K-4)
sb = 5.670373e-8


def TSEB_PT(
    Tr_K,
    vza,
    T_A_K,
    u,
    ea,
    p,
    Sn_C,
    Sn_S,
    L_dn,
    LAI,
    hc,
    emis_C,
    emis_S,
    z_0M,
    d_0,
    z_u,
    z_T,
    nullMask,
    leaf_width=0.1,
    z0_soil=0.01,
    alpha_PT=1.26,
    x_LAD=1,
    f_c=1.0,
    f_g=1.0,
    w_C=1.0,
    resistance_form=0,
    calcG_array=[
        [1],
        0.35],
        UseL=False):
    '''Priestley-Taylor TSEB
    Calculates the Priestley Taylor TSEB fluxes using a single observation of
    composite radiometric temperature and using resistances in series.
    Parameters
    ----------
    Tr_K : float
        Radiometric composite temperature (Kelvin).
    vza : float
        View Zenith Angle (degrees).
    T_A_K : float
        Air temperature (Kelvin).
    u : float
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).
    L_dn : float
        Downwelling longwave radiation (W m-2).
    LAI : float
        Effective Leaf Area Index (m2 m-2).
    hc : float
        Canopy height (m).
    emis_C : float
        Leaf emissivity.
    emis_S : flaot
        Soil emissivity.
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    z_u : float
        Height of measurement of windspeed (m).
    z_T : float
        Height of measurement of air temperature (m).
    leaf_width : float, optional
        average/effective leaf width (m).
    z0_soil : float, optional
        bare soil aerodynamic roughness length (m).
    alpha_PT : float, optional
        Priestley Taylor coeffient for canopy potential transpiration,
        use 1.26 by default.
    x_LAD : float, optional
        Campbell 1990 leaf inclination distribution function chi parameter.
    f_c : float, optional
        Fractional cover.
    f_g : float, optional
        Fraction of vegetation that is green.
    w_C : float, optional
        Canopy width to height ratio.
    resistance_form : int, optional
        Flag to determine which Resistances R_x, R_S model to use.
            * 0 [Default] Norman et al 1995 and Kustas et al 1999.
            * 1 : Choudhury and Monteith 1988.
            * 2 : McNaughton and Van der Hurk 1995.
    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.
            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.calc_G_time_diff`).
    UseL : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.
    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    T_S : float
        Soil temperature  (Kelvin).
    T_C : float
        Canopy temperature  (Kelvin).
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
    L_nS : float
        Soil net longwave radiation (W m-2)
    L_nC : float
        Canopy net longwave radiation (W m-2)
    LE_C : float
        Canopy latent heat flux (W m-2).
    H_C : float
        Canopy sensible heat flux (W m-2).
    LE_S : float
        Soil latent heat flux (W m-2).
    H_S : float
        Soil sensible heat flux (W m-2).
    G : float
        Soil heat flux (W m-2).
    R_S : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_A : float
        Aerodynamic resistance to heat transport (s m-1).
    u_friction : float
        Friction velocity (m s-1).
    L : float
        Monin-Obuhkov length (m).
    n_iterations : int
        number of iterations until convergence of L.
    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293,
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    .. [Kustas1999] William P Kustas, John M Norman, Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29,
        http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''

    # Convert input float scalars to arrays and parameters size
    Tr_K = np.asarray(Tr_K)
#    (vza,
#     T_A_K,
#     u,
#     ea,
#     p,
#     Sn_C,
#     Sn_S,
#     L_dn,
#     LAI,
#     hc,
#     emis_C,
#     emis_S,
#     z_0M,
#     d_0,
#     z_u,
#     z_T,
#     nullMask,
#     leaf_width,     
#     z0_soil,
#     alpha_PT,
#     x_LAD,
#     f_c,
#     f_g,
#     w_C,
#     calcG_array) = map(_check_default_parameter_size,
#                        [vza,
#                         T_A_K,
#                         u,
#                         ea,
#                         p,
#                         Sn_C,
#                         Sn_S,
#                         L_dn,
#                         LAI,
#                         hc,
#                         emis_C,
#                         emis_S,
#                         z_0M,
#                         d_0,
#                         z_u,
#                         z_T,
#                         nullMask,
#                         leaf_width,
#                         z0_soil,
#                         alpha_PT,
#                         x_LAD,
#                         f_c,
#                         f_g,
#                         w_C,
#                         calcG_params[1]],
#                        [Tr_K] * 24)
    calcG_array[1] = _check_default_parameter_size(calcG_array[1],Tr_K)
    # Create the output variables
    [flag, T_S, T_C, T_AC, Ln_S, Ln_C, LE_C, H_C, LE_S, H_S, G, R_S, R_x,
        R_A, iterations] = [np.zeros(Tr_K.shape) for i in range(15)]

    # iteration of the Monin-Obukhov length
    if isinstance(UseL, bool):
        # Initially assume stable atmospheric conditions and set variables for
        L = np.asarray(np.zeros(T_S.shape) + np.inf)
        L = np.asarray(np.zeros(T_S.shape))
        max_iterations = ITERATIONS
    else:  # We force Monin-Obukhov lenght to the provided array/value
        L = np.asarray(np.ones(T_S.shape) * UseL)
        max_iterations = 1  # No iteration
    # Calculate the general parameters
    if ea.sum() == 0.0:
        z = 350.0
        rho = 101.3*((((T_A_K)-(0.0065*z))/(T_A_K))**5.26)/1.01/(T_A_K)/0.287
        c_p = np.tile(1004.16,np.shape(T_A_K))
    else:
        rho = calc_rho(p, ea, T_A_K)  # Air density
        c_p = calc_c_p(p, ea)  # Heat capacity of air
    z_0H = calc_z_0H(z_0M, kB=kB)  # Roughness length for heat transport

    # Calculate LAI dependent parameters for dataset where LAI > 0
    omega0 = calc_omega0_Kustas(LAI, f_c, x_LAD=x_LAD, isLAIeff=True)
    F = np.asarray(LAI / f_c)  # Real LAI
    # Fraction of vegetation observed by the sensor
    f_theta = calc_F_theta_campbell(vza, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)

    # Initially assume stable atmospheric conditions and set variables for
    # iteration of the Monin-Obukhov length
    u_friction = calc_u_star(u, z_u, L, d_0, z_0M)
    u_friction = np.asarray(np.maximum(u_friction_min, u_friction))
    L_old = np.ones(Tr_K.shape)
    L_diff = np.asarray(np.ones(Tr_K.shape))
    # First assume that canopy temperature equals the minumum of Air or
    # radiometric T
    T_C = np.asarray(np.minimum(Tr_K, T_A_K))
    flag, T_S = calc_T_S(Tr_K, T_C, f_theta)
    flag[nullMask==-9999]=255
    iterMask = np.zeros(LAI.shape)
    conv1 = 0.0
    # Outer loop for estimating stability.
    # Stops when difference in consecutives L is below a given threshold
    for n_iterations in range(max_iterations):
        i = flag != 255
        #if np.all(L_diff[i] < L_thres):
        if conv1 > 0.98:
            #print("Finished interation with a max. L diff: " + str(np.max(L_diff)))
            break
#        print("Iteration " + str(n_iterations) +
#              ", max. L diff: " + str(np.max(L_diff)))
        iterations[
            np.logical_and(
                L_diff >= L_thres,
                flag != 255)] = n_iterations

        # Inner loop to iterativelly reduce alpha_PT in case latent heat flux
        # from the soil is negative. The initial assumption is of potential
        # canopy transpiration.
        flag[np.logical_and(L_diff >= L_thres, flag != 255)] = 0
        LE_S[np.logical_and(L_diff >= L_thres, flag != 255)] = -1
        alpha_PT_rec = np.asarray(alpha_PT + 0.1)
        while np.any(LE_S[i] < 0):
            i = np.logical_and.reduce(
                (LE_S < 0, L_diff >= L_thres, flag != 255))

            alpha_PT_rec[i] -= 0.1

            # There cannot be negative transpiration from the vegetation
            alpha_PT_rec[alpha_PT_rec <= 0.0] = 0.0
            flag[np.logical_and(i, alpha_PT_rec == 0.0)] = 5

            flag[
                np.logical_and.reduce(
                    (i, alpha_PT_rec < alpha_PT, alpha_PT_rec > 0.0))] = 3

            # Calculate the aerodynamic resistance
            R_A[i] = calc_R_A(z_T[i], u_friction[i], L[i], d_0[i], z_0H[i])
            # Calculate soil and canopy resistances
            U_C = calc_u_C_star(
                u_friction[i], hc[i], d_0[i], z_0M[i], L=L[i])
            if resistance_form == 0:
                # Wind speed is highly attenuated within the canopy volume
                u_d_zm = calc_u_Goudriaan(
                    U_C, hc[i], F[i], leaf_width[i], d_0[i] + z_0M[i])
    
                # Vegetation in series with soil, i.e. well mixed, so we use
                # the landscape LAI
                R_x[i] = calc_R_x_Norman(LAI[i], leaf_width[i], u_d_zm)
                i = np.logical_and.reduce(
                (LE_S < 0, flag != 255, L_diff >= L_thres,
                 np.logical_not(np.isnan(R_x))))
                # Calculate soil and canopy resistances
                U_C = calc_u_C_star(
                u_friction[i], hc[i], d_0[i], z_0M[i], L=L[i])
                # Clumped vegetation enhanced wind speed for the soil surface
                u_S = calc_u_Goudriaan(
                    U_C, hc[i], omega0[i] * F[i], leaf_width[i], z0_soil[i])
                R_S[i] = calc_R_S_Kustas(u_S, T_S[i] - T_C[i])
            elif resistance_form == 1:
                # Vegetation in series with soil, i.e. well mixed, so we use
                # the landscape LAI
                R_x[i] = calc_R_x_Choudhury(U_C, LAI[i], leaf_width[i])
                R_S[i] = calc_R_S_Choudhury(
                    u_friction[i], hc[i], z_0M[i], d_0[i], z_u[i], z0_soil[i])
            elif resistance_form == 2:
                # Vegetation in series with soil, i.e. well mixed, so we use
                # the landscape LAI
                R_x[i] = calc_R_x_McNaughton(
                    LAI[i], leaf_width[i], u_friction[i])
                R_S[i] = calc_R_S_McNaughton(u_friction[i])
            elif resistance_form == 3:
                # Clumped vegetation enhanced wind speed for the soil surface
                alpha_k = calc_A_Goudriaan(
                    hc[i], omega0[i] * F[i], leaf_width[i])
                # Wind speed is highly attenuated within the canopy volume
                alpha_prime = calc_A_Goudriaan(hc[i], F[i], leaf_width[i])
                # Vegetation in series with soil, i.e. well mixed, so we use
                # the landscape LAI
                R_x[i] = calc_R_x_Choudhury(
                    U_C, LAI[i], leaf_width[i], alpha_prime=alpha_prime)
                R_S[i] = calc_R_S_Choudhury(u_friction[i], hc[i], z_0M[i], d_0[
                                               i], z_u, z0_soil[i], alpha_k=alpha_k)
            else:
                # Clumped vegetation enhanced wind speed for the soil surface
                u_S = calc_u_Goudriaan(
                    U_C, hc[i], omega0[i] * F[i], leaf_width[i], z0_soil[i])
                # Wind speed is highly attenuated within the canopy volume
                u_d_zm = calc_u_Goudriaan(
                    U_C, hc[i], F[i], leaf_width[i], d_0[i] + z_0M[i])
                # Vegetation in series with soil, i.e. well mixed, so we use
                # the landscape LAI
                R_x[i] = calc_R_x_Norman(LAI[i], leaf_width[i], u_d_zm)
                R_S[i] = calc_R_S_Kustas(u_S, T_S[i] - T_C[i])
            R_S = np.asarray(np.maximum(1e-3, R_S))
            R_x = np.asarray(np.maximum(1e-3, R_x))
            R_A = np.asarray(np.maximum(1e-3, R_A))

            # Calculate net longwave radiation with current values of T_C and T_S
            Ln_C[i], Ln_S[i] = calc_L_n_Kustas(
                T_C[i], T_S[i], L_dn[i], LAI[i], emis_C[i], emis_S[i])
            delta_Rn = Sn_C + Ln_C
            Rn_S = Sn_S + Ln_S
            # Calculate the canopy and soil temperatures using the Priestley
            # Taylor appoach
            H_C[i] = calc_H_C_PT(
                delta_Rn[i],
                f_g[i],
                T_A_K[i],
                p[i],
                c_p[i],
                alpha_PT_rec[i])
            T_C[i] = calc_T_C_series(Tr_K[i], T_A_K[i], R_A[i], R_x[i], R_S[
                                   i], f_theta[i], H_C[i], rho[i], c_p[i])

            # Calculate soil temperature
            flag_t = np.zeros(flag.shape)
            flag_t[i], T_S[i] = calc_T_S(Tr_K[i], T_C[i], f_theta[i])
            flag[flag_t == 255] = 255
            LE_S[flag_t == 255] = 0

            # Recalculate soil resistance using new soil temperature
            if resistance_form == 0:
                R_S[i] = calc_R_S_Kustas(u_S, T_S[i] - T_C[i])
                R_S = np.asarray(np.maximum(1e-3, R_S))

            i = np.logical_and.reduce(
                (LE_S < 0, flag != 255, L_diff >= L_thres,
                 np.logical_not(np.isnan(R_x))))

            # Get air temperature at canopy interface
            T_AC[i] = ((T_A_K[i] / R_A[i] + T_S[i] / R_S[i] + T_C[i] / R_x[i])
                       / (1.0 / R_A[i] + 1.0 / R_S[i] + 1.0 / R_x[i]))

            # Calculate soil fluxes
            H_S[i] = rho[i] * c_p[i] * (T_S[i] - T_AC[i]) / R_S[i]

            # Compute Soil Heat Flux Ratio
            G[i] = calc_G([calcG_array[0], calcG_array], Rn_S, i)
            #G[i]=calc_G_ratio(Rn_S[i], calcG_array[1][i])
            #Compute Soil Heat Flux Ratio
            if calc_G[0]==0:
                G[i]=CalcG[1][i]
            elif calc_G[0]==1:
                G[i]=CalcG_Ratio(Rn_S[i], calcG_array[1][i])
            elif calc_G[0]==2:
                G[i]=calc_G_time_diff(Rn_S[i], calcG_array[1][i])
                
            # Estimate latent heat fluxes as residual of energy balance at the
            # soil and the canopy
            LE_S[i] = Rn_S[i] - G[i] - H_S[i]
            LE_C[i] = delta_Rn[i] - H_C[i]

            # Special case if there is no transpiration from vegetation.
            # In that case, there should also be no evaporation from the soil
            # and the energy at the soil should be conserved.
            # See end of appendix A1 in Guzinski et al. (2015).
            noT = np.logical_and(i, LE_C == 0)
            H_S[noT] = np.minimum(H_S[noT], Rn_S[noT] - G[noT])
            G[noT] = np.maximum(G[noT], Rn_S[noT] - H_S[noT])
            LE_S[noT] = 0

            # Calculate total fluxes
            H = np.asarray(H_C + H_S)
            LE = np.asarray(LE_C + LE_S)
            # Now L can be recalculated and the difference between iterations
            # derived
            if isinstance(UseL, bool):
                L[i] = calc_L(
                    u_friction[i],
                    T_A_K[i],
                    rho[i],
                    c_p[i],
                    H[i],
                    LE[i])
                # Calculate again the friction velocity with the new stability
                # correctios
                u_friction[i] = calc_u_star(
                    u[i], z_u[i], L[i], d_0[i], z_0M[i])
                u_friction = np.asarray(np.maximum(u_friction_min, u_friction))

        if isinstance(UseL, bool):
            L_diff = np.asarray(np.fabs(L - L_old) / np.fabs(L_old))
            #L_diff[np.isnan(L_diff)] = float('inf')
            L_old = np.array(L)
            L_old[L_old == 0] = 1e-36
            iterMask[np.where(L_diff < L_thres)]=1.0
            conv1 = np.sum(iterMask)/(LAI.size)
            #print 'TSEB convergence=%f percent' % (conv1*100.)

    (flag,
     T_S,
     T_C,
     T_AC,
     L_nS,
     L_nC,
     LE_C,
     H_C,
     LE_S,
     H_S,
     G,
     R_S,
     R_x,
     R_A,
     u_friction,
     L,
     n_iterations) = map(np.asarray,
                         (flag,
                          T_S,
                          T_C,
                          T_AC,
                          Ln_S,
                          Ln_C,
                          LE_C,
                          H_C,
                          LE_S,
                          H_S,
                          G,
                          R_S,
                          R_x,
                          R_A,
                          u_friction,
                          L,
                          iterations))

    return flag, T_S, T_C, T_AC, L_nS, L_nC, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, u_friction, L, n_iterations

def TSEB_PT_usda(
    Tr_K,
    vza,
    T_A_K,
    u,
    p,
    Rs_1,
    zs,
    aleafv, 
    aleafn, 
    aleafl, 
    adeadv, 
    adeadn, 
    adeadl,
    albedo,
    ndvi,
    lai,
    clump,
    hc,
    mask,
    time,
    t_rise,
    t_end,
    leaf_width=0.1,
    a_PT_in=1.32):
    '''Priestley-Taylor TSEB
    Calculates the Priestley Taylor TSEB fluxes using a single observation of
    composite radiometric temperature and using resistances in series.
    Parameters
    ----------
    Tr_K : float
        Radiometric composite temperature (Kelvin).
    vza : float
        View Zenith Angle (degrees).
    T_A_K : float
        Air temperature (Kelvin).
    u : float
        Wind speed above the canopy (m s-1).
    ea : float
        Water vapour pressure above the canopy (mb).
    p : float
        Atmospheric pressure (mb), use 1013 mb by default.
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).
    L_dn : float
        Downwelling longwave radiation (W m-2).
    lai : float
        Effective Leaf Area Index (m2 m-2).
    hc : float
        Canopy height (m).
    emis_C : float
        Leaf emissivity.
    emis_S : flaot
        Soil emissivity.
    z_0M : float
        Aerodynamic surface roughness length for momentum transfer (m).
    d_0 : float
        Zero-plane displacement height (m).
    z_u : float
        Height of measurement of windspeed (m).
    z_T : float
        Height of measurement of air temperature (m).
    leaf_width : float, optional
        average/effective leaf width (m).
    z0_soil : float, optional
        bare soil aerodynamic roughness length (m).
    alpha_PT : float, optional
        Priestley Taylor coeffient for canopy potential transpiration,
        use 1.26 by default.
    x_LAD : float, optional
        Campbell 1990 leaf inclination distribution function chi parameter.
    f_c : float, optional
        Fractional cover.
    f_g : float, optional
        Fraction of vegetation that is green.
    w_C : float, optional
        Canopy width to height ratio.
    resistance_form : int, optional
        Flag to determine which Resistances R_x, R_S model to use.
            * 0 [Default] Norman et al 1995 and Kustas et al 1999.
            * 1 : Choudhury and Monteith 1988.
            * 2 : McNaughton and Van der Hurk 1995.
    calcG_params : list[list,float or array], optional
        Method to calculate soil heat flux,parameters.
            * [[1],G_ratio]: default, estimate G as a ratio of Rn_S, default Gratio=0.35.
            * [[0],G_constant] : Use a constant G, usually use 0 to ignore the computation of G.
            * [[2,Amplitude,phase_shift,shape],time] : estimate G from Santanello and Friedl with G_param list of parameters (see :func:`~TSEB.calc_G_time_diff`).
    UseL : float or None, optional
        If included, its value will be used to force the Moning-Obukhov stability length.
    Returns
    -------
    flag : int
        Quality flag, see Appendix for description.
    T_S : float
        Soil temperature  (Kelvin).
    T_C : float
        Canopy temperature  (Kelvin).
    T_AC : float
        Air temperature at the canopy interface (Kelvin).
    L_nS : float
        Soil net longwave radiation (W m-2)
    L_nC : float
        Canopy net longwave radiation (W m-2)
    LE_C : float
        Canopy latent heat flux (W m-2).
    H_C : float
        Canopy sensible heat flux (W m-2).
    LE_S : float
        Soil latent heat flux (W m-2).
    H_S : float
        Soil sensible heat flux (W m-2).
    G : float
        Soil heat flux (W m-2).
    R_S : float
        Soil aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk canopy aerodynamic resistance to heat transport (s m-1).
    R_A : float
        Aerodynamic resistance to heat transport (s m-1).
    u_friction : float
        Friction velocity (m s-1).
    L : float
        Monin-Obuhkov length (m).
    n_iterations : int
        number of iterations until convergence of L.
    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293,
        http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    .. [Kustas1999] William P Kustas, John M Norman, Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29,
        http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''
      #************************************************************************
      # Correct Clumping Factor
    
    f_green  = 1.
    F = lai*clump                                 # LAI for leaf spherical distribution 
    fc = 1-(np.exp(-0.5*F))  
    fc[fc <= 0.01] = 0.01 
    fc[fc >= 0.9] = 0.9                           # Fraction cover at nadir (view=0)

    lai_c = lai/fc                                # LAI relative to canopy projection only
    fc_q=1-(np.exp(-0.5*F/np.cos(np.deg2rad(vza))))          # Houborg modification (according to Anderson et al. 2005)
    fc_q[fc_q <= 0.05] = 0.05
    fc_q[fc_q >= 0.90] = 0.90

    z0m = 0.123*hc                              #;Brutsaert (1982)
    z0h = z0m.copy()
    d_0 = 2./3.*hc
     
    # Correction of roughness parameters for bare soils (F < 0.1)
    d_0[F<=0.1]=0.00001
    z0m[F<=0.1]=0.01
    z0h[F<=0.1]=0.0001
    
    # Correction of roughness parameters for water bodies (NDVI < 0 and albedo < 0.05)
    ind  = np.logical_and((ndvi<=0.),(albedo <=0.05))
    d_0[ind]=0.00001
    z0m[ind]=0.00035
    z0h[ind]=0.00035
    
    # Check to avoid division by 0 in the next computations
    z0h[z0h==0]=0.001
    z0m[z0m==0]=0.01
        
    
    z_u = np.tile(50.,np.shape(lai))
    z_T = np.tile(50.,np.shape(lai))
# Parameters for In-Canopy Wind Speed Extinction
    
    leaf = (0.28*(F**(0.66667))*(hc**(0.33333))*(leaf_width**(-0.33333)))
    leafc = (0.28*(lai_c**(0.66667))*(hc**(0.33333))*(leaf_width**(-0.33333)))
    leafs = (0.28*(0.1**(0.66667))*(hc**(0.33333))*(leaf_width**(-0.33333)))
    
    #************************************************************************
    # Atmospheric Parameters
    e_s = (0.6108*np.exp((17.27*(T_A_K-273.16))/((T_A_K-273.16)+237.3)))
    Ss = 4098.*e_s/(((T_A_K-273.16)+237.3)**2)
    lambda1 = (2.501-(0.002361*(T_A_K-273.16)))*1000000
    z = np.tile(350.,np.shape(hc))
    ####+++TESING IDL SCRIPT##########
    p = 101.3*(((293.-0.0065*z)/293.)**5.26)
    ###################################
    g = 1615.*p/lambda1

#      sunset_sunrise, julian, lon, lat, 0
#      sunset_sunrise, julian, lon, lat, time

#      albedo_separation, albedo, Rs, F, fc, aleafv, aleafn, aleafl, adeadv, adeadn, adeadl, z, T_A_K, zs, 1

    #************************************************************************
    # Inizialitaziono of TSEB
    a_PT = mask*a_PT_in
    e_atm = 1.0-(0.2811*(np.exp(-0.0003523*((T_A_K-273.16)**2))))

    Rs_c, Rs_s, albedo_c, albedo_s, e_atm, rsoilv_itr, fg_itr = albedo_separation(
                albedo, Rs_1, F, fc, aleafv, aleafn, aleafl, adeadv, adeadn, adeadl, 
                z, T_A_K, zs, 1)
    r_air = 101.3*((((T_A_K)-(0.0065*z))/(T_A_K))**5.26)/1.01/(T_A_K)/0.287  
    cp = np.tile(1004.16,np.shape(T_A_K))
  
    # Assume neutral conditions on first iteration  
    r_ah, r_s, r_x, u_attr = compute_resistence(u, T_A_K, T_A_K, hc, lai, d_0, z0m, z0h, z_u, z_T, leaf_width, leaf, leafs, leafc, 0, 0, 0)
#      compute_resistence, U, T_A_K, T_A_K, hc, lai, d0, z0m, z0h, z_U, z_T, leaf_width, leaf, leafs, leafc, 0, 0, 0     

    Tc=T_A_K
    Ts = (Tr_K-(fc_q*Tc))/(1-fc_q)
    H_iter = np.tile(200.,np.shape(Tc))
#      H_iter = (Tc ne 1000)*200.
    EF_s = np.tile(0.,np.shape(Tc))

    #************************************************************************
    # Start Loop for Stability Correction and Water Stress
    for i in range(35):
        Rn_s, Rn_c, Rn = compute_Rn(albedo_c, albedo_s, T_A_K, Tc, Ts, e_atm, Rs_c, Rs_s, F)
        G0 = compute_G0(Rn, Rn_s, albedo, ndvi, t_rise, t_end, time, EF_s)
      
        lETc = f_green*(a_PT*Ss/(Ss+g))*Rn_c
        lETc[lETc <=0.] = 0.
    #        lETc = ((lETc le 0.)*0.)+((lETc gt 0)*lETc)
        H_c = Rn_c-lETc
        Tc,Ts,Tac = temp_separation(H_c, fc_q, T_A_K, Tr_K, r_ah, r_x, r_s, r_air,cp)
    #        temp_separation, H_c, fc_q, T_A_K, Tr_K, r_ah, r_x, r_s, r_air
      
        H_s = r_air*cp*(Ts-Tac)/r_s
        H_c = r_air*cp*(Tc-Tac)/r_x
        H = H_s+H_c
    
        lEs = Rn_s-G0-H_s
        lETc = Rn_c-H_c
      
        H[H==0.]=10.
    #        H = ((H eq 0)*10)+((H ne 0)*H)
        r_ah[r_ah ==0.]=10.
    #        r_ah = ((r_ah ne 0)*r_ah)+((r_ah eq 0)*10)
        mask_iter  = np.logical_and((H_iter/H) <= 1.05,(H_iter/H) >= 0.95)
    #        mask_iter = (((H_iter/H) le 1.05) and ((H_iter/H) ge 0.95))
        chk_iter = np.sum(mask_iter)/np.size(mask_iter)
#        print(chk_iter)
    #        chk_iter = total(mask_iter)/n_elements(mask_iter)
        fm,fh,fm_h = compute_stability(H, Tr_K, r_air,cp, u_attr, z_u, z_T, hc, d_0, z0m, z0h)
    #        compute_stability, H, Tr_K, r_air, u_attr, z_u, z_T, hc, d0, z0m, z0h
        r_ah, r_s, r_x, u_attr = compute_resistence(u, Ts, Tc, hc, lai, d_0, z0m, z0h, z_u, z_T, leaf_width, leaf, leafs, leafc, fm, fh, fm_h)
    #        compute_resistence, u, Ts, Tc, hc, lai, d0, z0m, z0h, z_u, z_T, leaf_width, leaf, leafs, leafc, fm, fh, fm_h
    
        a_PT[lEs<=0.] = a_PT[lEs<=0.]-0.05
    #        a_PT = ((lEs gt 0)*a_PT)+((lEs le 0)*(a_PT-0.05))       ; reduce alpha-PT if soil evaporaton is <0
    #       ;a_PT = ((lETc gt 0)*a_PT)+((lETc le 0)*0.)             ; canopy transpiration should be >=0
        a_PT[a_PT <= 0.] =  0.01
    #        a_PT = ((a_PT gt 0)*a_PT)+((a_PT le 0)*0.01)            ; canopy transpiration should be >=0
      
        H_iter = H
        den_s = Rn_s-G0
        den_s[den_s==0.] = np.nan
    #        den_s = (((Rn_s-G0) eq 0)*(bad))+(((Rn_s-G0) ne 0)*(Rn_s-G0))
    #        hold=where(den_s EQ bad, vct)
    #        if vct GT 0 then den_s[where(den_s EQ bad)]=!values.f_nan:
        
        EF_s = lEs/den_s
      
    #      ENDFOR ; ii (Loop for Stability Correction and Water Stress)
    
    #************************************************************************
    # Check Energy Balance Closure
    ind = [a_PT <= 0.01]
    lEs[ind]=1.
      #      lEs = ((a_PT gt 0.01)*lEs)+((a_PT le 0.01)*1.)
    lETc[ind]=1.
#      lETc = ((a_PT gt 0.01)*lETc)+((a_PT le 0.01)*1.)
    G0[ind] = Rn_s[ind]-H_s[ind]
#      G0 = ((a_PT gt 0.01)*G0)+((a_PT le 0.01)*(Rn_s-H_s))
    ind = lEs > Rn_s
    lEs[ind] = Rn_s[ind]
#      lEs = ((lEs gt Rn_s)*Rn_s)+((lEs le Rn_s)*lEs)
    H_s[ind] = Rn_s[ind]-G0[ind]-lEs[ind]
#      H_s = ((lEs gt Rn_s)*(Rn_s-G0-lEs))+((lEs le Rn_s)*H_s)
    ind = lETc > Rn_c+100.
    lETc[ind] = Rn_c[ind]+100.
#      lETc = ((lETc gt (Rn_c+100))*(Rn_c+100))+((lETc le (Rn_c+100))*lETc)
    H_c[ind] = -100.
#      H_c = ((lETc gt (Rn_c+100))*(-100.))+((lETc le Rn_c)*H_c)
    
    lEs = Rn_s-G0-H_s
    lETc = Rn_c-H_c 
    
    flag=mask

    (flag,
     Ts,
     Tc,
     Tac,
     lETc,
     H_c,
     lEs,
     H_s,
     G0,
     R_s,
     R_x,
     R_ah) = map(np.asarray,
                    (flag,
                     Ts,
                     Tc,
                     Tac,
                     lETc,
                     H_c,
                     lEs,
                     H_s,
                     G0,
                     r_s,
                     r_x,
                     r_ah))

    return flag, Ts, Tc, Tac, lETc, H_c, lEs, H_s, G0, r_s, r_x, r_ah      
 


def calc_F_theta_campbell(theta, F, w_C=1, Omega0=1, x_LAD=1):
    '''Calculates the fraction of vegetatinon observed at an angle.
    Parameters
    ----------
    theta : float
        Angle of incidence (degrees).
    F : float
        Real Leaf (Plant) Area Index.
    w_C : float
        Ratio of vegetation height versus width, optional (default = 1).
    Omega0 : float
        Clumping index at nadir, optional (default =1).
    x_LAD : float
        Chi parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_LAD=1 for a spherical LAD.
    Returns
    -------
    f_theta : float
        fraction of vegetation obsserved at an angle.
    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293, http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    '''

    # First calcualte the angular clumping factor Omega based on eq (3) from
    # W.P. Kustas, J.M. Norman,  Agricultural and Forest Meteorology 94 (1999)
    # CHECK: should theta here be in degrees or radians
    OmegaTheta = Omega0 / (Omega0 + (1.0 - Omega0) *
                           np.exp(-2.2 * np.radians(theta)**(3.8 - 0.46 * w_C)))
    # Estimate the beam extinction coefficient based on a elipsoidal LAD function
    # Eq. 15.4 of Campbell and Norman (1998)
    K_be = calc_K_be_Campbell(theta, x_LAD)
    ftheta = 1.0 - np.exp(-K_be * OmegaTheta * F)
    return np.asarray(ftheta)


def calc_G(calcG_params, Rn_S, i=None):

    if i is None:
        i = np.ones(Rn_S.shape, dtype=bool)
    if calcG_params[0][0] == 0:
        G = calcG_params[1][i]
    elif calcG_params[0][0] == 1:
        G = calc_G_ratio(Rn_S[i], calcG_params[1][i])
    elif calcG_params[0][0] == 2:
        G = calc_G_time_diff(Rn_S[i], [calcG_params[1][i], calcG_params[
                           0][1], calcG_params[0][2], calcG_params[0][3]])
    return np.asarray(G)


def calc_G_time_diff(R_n, G_param=[12.0, 0.35, 3.0, 24.0]):
    ''' Estimates Soil Heat Flux as function of time and net radiation.
    Parameters
    ----------
    R_n : float
        Net radiation (W m-2).
    G_param : tuple(float,float,float,float)
        tuple with parameters required (time, Amplitude,phase_shift,shape).
            time: float
                time of interest (decimal hours).
            Amplitude : float
                maximum value of G/Rn, amplitude, default=0.35.
            phase_shift : float
                shift of peak G relative to solar noon (default 3hrs after noon).
            shape : float
                shape of G/Rn, default 24 hrs.
    Returns
    -------
    G : float
        Soil heat flux (W m-2).
    References
    ----------
    .. [Santanello2003] Joseph A. Santanello Jr. and Mark A. Friedl, 2003: Diurnal Covariation in
        Soil Heat Flux and Net Radiation. J. Appl. Meteor., 42, 851-862,
        http://dx.doi.org/10.1175/1520-0450(2003)042<0851:DCISHF>2.0.CO;2.'''

    # Get parameters
    time = 12.0 - G_param[0]
    A = G_param[1]
    phase_shift = G_param[2]
    B = G_param[3]
    G_ratio = A * np.cos(2.0 * np.pi * (time + phase_shift) / B)
    G = R_n * G_ratio
    return np.asarray(G)


def calc_G_ratio(Rn_S, G_ratio=0.35):
    '''Estimates Soil Heat Flux as ratio of net soil radiation.
    Parameters
    ----------
    Rn_S : float
        Net soil radiation (W m-2).
    G_ratio : float, optional
        G/Rn_S ratio, default=0.35.
    Returns
    -------
    G : float
        Soil heat flux (W m-2).
    References
    ----------
    .. [Choudhury1987] B.J. Choudhury, S.B. Idso, R.J. Reginato, Analysis of an empirical model
        for soil heat flux under a growing wheat crop for estimating evaporation by an
        infrared-temperature based energy balance equation, Agricultural and Forest Meteorology,
        Volume 39, Issue 4, 1987, Pages 283-297,
        http://dx.doi.org/10.1016/0168-1923(87)90021-9.
    '''

    G = G_ratio * Rn_S
    return np.asarray(G)



def calc_H_C_PT(delta_R_ni, f_g, T_A_K, P, c_p, alpha):
    '''Calculates canopy sensible heat flux based on the Priestley and Taylor formula.
    Parameters
    ----------
    delta_R_ni : float
        net radiation divergence of the vegetative canopy (W m-2).
    f_g : float
        fraction of vegetative canopy that is green.
    T_A_K : float
        air temperature (Kelvin).
    P : float
        air pressure (mb).
    c_p : float
        heat capacity of moist air (J kg-1 K-1).
    alpha : float
        the Priestley Taylor parameter.
    Returns
    -------
    H_C : float
        Canopy sensible heat flux (W m-2).
    References
    ----------
    Equation 14 in [Norman1995]_
    '''

    # slope of the saturation pressure curve (kPa./deg C)
    s = calc_delta_vapor_pressure(T_A_K)
    s = s * 10  # to mb
    # latent heat of vaporisation (MJ./kg)
    Lambda = calc_lambda(T_A_K)
    # psychrometric constant (mb C-1)
    gama = calc_psicr(P, Lambda)
    s_gama = s / (s + gama)
    H_C = delta_R_ni * (1.0 - alpha * f_g * s_gama)
    return np.asarray(H_C)

def calc_T_C_series(Tr_K, T_A_K, R_A, R_x, R_S, f_theta, H_C, rho, c_p):
    '''Estimates canopy temperature from canopy sensible heat flux and
    resistance network in series.
    Parameters
    ----------
    Tr_K : float
        Directional Radiometric Temperature (K).
    T_A_K : float
        Air Temperature (K).
    R_A : float
        Aerodynamic resistance to heat transport (s m-1).
    R_x : float
        Bulk aerodynamic resistance to heat transport at the canopy boundary layer (s m-1).
    R_S : float
        Aerodynamic resistance to heat transport at the soil boundary layer (s m-1).
    f_theta : float
        Fraction of vegetation observed.
    H_C : float
        Sensible heat flux of the canopy (W m-2).
    rho : float
        Density of air (km m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    Returns
    -------
    T_C : float
        Canopy temperature (K).
    References
    ----------
    Eqs. A5-A13 in [Norman1995]_'''

    T_R_K_4 = Tr_K**4
    # equation A7 from Norman 1995, linear approximation of temperature of the
    # canopy
    T_C_lin = ((T_A_K / R_A + Tr_K / (R_S * (1.0 - f_theta))
                + H_C * R_x / (rho * c_p) * (1.0 / R_A + 1.0 / R_S + 1.0 / R_x))
               / (1.0 / R_A + 1.0 / R_S + f_theta / (R_S * (1.0 - f_theta))))
    # equation A12 from Norman 1995
    T_D = (T_C_lin * (1 + R_S / R_A) - H_C * R_x / (rho * c_p)
           * (1.0 + R_S / R_x + R_S / R_A) - T_A_K * R_S / R_A)
    # equation A11 from Norman 1995
    delta_T_C = ((T_R_K_4 - f_theta * T_C_lin**4 - (1.0 - f_theta) * T_D**4) / \
                 (4.0 * (1.0 - f_theta) * T_D**3 * (1.0 + R_S / R_A) + 4.0 * f_theta * T_C_lin**3))
    # get canopy temperature in Kelvin
    T_C = T_C_lin + delta_T_C
    return np.asarray(T_C)


def calc_T_S(T_R, T_C, f_theta):
    '''Estimates soil temperature from the directional LST.
    Parameters
    ----------
    T_R : float
        Directional Radiometric Temperature (K).
    T_C : float
        Canopy Temperature (K).
    f_theta : float
        Fraction of vegetation observed.
    Returns
    -------
    flag : float
        Error flag if inversion not possible (255).
    T_S: float
        Soil temperature (K).
    References
    ----------
    Eq. 1 in [Norman1995]_'''

    # Convert the input scalars to numpy arrays
    T_R, T_C, f_theta = map(np.asarray, (T_R, T_C, f_theta))
    T_temp = T_R**4 - f_theta * T_C**4
    T_S = np.zeros(T_R.shape)
    flag = np.zeros(T_R.shape)

    # Succesfull inversion
    T_S[T_temp >= 0] = (T_temp[T_temp >= 0] /
                        (1.0 - f_theta[T_temp >= 0]))**0.25

    # Unsuccesfull inversion
    T_S[T_temp < 0] = 1e-6
    flag[T_temp < 0] = 255

    return np.asarray(flag), np.asarray(T_S)


def _check_default_parameter_size(parameter, input_array):

    parameter = np.asarray(parameter)
    if parameter.size == 1:
        parameter = np.ones(input_array.shape) * parameter
        return np.asarray(parameter)
    elif parameter.shape != input_array.shape:
        raise ValueError(
            'dimension mismatch between parameter array and input array with shapes %s and %s' %
            (parameter.shape, input_array.shape))
    else:
        return np.asarray(parameter)
