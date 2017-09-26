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
from .TSEB_utils_usda import compute_G0,compute_resistence,albedo_separation
from .TSEB_utils_usda import compute_Rn,temp_separation,compute_stability

#from cachetools import cached


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
    z = np.tile(300.,np.shape(hc))
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
    print(Tc[500,500])
    Ts = (Tr_K-(fc_q*Tc))/(1-fc_q)
    print(Ts[500,500])
    print(Tr_K[500,500])
    print("lai: %f" % lai[500,500])
    print("hc: %f" % hc[500,500])
    print("leaf: %f" % leaf[500,500])
    print("leafs: %f" % leafs[500,500])
    print("leafc: %f" % leafc[500,500])
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
        H_c = Rn_c-lETc
        
        Tc,Ts,Tac = temp_separation(H_c, fc_q, T_A_K, Tr_K, r_ah, r_x, r_s, r_air,cp)
      
        H_s = r_air*cp*(Ts-Tac)/r_s
        H_c = r_air*cp*(Tc-Tac)/r_x
        H = H_s+H_c
    
        lEs = Rn_s-G0-H_s
        lETc = Rn_c-H_c
      
        H[H==0.]=10.
        r_ah[r_ah ==0.]=10.
        mask_iter  = np.logical_and((H_iter/H) <= 1.05,(H_iter/H) >= 0.95)
        chk_iter = np.sum(mask_iter)/np.size(mask_iter)
        print("mask_iter size: %d" % np.size(mask_iter))
        print("mask_iter sum: %d" % np.sum(mask_iter))
        print("check_iter: %f" % chk_iter)
        fm,fh,fm_h = compute_stability(H, Tr_K, r_air,cp, u_attr, z_u, z_T, hc, d_0, z0m, z0h)
        r_ah, r_s, r_x, u_attr = compute_resistence(u, Ts, Tc, hc, lai, d_0, z0m, z0h, z_u, z_T, leaf_width, leaf, leafs, leafc, fm, fh, fm_h)
    
        a_PT[lEs<=0.] = a_PT[lEs<=0.]-0.05
        a_PT[a_PT <= 0.] =  0.01
      
        H_iter = H
        den_s = Rn_s-G0
        den_s[den_s==0.] = np.nan        
        EF_s = lEs/den_s
      
    #      ENDFOR ; ii (Loop for Stability Correction and Water Stress)
    
    #************************************************************************
    # Check Energy Balance Closure
    ind = [a_PT <= 0.01]
    lEs[ind]=1.
    lETc[ind]=1.
    G0[ind] = Rn_s[ind]-H_s[ind]
    ind = lEs > Rn_s
    lEs[ind] = Rn_s[ind]
    H_s[ind] = Rn_s[ind]-G0[ind]-lEs[ind]
    ind = lETc > Rn_c+100.
    lETc[ind] = Rn_c[ind]+100.
    H_c[ind] = -100.
    
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

