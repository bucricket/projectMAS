#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:32:10 2017

@author: mschull
"""
import numpy as np
import math
#   script imports
#imports


def to_jd(datetime):
    """
    Converts a given datetime object to Julian date.
    Algorithm is copied from https://en.wikipedia.org/wiki/Julian_day
    All variable names are consistent with the notation on the wiki page.
    Parameters
    ----------
    fmt
    dt: datetime
        Datetime object to convert to MJD
    Returns
    -------
    jd: float
    """
    dt = datetime
    a = math.floor((14.-dt.month)/12.)
    y = dt.year + 4800. - a
    m = dt.month + 12.*a - 3.

    jdn = dt.day + math.floor((153.*m + 2.)/5.) + 365.*y + math.floor(y/4.) - math.floor(y/100.) + math.floor(y/400.) - 32045.

    jd = jdn + (dt.hour - 12.) / 24. + dt.minute / 1440. + dt.second / 86400. + dt.microsecond / 86400000000.

    return jd


#;
#;  PROCEDURE:  SUNSET_SUNRISE
#;
#;  CALLED BY:  DISALEXI (found at end of file)
#;
#;  PURPOSE:
#;  Computes solar time variables following Campbell & Norman 1998
#;
#;======================================================================================================
#PRO sunset_sunrise, julian, lon, lat, time_t
#
#  COMMON com_time, t_rise, t_end, zs

def sunset_sunrise(dt,lon,lat,time_t):
    julian = to_jd(dt)
    # Sunrise time
    julian_ = julian+(time_t/24.)
    j_cen = ((julian_+0.5-2451545.)/36525.)
    lon_sun = (280.46646+j_cen*(36000.76983+j_cen*0.0003032) % 360.)-360.
    an_sun = 357.52911+j_cen*(35999.05029 - 0.0001537*j_cen)
    ecc = 0.016708634-j_cen*(0.000042037+0.0000001267*j_cen)
    ob_ecl = 23.+(26.+((21.448-j_cen*(46.815+j_cen*(0.00059-j_cen*0.001813))))/60.)/60.
    ob_corr = ob_ecl+0.00256*np.cos(np.deg2rad(125.04-1934.136*j_cen))
    var_y = np.tan(np.deg2rad(ob_corr/2.))*np.tan(np.deg2rad(ob_corr/2.))
    eq_t = 4.*np.rad2deg(var_y*np.sin(np.deg2rad(2.*lon_sun))-2.*ecc*np.sin(np.deg2rad(an_sun))
    +4.*ecc*var_y*np.sin(np.deg2rad(an_sun))*np.cos(np.deg2rad(2.*lon_sun))-0.5*var_y*
    var_y*np.sin(np.deg2rad(4.*lon_sun))-1.25*ecc*ecc*np.sin(np.deg2rad(2*an_sun)))
    
    sun_eq = np.sin(np.deg2rad(an_sun))*(1.914602-j_cen*(0.004817+0.000014*j_cen))+\
        np.sin(np.deg2rad(2.*an_sun))*(0.019993-0.000101*j_cen)+np.sin(np.deg2rad(3.*an_sun))*0.000289
    sun_true = sun_eq+lon_sun
    sun_app = sun_true-0.00569-0.00478*np.sin(np.deg2rad((125.04-1934.136*j_cen)))
    d = np.rad2deg((np.arcsin(np.sin(np.deg2rad(ob_corr))*np.sin(np.deg2rad(sun_app)))))
    ha_t = np.rad2deg(np.arccos(np.cos(np.deg2rad(90.833))/(np.cos(lat)*np.cos(np.deg2rad(d)))-np.tan(lat)*np.tan(np.deg2rad(d))))


    t_noon = (720.-4.*np.rad2deg(lon)-eq_t)/1440.*24.
    t_rise = ((t_noon/24.)-(ha_t*4./1440.))*24.
    t_end = ((t_noon/24.)+(ha_t*4./1440.))*24.

    ts_time = ((time_t/24.*1440+eq_t+4.*np.rad2deg(lon)) % 1440.)
    ts_time[ts_time > 1440.] = ts_time[ts_time > 1440.]-1440.
    w = ts_time/4.+180.
    w[ts_time/4. >= 0] = ts_time[ts_time/4. >= 0.]/4.-180.
 
    zs = np.arccos((np.sin(lat)*np.sin(np.deg2rad(d)))+(np.cos(lat)*np.cos(np.deg2rad(d))*np.cos(np.deg2rad(w))))
  
    return t_rise, t_end, zs

#PRO albedo_separation, albedo, Rs_1, F, fc, aleafv, aleafn, aleafl, adeadv, adeadn, adeadl, z, t_air, zs, control
#
#  COMMON com_alb, Rs_c, Rs_s, albedo_c, albedo_s, e_atm, rsoilv_itr, fg_itr
#  
#  ;*******************************************************************************************************************
#  ; Compute Solar Components and atmospheric properties (Campbell & Norman 1998)
def albedo_separation(albedo, Rs_1, F, fc, aleafv, aleafn, aleafl, adeadv, adeadn, adeadl, z, t_air, zs, control): 
#    ; Compute Solar Components and atmospheric properties (Campbell & Norman 1998)
    #DAYTIME
    #Calculate potential (clear-sky) VIS and NIR solar components
  
    
    airmas = (np.sqrt(np.cos(zs)**2+.0025)-np.cos(zs))/.00125             #Correct for curvature of atmos in airmas
    zs_temp = zs.copy()
    zs_temp[np.rad2deg(zs)>=89.5] = np.deg2rad(89.5)
    ind = np.rad2deg(zs) <89.5 
    airmas[ind] = (airmas[ind]-2.8/(90.-np.rad2deg(zs_temp[ind]))**2.)  #Correct for refraction(good up to 89.5 deg.)

    potbm1 = 600.*np.exp(-.160*airmas)
    potvis = (potbm1+(600.-potbm1)*.4)*np.cos(zs)
    potdif = (600.-potbm1)*.4*np.cos(zs)
    uu = 1.0/np.cos(zs)
    uu[uu <= 0.01] = 0.01
    axlog = np.log10(uu)
    a = 10**(-1.195+.4459*axlog-.0345*axlog*axlog)
    watabs = 1320.*a
    potbm2 = 720.*np.exp(-.05*airmas)-watabs
    evaL = (720.-potbm2-watabs)*.54*np.cos(zs)
    potnir = evaL+potbm2*np.cos(zs)
    fclear = Rs_1/(potvis+potnir)
    fclear[fclear > 1.]=1.
    fclear[np.cos(zs) <= 0.01]=1.
    fclear[fclear <= 0.01]= 0.01
    
    #Partition SDN into VIS and NIR
    fvis = potvis/(potvis+potnir)
    fnir = potnir/(potvis+potnir)
  
    #Estimate direct beam and diffuse fraction in VIS and NIR wavebands
    fb1 = potbm1*np.cos(zs)/potvis
    fb2 = potbm2*np.cos(zs)/potnir

    ratiox = fclear.copy()
    ratiox[fclear > 0.9] = 0.9
    dirvis = fb1*(1.-((.9-ratiox)/.7)**.6667)
    ind = dirvis >= fb1
    dirvis[ind]=fb1[ind]
    
    ratiox = fclear.copy()
    ratiox[fclear > 0.88] = 0.88
    dirnir = fb1*(1.-((.88-ratiox)/.68)**.6667)
    ind = dirnir >= fb2
    dirnir[ind]=fb1[ind]
    
    ind = np.logical_and((dirvis < 0.01),(dirnir > 0.01))
    dirvis[ind] = 0.011
    ind  = np.logical_and((dirnir < 0.01),(dirvis > 0.01))
    dirnir[ind] = 0.011

    
    difvis = 1.-dirvis
    difnir = 1.-dirnir

    #Correction for NIGHTIME
    ind = np.cos(zs) <= 0.01
    fvis[ind] = 0.5
    fnir[ind] = 0.5
    difvis[ind] = 1.
    difnir[ind] = 1.
    dirvis[ind] = 0.
    dirnir[ind] = 0.
  
    Rs0 = potvis+potnir
    Rs0[ind] = 0.

    #apparent emissivity (Sedlar and Hock, 2009: Cryosphere 3:75-84)
    e_atm = 1.-(0.2811*(np.exp(-0.0003523*((t_air-273.16)**2.))))              #atmospheric emissivity (clear-sly) Idso and Jackson (1969)
    fclear[Rs0 <= 50.] = 1.

    #**********************************************
    # Compute Albedo
    ratio_soil = 2.
    if control ==1:
        rsoilv = np.tile(0.12, np.shape(F))
        fg = np.tile(1.,np.shape(albedo))
        z_inter = 9
#    else:
#        rsoilv = rsoilv_itr
#        fg = fg_itr
#        z_inter = 0.

    for zzz in range(z_inter+1): # +1 to do what IDL does 
  
        rsoiln = rsoilv*ratio_soil
        
        #Weighted live/dead leaf average properties
        ameanv = aleafv*fg + adeadv*(1.-fg)
        ameann = aleafn*fg + adeadn*(1.-fg)
        ameanl = aleafl*fg + adeadl*(1.-fg)
        
        #DIFFUSE COMPONENT
        #*******************************
        #canopy reflection (deep canopy)
        akd = -0.0683*np.log(F)+0.804                       #Fit to Fig 15.4 for x=1
        rcpyn = (1.0-np.sqrt(ameann))/(1.0+np.sqrt(ameann))     #Eq 15.7
        rcpyv = (1.0-np.sqrt(ameanv))/(1.0+np.sqrt(ameanv))
        rcpyl = (1.0-np.sqrt(ameanl))/(1.0+np.sqrt(ameanl))
        rdcpyn = 2.0*akd*rcpyn/(akd+1.0)                  #Eq 15.8
        rdcpyv = 2.0*akd*rcpyv/(akd+1.0)
        rdcpyl = 2.0*akd*rcpyl/(akd+1.0)
        
        #canopy transmission (VIS)
        expfac = np.sqrt(ameanv)*akd*F
        expfac[expfac < 0.001] = 0.001

        xnum = (rdcpyv*rdcpyv-1.0)*np.exp(-expfac)
        xden = (rdcpyv*rsoilv-1.0)+rdcpyv*(rdcpyv-rsoilv)*np.exp(-2.0*expfac)
        taudv = xnum/xden         #Eq 15.11
        
        #canopy transmission (NIR)
        expfac = np.sqrt(ameann)*akd*F
        expfac[expfac < 0.001] = 0.001
        xnum = (rdcpyn*rdcpyn-1.0)*np.exp(-expfac)
        xden = (rdcpyn*rsoiln-1.0)+rdcpyn*(rdcpyn-rsoiln)*np.exp(-2.0*expfac)
        taudn = xnum/xden         #Eq 15.11
        
        #canopy transmission (LW)
        taudl = np.exp(-np.sqrt(ameanl)*akd*F)
        
        #diffuse albedo for generic canopy
        fact = ((rdcpyn-rsoiln)/(rdcpyn*rsoiln-1.0))*np.exp(-2.0*np.sqrt(ameann)*akd*F)   #Eq 15.9
        albdn = (rdcpyn+fact)/(1.0+rdcpyn*fact)
        fact = ((rdcpyv-rsoilv)/(rdcpyv*rsoilv-1.0))*np.exp(-2.0*np.sqrt(ameanv)*akd*F)   #Eq 15.9
        albdv = (rdcpyv+fact)/(1.0+rdcpyv*fact)

        #BEAM COMPONENT
        #*******************************
        #canopy reflection (deep canopy)
        akb = 0.5/np.cos(zs)
        akb[np.cos(zs) <= 0.01] = 0.5
        rcpyn = (1.0-np.sqrt(ameann))/(1.0+np.sqrt(ameann))     #Eq 15.7
        rcpyv = (1.0-np.sqrt(ameanv))/(1.0+np.sqrt(ameanv))
        rbcpyn = 2.0*akb*rcpyn/(akb+1.0)                  #Eq 15.8
        rbcpyv = 2.0*akb*rcpyv/(akb+1.0)

        #beam albedo for generic canopy
        fact = ((rbcpyn-rsoiln)/(rbcpyn*rsoiln-1.0))*np.exp(-2.0*np.sqrt(ameann)*akb*F)    #Eq 15.9
        albbn = (rbcpyn+fact)/(1.0+rbcpyn*fact)
        fact = ((rbcpyv-rsoilv)/(rbcpyv*rsoilv-1.0))*np.exp(-2.0*np.sqrt(ameanv)*akb*F)    #Eq 15.9
        albbv = (rbcpyv+fact)/(1.0+rbcpyv*fact)
        
        #weighted albedo (canopy)
        albedo_c = fvis*(dirvis*albbv+difvis*albdv)+fnir*(dirnir*albbn+difnir*albdn)
        ind = np.cos(zs) <= 0.01
        albedo_c[ind] = (fvis[ind]*(difvis[ind]*albdv[ind])+fnir[ind]*(difnir[ind]*albdn[ind]))
        albedo_s = fvis*rsoilv+fnir*rsoiln

        albedo_avg = (fc*albedo_c)+((1-fc)*albedo_s)
        diff = albedo_avg-albedo

        ind = np.logical_and((fc< 0.75), (diff <= -0.01))
        rsoilv[ind] = rsoilv[ind]+0.01
        ind = np.logical_and((fc< 0.75),(diff > 0.01))
        rsoilv[ind] = rsoilv[ind]-0.01
        

        ind = np.logical_and((fc>= 0.75), (diff <= -0.01))
        fg[ind] = fg[ind]-0.05
        ind = np.logical_and((fc>= 0.75),(diff > 0.01))
        fg[ind] = fg[ind]+0.05
        
        fg[fg >1.] = 1.
        fg[fg<0.01] = 0.01
        

    if control == 1:
        fg_itr = fg
        rsoilv_itr = rsoilv
    ind = abs(diff) > 0.05
    albedo_c[ind] = albedo[ind]
    albedo_s[ind] = albedo[ind]                                      #if a solution is not reached, alb_c=alb_s=alb

  
    #Direct beam+scattered canopy transmission coeff (visible)
    expfac = np.sqrt(ameanv)*akb*F
    xnum = (rbcpyv*rbcpyv-1.0)*np.exp(-expfac)
    xden = (rbcpyv*rsoilv-1.0)+rbcpyv*(rbcpyv-rsoilv)*np.exp(-2.0*expfac)
    taubtv = xnum/xden        #Eq 15.11
  
    #Direct beam+scattered canopy transmission coeff (NIR)
    expfac = np.sqrt(ameann)*akb*F
    xnum = (rbcpyn*rbcpyn-1.0)*np.exp(-expfac)
    xden = (rbcpyn*rsoiln-1.0)+rbcpyn*(rbcpyn-rsoiln)*np.exp(-2.0*expfac)
    taubtn = xnum/xden        #Eq 15.11
  
    #shortwave radition components
    tausolar = fvis*(difvis*taudv+dirvis*taubtv)+fnir*(difnir*taudn+dirnir*taubtn)
    Rs_c = Rs_1*(1.-tausolar)
    Rs_s = Rs_1*tausolar
    
    return Rs_c, Rs_s, albedo_c, albedo_s, e_atm, rsoilv_itr, fg_itr


    
def compute_G0(Rn,Rn_s,albedo,ndvi,t_rise,t_end,time,EF_s):
    w = 1/(1+(EF_s/0.5)**8.)
    c_g = (w*0.35)+((1-w)*0.31)       #maximum fraction of Rn,s that become G0 (0.35 for dry soil and 0.31 for wet soil)
    t_g = (w*100000.)+((1-w)*74000.)
      
    tnoon=0.5*(t_rise+t_end)
    t_g0=(time-tnoon)*3600.
      
    G0=c_g*np.cos(2*np.pi*(t_g0+10800.)/t_g)*Rn_s
    ind = np.logical_and(ndvi<=0, albedo <=0.05)
    G0[ind]=Rn[ind]*0.5  

    
    return G0

#PRO compute_resistence, U, Ts, Tc, hc, F, d0, z0m, z0h, z_u, z_T, xl, leaf, leafs, leafc, fm, fh, fm_h
#
#  COMMON com_res, r_ah, r_s, r_x, u_attr
    
def compute_resistence(U, Ts, Tc, hc, F, d0, z0m, z0h, z_u, z_T, xl, leaf, leafs, leafc, fm, fh, fm_h):
    c_a = 0.004         #Free convective velocity constant for r_s modelling
    c_b = 0.012         #Empirical constant for r_s modelling
    c_c = 0.0025        #Empirical constant for r_s modelling (new formulation Kustas and Norman, 1999)
      
    C = 175.            #Parameter for canopy boundary-layer resistance (C=90 Grace '81, C=175 Cheubouni 2001, 144 Li '98)
      
      # Computation of friction velocity and aerodynamic resistance
    u_attr = 0.41*U/((np.log((z_u-d0)/z0m))-fm)
    u_attr[u_attr==0]=10.
    u_attr[u_attr<0]=0.01
    r_ah = ((np.log((z_T-d0)/z0h))-fh)/u_attr/0.41
    r_ah[r_ah==0]=500.
    r_ah[r_ah<=1.]=1.
      
    # Computation of the resistance of the air between soil and canopy space
    Uc = u_attr/0.41*((np.log((hc-d0)/z0m))-fm_h)
    Uc[Uc<=0]=0.1
    Us = Uc*np.exp(-leaf*(1.-(0.05/hc)))

    r_ss = 1./(c_a+(c_b*(Uc*np.exp(-leafs*(1.-(0.05/hc))))))
    r_s1 = 1./((((abs(Ts-Tc))**(1./3.))*c_c)+(c_b*Us))
    r_s2 = 1./(c_a+(c_b*Us))
  
    r_s = (((r_ss-1.)/0.09*(F-0.01))+1.)
    r_s[F>0.1]=r_s1[F>0.1]                                                         #linear fuction between 0(bare soil) anf the value at F=0.1
    r_s[abs(Ts-Tc) <1.]=r_s2[abs(Ts-Tc) <1.]  
    r_s[F >3.]=r_s2[F >3.]    
      
    # Computation of the canopy boundary layer resistance
    Ud = Uc*np.exp(-leafc*(1.-((d0+z0m)/hc)))
    Ud[Ud <= 0.] = 100.
    r_x = C/F*((xl/Ud)**0.5)
    r_x[Ud==100.] = 0.1
     
    return r_ah, r_s, r_x, u_attr

#PRO compute_Rn, albedo_c, albedo_s, t_air, Tc, Ts, e_atm, Rs_c, Rs_s, F
#
#  COMMON com_Rn, Rn_s, Rn_c, Rn
def compute_Rn(albedo_c, albedo_s, t_air, Tc, Ts, e_atm, Rs_c, Rs_s, F):
    kL=0.95             #long-wave extinction coefficient [-]
    eps_s = 0.94        #Soil Emissivity [-]
    eps_c = 0.99        #Canopy emissivity [-]
      
    Lc = eps_c*0.0000000567*(Tc**4.)
    Ls = eps_s*0.0000000567*(Ts**4.)
    Rle = e_atm*0.0000000567*(t_air**4.)
    Rn_c = ((1.-albedo_c)*Rs_c)+((1.-np.exp(-kL*F))*(Rle+Ls-2.*Lc))
    Rn_s = ((1.-albedo_s)*Rs_s)+((np.exp(-kL*F))*Rle)+((1.-np.exp(-kL*F))*Lc)-Ls
    Rn = Rn_s+Rn_c

    
    return Rn_s, Rn_c, Rn  

#PRO temp_separation, H_c, fc, t_air, t0, r_ah, r_x, r_s, r_air
#
#  COMMON com_sep, Tc, Ts, Tac

def temp_separation(H_c, fc, t_air, t0, r_ah, r_x, r_s, r_air,cp):
    
    Tc_lin = ((t_air/r_ah)+(t0/r_s/(1.-fc))+(H_c*r_x/r_air/cp*((1./r_ah)+(1./r_s)+(1./r_x))))/((1./r_ah)+(1./r_s)+(fc/r_s/(1.-fc)))

    Td = (Tc_lin*(1+(r_s/r_ah)))-(H_c*r_x/r_air/cp*(1.+(r_s/r_x)+(r_s/r_ah)))-(t_air*r_s/r_ah)

    delta_Tc = ((t0**4.)-(fc*(Tc_lin**4.))-((1.-fc)*(Td**4.)))/((4.*(1.-fc)*(Td**3.)*(1.+(r_s/r_ah)))+(4.*fc*(Tc_lin**3.)))

    Tc = (Tc_lin+delta_Tc)  
    Tc[fc < 0.10]=t0[fc < 0.10]
    Tc[fc >0.90]=t0[fc >0.90]
#======get Ts==================================================================  
    Delta = (t0**4.)-(fc*(Tc**4.))
    Delta[Delta<=0.]=10.
    
    Ts= (Delta/(1-fc))**0.25
    ind = ((t0**4)-(fc*Tc**4.))<=0.
    Ts[ind]=(t0[ind]-(fc[ind]*Tc[ind]))/(1-fc[ind])
    
    Ts[fc < 0.1] = t0[fc < 0.1]
    Ts[fc > 0.9] = t0[fc > 0.9]


    ind = (Tc <= (t_air-10.))
    Tc[ind] = (t_air[ind]-10.)
    ind = (Tc >= t_air+50.)
    Tc[ind] = (t_air[ind]+50.)
    ind = (Ts <= (t_air-10.))
    Ts[ind] = (t_air[ind]-10.)
    ind = (Ts >= t_air+50.)
    Ts[ind] = (t_air[ind]+50.)

  
    Tac = ((((t_air)/r_ah)+((Ts)/r_s)+((Tc)/r_x))/((1/r_ah)+(1/r_s)+(1/r_x)))
    return Tc, Ts, Tac


#PRO compute_stability, H, t0, r_air, u_attr, z_u, z_T, hc, d0, z0m, z0h
#
#  COMMON com_stab, fm, fh, fm_h
def compute_stability(H, t0, r_air,cp, u_attr, z_u, z_T, hc, d0, z0m, z0h):
    t0[t0 == 0.] = 100.
    L_ob = -(r_air*cp*t0*(u_attr**3.0)/0.41/9.806/H)
    L_ob[L_ob>=0.] = -99.
  
    mm = ((1.-(16.*(z_u-d0)/L_ob))**0.25)
    mm_h = ((1.-(16.*(hc-d0)/L_ob))**0.25)
    mh = ((1.-(16.*(z_T-d0)/L_ob))**0.25)
    ind = L_ob==-99.
    mm[ind] = 0.
    mm_h[ind] = 0.
    mh[ind] = 0.
    

    fm = np.zeros(mh.shape)
    ind = np.logical_and((L_ob < 100.),(L_ob > (-100.)))
    fm[ind] = ((2.0*np.log((1.0+mm[ind])/2.0))+(np.log((1.0+(mm[ind]**2.))/2.0))-(2.0*np.arctan(mm[ind]))+(np.pi/2.))

    fm_h = np.zeros(mh.shape)
    fm_h[ind] = ((2.0*np.log((1.0+mm_h[ind])/2.0))+(np.log((1.0+(mm_h[ind]**2.))/2.0))-(2.0*np.arctan(mm_h[ind]))+(np.pi/2.))

    fh = np.zeros(mh.shape)
    fh[ind] = ((2.0*np.log((1.0+(mh[ind]**2.))/2.0)))
    ind = (fm == (np.log((z_u-d0)/z0m)))
    fm[ind]=fm[ind]+1.
    ind = (fm_h == (np.log((hc-d0)/z0m)))
    fm_h[ind]= fm_h[ind]+1.

    return fm, fh, fm_h