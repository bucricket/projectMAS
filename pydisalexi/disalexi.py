#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 08:46:02 2017

@author: mschull
"""

# This file is part of PyDisALEXI for running disALEXI using different TSEB models
# Copyright 2016 Mitchell Schull and contributors listed in the README.md file.
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
import os
import numpy as np
import subprocess
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from .TSEB import TSEB_PT
from .net_radiation import calc_difuse_ratio,calc_Sn_Campbell
from .clumping_index import calc_omega_Kustas,calc_omega0_Kustas
from .utils import writeArray2Tiff,getParFromExcel,findRSOILV,warp,folders
from scipy import ndimage
from pyhdf.SD import SD, SDC
from .landsatTools import landsat_metadata,GeoTIFF



class disALEXI(object):
    def __init__(self, fn,isUSA):
        base = os.path.abspath(os.path.join(fn,os.pardir,os.pardir,os.pardir,
                                            os.pardir,os.pardir))

        Folders = folders(base)  
        self.landsatSR = Folders['landsatSR']
        self.landsatNDVI = Folders['landsatNDVI']
        self.ALEXIbase = Folders['ALEXIbase']
        self.metBase = Folders['metBase']
        self.landsatDataBase = Folders['landsatDataBase']
        self.resultsBase = Folders['resultsBase']
        self.fn = fn
        self.meta = landsat_metadata(fn)
        self.sceneID = self.meta.LANDSAT_SCENE_ID
        self.productID = fn.split(os.sep)[-1][:-8]
#        self.productID = self.meta.LANDSAT_PRODUCT_ID
        self.scene = self.sceneID[3:9]
        self.isUSA = isUSA





    '''
    Created on Sept. 8, 2016
    @author: Mitchell Schull (mitch.schull@noaa.gov)
    
    
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
    * :func:`TSEB_2T` TSEB using derived/measured canopy and soil component temperatures.
    * :func:`TSEB_PT` Priestley-Taylor TSEB using a single observation of composite radiometric temperature.
    * :func:`DTD` Dual-Time Differenced TSEB using composite radiometric temperatures at two times: early morning and near afternoon.
    
    OSEB models
    -----------
    * :func:`OSEB`. One Source Energy Balance Model.
    * :func:`OSEB_below_canopy`. One Source Energy Balance Model of baresoil/understory beneath a canopy.
    * :func:`OSEB_canopy`. One Source Energy Balance Model of a very dense canopy, i.e. `Big-leaf` model.
    
    Ancillary functions
    -------------------
    * :func:`calc_F_theta_campbell`. Gap fraction estimation.
    * :func:`calc_G_time_diff`. Santanello & Friedl (2003) [Santanello2003]_ soil heat flux model.
    * :func:`calc_G_ratio`. Soil heat flux as a fixed fraction of net radiation [Choudhury1987]_.
    * :func:`calc_H_C`. canopy sensible heat flux in a parallel resistance network.
    * :func:`calc_H_C_PT`. Priestley- Taylor Canopy sensible heat flux.
    * :func:`calc_H_DTD_parallel`. Priestley- Taylor Canopy sensible heat flux for DTD and resistances in parallel.
    * :func:`calc_H_DTD_series`. Priestley- Taylor Canopy sensible heat flux for DTD and resistances in series.
    * :func:`calc_H_S`. Soil heat flux with resistances in parallel.
    * :func:`calc_T_C`. Canopy temperature form composite radiometric temperature.
    * :func:`calc_T_C_series.` Canopy temperature from canopy sensible heat flux and resistance in series.
    * :func:`calc_T_CS_Norman`. Component temperatures from dual angle composite radiometric tempertures.
    * :func:`calc_T_CS_4SAIL`. Component temperatures from dual angle composite radiometric tempertures. Using 4SAIl for the inversion.
    * :func:`calc_4SAIL_emission_param`. Effective surface reflectance, and emissivities for soil and canopy using 4SAIL.
    * :func:`calc_T_S`. Soil temperature from form composite radiometric temperature.
    * :func:`calc_T_S_series`. Soil temperature from soil sensible heat flux and resistance in series.
    '''
       
    def DisALEXI_PT(self,
        ET_ALEXI,
        Rs_1,
        Rs24in,
        Tr_K,
        vza,
        u,
        ea,
        p,
        Sn_C,
        Sn_S,
        LAI,
        hc,
        emis_C,
        emis_S,
        z_0M,
        d_0,
        z_u,
        z_T,
        nullMask,
        leaf_width,
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
        '''DisALEXI based on Priestley-Taylor TSEB
    
        Calculates the Priestley Taylor TSEB fluxes using a single observation of
        composite radiometric temperature and using resistances in series.
    
        Parameters
        ----------
        ET_ALEXI : float
            Coarse resolution daily ET from ALEXI
        geoDict : dictionary
            Dictionary containing:
            inProj4 : proj4 string
                ALEXI ET proj4 string
            outProj4 : proj4 string
                DisALEXI ET proj4 string
            inUL : float array
                Upper left lat/lon coordinates of ALEXI image
            inRes : float array
                ALEXI ET lat/lon resolution
        Rs_1 : float
            Overpass insolation (w m-2)
        Rs24 : float
            Total daily insolation (w m-2)
        Tr_K : float
            Radiometric composite temperature (Kelvin).
        vza : float
            View Zenith Angle (degrees).
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

        # Set up input parameters
        MatXsize = 7
        nullMask_resize = np.tile(np.array(np.resize(nullMask,[np.size(nullMask),1])),(1,MatXsize))
        Tr_Kresize = np.tile(np.array(np.resize(Tr_K,[np.size(Tr_K),1])),(1,MatXsize))
        vzaresize = np.tile(np.resize(vza,[np.size(vza),1]),(1,MatXsize))
        T_A_Kresize = np.tile(range(270,340,10),(np.size(vza),1))
        uresize = np.tile(np.resize(u,[np.size(u),1]),(1,MatXsize))
        earesize = np.tile(np.resize(ea,[np.size(ea),1]),(1,MatXsize))
        presize = np.tile(np.resize(p,[np.size(p),1]),(1,MatXsize))
        Sn_Cresize = np.tile(np.resize(Sn_C,[np.size(Sn_C),1]),(1,MatXsize))
        Sn_Sresize = np.tile(np.resize(Sn_S,[np.size(Sn_S),1]),(1,MatXsize))
        e_atm = 1.0-(0.2811*(np.exp(-0.0003523*((T_A_Kresize-273.16)**2))))                             #atmospheric emissivity (clear-sly) Idso and Jackson (1969)
        L_dn = e_atm*0.0000000567*((T_A_Kresize)**4)
        LAIresize = np.tile(np.resize(LAI,[np.size(LAI),1]),(1,MatXsize))
        hcresize = np.tile(np.resize(hc,[np.size(hc),1]),(1,MatXsize))
        emis_Cresize = np.tile(np.resize(emis_C,[np.size(hc),1]),(1,MatXsize))
        emis_Sresize = np.tile(np.resize(emis_S,[np.size(hc),1]),(1,MatXsize))
        z_0Mresize = np.tile(np.resize(z_0M,[np.size(hc),1]),(1,MatXsize))
        d_0resize = np.tile(np.resize(d_0,[np.size(hc),1]),(1,MatXsize))
        z_uresize = np.tile(np.resize(z_u,[np.size(hc),1]),(1,MatXsize))
        z_Tresize = np.tile(np.resize(z_T,[np.size(hc),1]),(1,MatXsize))
        leaf_widthresize = np.tile(np.resize(leaf_width,[np.size(hc),1]),(1,MatXsize))
        z0_soilresize = np.tile(np.resize(z0_soil,[np.size(hc),1]),(1,MatXsize))
        alpha_PTresize = np.tile(np.resize(alpha_PT,[np.size(hc),1]),(1,MatXsize))
        f_cresize = np.tile(np.resize(f_c,[np.size(hc),1]),(1,MatXsize))
        f_gresize = np.tile(np.resize(f_g,[np.size(hc),1]),(1,MatXsize))
        w_Cresize = np.tile(np.resize(w_C,[np.size(hc),1]),(1,MatXsize))
        # run TSEB over TA options
        output = TSEB_PT(
            Tr_Kresize,
            vzaresize,
            T_A_Kresize,
            uresize,
            earesize,
            presize,
            Sn_Cresize,
            Sn_Sresize,
            L_dn,
            LAIresize,
            hcresize,
            emis_Cresize,
            emis_Sresize,
            z_0Mresize,
            d_0resize,
            z_uresize,
            z_Tresize,
            nullMask_resize,
            leaf_width=leaf_widthresize,
            z0_soil=z0_soilresize,
            alpha_PT=alpha_PTresize,
            x_LAD=1,
            f_c=f_cresize,
            f_g=f_gresize,
            w_C=w_Cresize,
            resistance_form=0,
            calcG_array=[
                [1],
                0.35],
                UseL=False)
            
        scaling = 1.0
        Fsun =  (output[6]+output[8])/np.resize(Rs_1,[np.size(hc),1])
        EFeq=Fsun*(np.reshape(Rs24in,[np.size(hc),1]))
        et = EFeq/2.45*scaling
        
        # ======interpolate over mutiple Ta solutions===========================================

        from scipy.interpolate import interp1d
        x = range(270,340,10)
        et_alexi = np.reshape(ET_ALEXI,[np.size(hc),1])
        bias = et_alexi-et
        # check if all values inrow are nan
        nanIndex = np.sum(np.isnan(bias),axis=1)
        # set all to 1 so it doesnt throw an error below
        bias[np.where(nanIndex==7),:]=1.
        f_bias = interp1d(x,bias,kind='linear', bounds_error=False)
        f_ta= interp1d(x,T_A_Kresize,kind='linear', bounds_error=False)
        biasInterp = f_bias(range(270,340))
        TaInterp = f_ta(range(270,340))
        # extract the Ta based on minimum bias at Fine resolution
        minBiasIndex = np.array(np.nanargmin(abs(biasInterp),axis=1))
        TaExtrap = TaInterp[np.array(range(np.size(hc))),minBiasIndex]
        TaExtrap[np.where(nanIndex==7)]=np.nan
        Tareshape = np.reshape(TaExtrap,np.shape(hc))
        
        #============Run one last time with the final smoothed T_A_K
        T_A_K = Tareshape
        output ={'T_A_K':T_A_K}
        return output
    
    def runDisALEXI(self,xStart,yStart,ALEXIgeodict,TSEB_only):
        # USER INPUT============================================================
        ALEXILatRes = ALEXIgeodict['ALEXI_LatRes']
        ALEXILonRes = ALEXIgeodict['ALEXI_LonRes']
        sceneID =self.sceneID
        scene = self.scene
        productID = self.productID
        xSize = 200
        ySize = 200
            
        #-------------pick Landcover map----------------
        if self.isUSA ==1:
            landcover = 'NLCD'
        else:
            landcover = 'GlobeLand30'
        
        yeardoy = sceneID[9:16]

        #-------------get Landsat information-----------

        ls = GeoTIFF(os.path.join(self.landsatSR,'temp','%s_lstSharp.tiff' % productID))
        solZen = self.meta.SUN_ELEVATION
#        nsamples = int(self.meta.REFLECTIVE_SAMPLES)
#        nlines = int(self.meta.REFLECTIVE_LINES)
        nsamples = ls.nrow
        nlines = ls.ncol
        if xStart==((nsamples/200)*200):
            xSize = (nsamples-xStart)-1
        if yStart==((nlines/200)*200):
            ySize = (nlines-yStart)-1
        inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        sz = np.radians(solZen) # convert sza to radians
    
    
        #===========================get the ETd data==============================
    
        sceneDir = os.path.join(self.ALEXIbase,'%s' % scene)        
        outFN = os.path.join(sceneDir,'%s_alexiETSub.tiff' % sceneID)         
        g = gdal.Open(outFN,GA_ReadOnly)
        ET_ALEXI = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
           
        
        #=============get MET data================================================
        #get CFSR MET data at overpass time
    
        sceneDir = os.path.join(self.metBase,'%s' % scene)
        #------------get-> surface pressure...
        outFN = os.path.join(sceneDir,'%s_pSub.tiff' % sceneID) 
        g = gdal.Open(outFN,GA_ReadOnly)
        p = g.ReadAsArray(xStart,yStart,xSize,ySize)
        p /=100. #convert to mb
        g= None
            
        #------------get-> ea...
        outFN = os.path.join(sceneDir,'%s_q2Sub.tiff' % sceneID) 
        g = gdal.Open(outFN,GA_ReadOnly)
        q2 = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
    
        ea = ((q2*(1000./621.9907))*(p*100.))*0.001                             #kPa
        ea *= 10. #mb
        
        if TSEB_only==1:

            #ls = GeoTIFF(os.path.join(self.landsatSR, scene,'%s_sr_band1.tif' % productID))

            #=======================convert fine TA to coarse resolution=========
            outfile = os.path.join(self.resultsBase,scene,'Taxxxxx.tif')
    
            coarseFile = os.path.join(self.resultsBase,scene,'TaCoarse.tif')
            coarse2fineFile = os.path.join(self.resultsBase,scene,'TaCoarse2Fine.tif')
            outFN = coarseFile[:-10]+'.tif'
            if not os.path.exists(outFN):
                print 'get->Ta'
                # get mask from Landsat LAI
                ls = GeoTIFF(outfile)
#                laiFN = os.path.join(self.landsatDataBase,'LAI',scene,'lndlai.%s.hdf' % sceneID)
#                hdf = SD(laiFN,SDC.READ)
#                data2D = hdf.select('cfmask')
#                cfmask = data2D[:,:].astype(np.double)
                
                maskFN = os.path.join(self.landsatDataBase,'Mask',scene,'%s_Mask.tiff' % sceneID)
                g = gdal.Open(maskFN,GA_ReadOnly)
                cfmask = g.ReadAsArray(xStart,yStart,xSize,ySize)
                g= None
                g = gdal.Open(outfile,GA_ReadOnly)
                ta = g.ReadAsArray()
                ta[cfmask > 0]=0
                mask = os.path.join(self.resultsBase,scene,"TafineMask.tif")
                masked = os.path.join(self.resultsBase,scene,"TafineMasked.tif")
                ls.clone(mask,ta)
                subprocess.check_output('gdal_fillnodata.py %s %s -mask %s -of GTiff' % (outfile,masked,mask),shell=True)
                
                
                optionList = ['-overwrite', '-s_srs', '%s' % ls.proj4,'-t_srs',
                              '%s' % inProj4,'-r', 'average','-tr', 
                              '%f' % ALEXILatRes, '%f' % ALEXILonRes,
                              '-srcnodata','270.','-dstnodata','0.0',
                              '-of','GTiff','%s' % masked, '%s' % coarseFile]
                
                warp(optionList)
                
#                #========fill in missing data from VIIRS and Landsat data======
#                sceneDir = os.path.join(self.ALEXIbase,'%s' % scene)
#                etFN = os.path.join(sceneDir,'%s_alexiETSub.tiff' % sceneID) 
#                g = gdal.Open(etFN,GA_ReadOnly)
#                et= g.ReadAsArray()
#                et[et==-9999]=0
#                ls = GeoTIFF(etFN)
#                mask = os.path.join(self.resultsBase,scene,"TaMask.tif")
#                masked = os.path.join(self.resultsBase,scene,"TaMasked.tif")
#                ls.clone(mask,et)
#                subprocess.check_output('gdal_fillnodata.py %s %s -mask %s -of GTiff' % (coarseFile,masked,mask),shell=True)
                #os.remove(outfile)
                #=======now convert the averaged coarse Ta to fine resolution==
#                nrow = int(self.meta.REFLECTIVE_SAMPLES)+100
#                ncol = int(self.meta.REFLECTIVE_LINES)+100
                nrow = ls.nrow+100
                ncol = ls.ncol+100
                optionList = ['-overwrite', '-s_srs', '%s' % inProj4, '-t_srs', 
                              '%s' % ls.proj4,'-r', 'bilinear','-ts', 
                              '%f' % nrow, '%f' % ncol,'-of',
                              'GTiff','%s' % coarseFile, '%s' % coarse2fineFile]
                
                warp(optionList)
                #======now subset to match original image
#                ulx = self.meta.CORNER_UL_PROJECTION_X_PRODUCT
#                uly = self.meta.CORNER_UL_PROJECTION_Y_PRODUCT
#                lrx = self.meta.CORNER_LR_PROJECTION_X_PRODUCT
#                lry = self.meta.CORNER_LR_PROJECTION_Y_PRODUCT
#                delx = self.meta.GRID_CELL_SIZE_REFLECTIVE
#                dely = self.meta.GRID_CELL_SIZE_REFLECTIVE
                ulx = ls.ulx
                uly = ls.uly
                lrx = ls.lrx
                lry = ls.lry
                delx = ls.delx
                dely = -ls.dely
                optionList = ['-overwrite','-te', '%f' % ulx, '%f' % lry,
                              '%f' % lrx,'%f' % uly,'-tr',
                              '%f' % delx, '%f' % dely ,'-multi','-of','GTiff',
                              '%s' % coarse2fineFile, '%s' % outFN]
                warp(optionList)
                #os.remove(coarseFile)
            g = gdal.Open(outFN,GA_ReadOnly)
            T_A_K = g.ReadAsArray(xStart,yStart,xSize,ySize)
            g= None
        

        sceneDir = os.path.join(self.metBase,'%s' % scene)
        outFN = os.path.join(sceneDir,'%s_uSub.tiff' % sceneID) 
        g = gdal.Open(outFN,GA_ReadOnly)
        u = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
        
        #====get overpass hour insolation======================================
        outFN = os.path.join(sceneDir,'%s_Insol1Sub.tiff' % sceneID)            
        g = gdal.Open(outFN,GA_ReadOnly)
        Rs_1 = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
    
        #====get daily insolation=========================================
        outFN = os.path.join(sceneDir,'%s_Insol24Sub.tiff' % sceneID)
        g = gdal.Open(outFN,GA_ReadOnly)
        Rs24 = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
    
        #===============get biophysical parameters at overpass time============
        sceneDir = os.path.join(self.landsatDataBase,'albedo',scene)
        outFN = os.path.join(sceneDir,'%s_albedo.tiff' % sceneID)         
        g = gdal.Open(outFN,GA_ReadOnly)
        albedo = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
    
        #------>get LAI...
#        outFN = os.path.join(self.landsatDataBase,'LAI',scene,'lndlai.%s.hdf' % sceneID)
        outFN = os.path.join(self.landsatDataBase,'LAI',scene,'%s_lai.tiff' % sceneID)
        g = gdal.Open(outFN,GA_ReadOnly)
        LAI = g.ReadAsArray(xStart,yStart,xSize,ySize)*0.001
        g= None
        LAI[np.where(LAI==-9.999)]=np.nan
        LAI[np.where(LAI<=0.)]=0.001
        
#        hdf = SD(outFN,SDC.READ)
#        data2D = hdf.select('LAI')
#        LAI = data2D[yStart:yStart+ySize,xStart:xStart+xSize].astype(np.double)*0.001
#    
#        LAI[np.where(LAI==-9.999)]=np.nan
#        LAI[np.where(LAI<=0.)]=0.001
        
        #------>get ndvi...'
        sceneDir = os.path.join(self.landsatNDVI,scene)
        outFN = os.path.join(sceneDir,'%s_ndvi.tiff' % sceneID)
        g = gdal.Open(outFN,GA_ReadOnly)
        ndvi = g.ReadAsArray(xStart,yStart,xSize,ySize)*0.001
        g= None
        ndvi[np.where(ndvi==-9.999)]=np.nan
        
#        data2D = hdf.select('NDVI')
#        ndvi = data2D[yStart:yStart+ySize,xStart:xStart+xSize].astype(np.double)*0.001
#        ndvi[np.where(ndvi==-9.999)]=np.nan
        
        #===get cfmask=======
        outFN = os.path.join(self.landsatDataBase,'Mask',scene,'%s_Mask.tiff' % sceneID)
        g = gdal.Open(outFN,GA_ReadOnly)
        cfmask = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
#        laiFN = os.path.join(self.landsatDataBase,'LAI',scene,'lndlai.%s.hdf' % sceneID)
#        hdf = SD(laiFN,SDC.READ)
#        data2D = hdf.select('cfmask')
#        cfmask = data2D[yStart:yStart+ySize,xStart:xStart+xSize].astype(np.double)
    
        
        #---------->get LST...   
        outFN = os.path.join(self.landsatDataBase,'LST',scene,'%s_lstSharp.tiff' % sceneID)
        g = gdal.Open(outFN,GA_ReadOnly)
        # *NOTE: version 0.2.0 forward------>
        # convert from scaled celcius to kelvin int16->float32
        Tr_K = (g.ReadAsArray(xStart,yStart,xSize,ySize)/100)+273.15 
        g= None    
        Tr_K[np.where(albedo<0)]=np.nan
        #---------->get LC...
        sceneDir = os.path.join(self.landsatDataBase,'LC',scene)
        outFN = os.path.join(sceneDir,'%s_LC.tiff' % sceneID)        
        g = gdal.Open(outFN,GA_ReadOnly)
        LCdata = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
        #---------->get ALEXI mask...
        ET_ALEXI[np.where(albedo<0)]=-9999
        nullMask = ET_ALEXI.copy()
        nullMask[cfmask>0]=-9999
        albedo[np.where(albedo<0)]=np.nan
        
        #====================get LC based variables===============================
        s = ndimage.__file__
        envPath = os.sep.join(s.split(os.sep)[:-6])
        landsatLC = os.path.join(envPath,'share','disalexi')
        aleafv = getParFromExcel(LCdata,landsatLC,landcover,'aleafv')
        aleafn = getParFromExcel(LCdata,landsatLC,landcover,'aleafn')
        aleafl = getParFromExcel(LCdata,landsatLC,landcover,'aleafl')
        adeadv = getParFromExcel(LCdata,landsatLC,landcover,'adeadv')
        adeadn = getParFromExcel(LCdata,landsatLC,landcover,'adeadn')
        adeadl = getParFromExcel(LCdata,landsatLC,landcover,'adeadl')
        hc_min = getParFromExcel(LCdata,landsatLC,landcover,'hmin')
        hc_max = getParFromExcel(LCdata,landsatLC,landcover,'hmax')
        xl     = getParFromExcel(LCdata,landsatLC,landcover,'xl')
        clump = getParFromExcel(LCdata,landsatLC,landcover,'omega')
    
    
        LAI[np.isnan(LAI)]=0.01
        F = LAI*clump                                 #LAI for leafs spherical distribution 
        f_c = 1-(np.exp(-0.5*F))                          #fraction cover at nadir (view=0)
        f_c[f_c<=0.01]=0.01
        f_c[f_c>=0.9]=0.9

        #************************************************************************
        #Compute Canopy height and Roughness Parameters
        hc = hc_min+((hc_max-hc_min)*f_c)
        z_0M = 0.123*hc                              #;Brutsaert (1982)
        z0h = z_0M.copy()
        d_0 = 2./3.*hc
         
        # Correction of roughness parameters for bare soils (F < 0.1)
        d_0[F<=0.1]=0.00001
        z_0M[F<=0.1]=0.01
        z0h[F<=0.1]=0.0001
        
        # Correction of roughness parameters for water bodies (NDVI < 0 and albedo < 0.05)
        d_0[(ndvi<=0) & (albedo <=0.05)]=0.00001
        z_0M[(ndvi<=0) & (albedo <=0.05)]=0.00035
        z0h[(ndvi<=0) & (albedo <=0.05)]=0.00035
        
        # Check to avoid division by 0 in the next computations
        z0h[z0h==0]=0.001
        z_0M[z_0M==0]=0.01
    
        LAI[np.where(LAI==0.0)]=0.001
    
        sZ = np.tile(solZen,np.shape(LAI))
        difvis,difnir,fvis,fnir=calc_difuse_ratio(Rs_1,sZ,press=p) # fraction of difuse and PAR/NIR radiation from shortwave irradiance (W m-2, solar zenith angle, atmospheric pressure and precipitable water vapour )
        Skyl=difvis*fvis+difnir*fnir # broadband difuse fraction
        Sdn_dir=Rs_1*(1.0-Skyl)
        Sdn_dif=Rs_1*Skyl
    
        omega0 = np.zeros(LAI.shape)
        omega0= calc_omega0_Kustas(LAI, f_c, isLAIeff=True)
        omega0[np.where(np.isnan(omega0))]=0.0
        
        vza = np.tile(0.0,np.shape(LAI))
        omega = calc_omega_Kustas(omega0, vza)
        F = LAI/f_c # Real LAI
        F[np.where(F<=0.0)]=0.001
        LAI_eff = F*omega
        LAI_eff[np.where(LAI_eff>=0.0)]=0.001
        fg =1.0
        
        zs = np.tile(sz,np.shape(LAI))
    
        rsoilv = findRSOILV(difvis,difnir,fvis,fnir,Rs_1,F,f_c,fg,zs,aleafv,
                               aleafn,aleafl,adeadv,adeadn,adeadl,albedo)
        Sn_C = np.empty([F.shape[0],F.shape[1]])
        Sn_S = np.empty([F.shape[0],F.shape[1]])
        Sn_C, Sn_S = calc_Sn_Campbell (LAI, sZ, Sdn_dir, Sdn_dif, fvis,\
                            fnir, (1-aleafv)/2., (1-aleafv)/2.,(1-aleafn)/2., (1-aleafn)/2., \
                            rsoilv, rsoilv*2., LAI_eff = LAI_eff)
        Rs24 = (Rs24*0.0864)/24.0 
                            
        # find total Longwave radiation
        emis_S = np.tile(0.94,np.shape(LAI))    #Soil Emissivity [-]
        emis_C = np.tile(0.97,np.shape(LAI))       #Canopy emissivity [-]
        z_u = np.tile(2.,np.shape(LAI))
        z_T = np.tile(2.,np.shape(LAI))
        leaf_width = xl
        z0_soil = np.tile(0.01,np.shape(LAI))
        alpha_PT = np.tile(1.32,np.shape(LAI))
        f_g = np.tile(1.,np.shape(LAI))
        w_C = np.tile(1.,np.shape(LAI))
    
    
       
    #================RUN DisALEXI=================================
        
        if TSEB_only==1:
            #convert TA from scaled celcius to kelvin
            T_A_K = (T_A_K/1000.)+273.15
            e_atm = 1.0-(0.2811*(np.exp(-0.0003523*((T_A_K-273.16)**2))))                             #atmospheric emissivity (clear-sly) Idso and Jackson (1969)
            L_dn = e_atm*0.0000000567*((T_A_K)**4)
            output = TSEB_PT(
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
                leaf_width=leaf_width,
                z0_soil=z0_soil,
                alpha_PT=alpha_PT,
                x_LAD=1,
                f_c=f_c,
                f_g=f_g,
                w_C=w_C,
                resistance_form=0,
                calcG_array=[
                    [1],
                    0.35],
                    UseL=False)
    
            scaling = 1.0
            Fsun =  (output[6]+output[8])/Rs_1
            Rs24 = ndimage.gaussian_filter(Rs24, sigma=5)
            EFeq=Fsun*(Rs24)
            ET_24 = EFeq/2.45*scaling
            ET_24[ET_24<0.]=0.
            ET_24 = np.array(ET_24*1000,dtype='uint16')
        else:
            output = self.DisALEXI_PT(
                ET_ALEXI,
                Rs_1,
                Rs24,
                Tr_K,
                vza,
                u,
                ea,
                p,
                Sn_C,
                Sn_S,
                LAI,
                hc,
                emis_C,
                emis_S,
                z_0M,
                d_0,
                z_u,
                z_T,
                nullMask,
                leaf_width=leaf_width,
                z0_soil=z0_soil,
                alpha_PT=alpha_PT,
                x_LAD=1,
                f_c=f_c,
                f_g=f_g,
                w_C=w_C,
                resistance_form=0,
                calcG_array=[
                    [1],
                    0.35],
                    UseL=False)
            T_A_K= np.array((output['T_A_K']-273.15)*1000,dtype='uint16')
            
        outFormat = gdal.GDT_UInt16
        outET24Path = os.path.join(self.resultsBase,scene)
        if not os.path.exists(outET24Path):
            os.makedirs(outET24Path)
        #set ouput location and resolution
#        ulx = self.meta.CORNER_UL_PROJECTION_X_PRODUCT
#        uly = self.meta.CORNER_UL_PROJECTION_Y_PRODUCT
#        delx = self.meta.GRID_CELL_SIZE_REFLECTIVE
#        dely = self.meta.GRID_CELL_SIZE_REFLECTIVE
        ulx = ls.ulx
        uly = ls.uly
        delx = ls.delx
        dely = -ls.dely
        inUL = [ulx+(xStart*delx),uly-(yStart*dely)]
        inRes = [delx,dely]

        if TSEB_only==1:
            ET_24outName = 'ETd_%s_part_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,ET_24outName)
            writeArray2Tiff(ET_24,inRes,inUL,ls.proj4,fName,outFormat)
        else:
            T_A_KoutName = 'Ta_%s_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,T_A_KoutName)
            writeArray2Tiff(T_A_K,inRes,inUL,ls.proj4,fName,outFormat)