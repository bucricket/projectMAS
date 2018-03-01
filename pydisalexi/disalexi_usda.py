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
import pandas as pd
from .TSEB_usda import TSEB_PT_usda
from .utils import writeArray2Tiff,getParFromExcel,warp,folders
from scipy import ndimage,interp
from .landsatTools import landsat_metadata,GeoTIFF
from .TSEB_utils_usda import sunset_sunrise,interp_ta
#from joblib import Parallel, delayed
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve_fft




class disALEXI(object):
    def __init__(self, fn,dt,isUSA):
#        base = os.path.abspath(os.path.join(fn,os.pardir,os.pardir,os.pardir,
#                                            os.pardir,os.pardir))
        base = os.getcwd()

        Folders = folders(base)  
        self.landsatSR = Folders['landsatSR']
#        self.landsatNDVI = Folders['landsatNDVI']
#        self.ALEXIbase = Folders['ALEXIbase']
#        self.metBase = Folders['metBase']
#        self.landsatDataBase = Folders['landsatDataBase']
        self.resultsBase = Folders['resultsBase']
        self.fn = fn
        self.meta = landsat_metadata(fn)
        self.sceneID = self.meta.LANDSAT_SCENE_ID
        self.productID = fn.split(os.sep)[-1][:-8]
#        self.productID = self.meta.LANDSAT_PRODUCT_ID
        self.scene = self.sceneID[3:9]
        self.isUSA = isUSA
        self.dt = dt
        self.satscene_path = os.sep.join(fn.split(os.sep)[:-2])
        





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
                    p,
                    zs,
                    aleafv, 
                    aleafn, 
                    aleafl, 
                    adeadv, 
                    adeadn, 
                    adeadl,
                    albedo,
                    ndvi,
                    LAI,
                    clump,
                    hc,
                    mask,
                    time,
                    t_rise,
                    t_end,
                    leaf_width=1.,
                    alpha_PT=1.32):
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
        Tr_Kresize = np.tile(np.array(np.resize(Tr_K,[np.size(Tr_K),1])),(1,MatXsize))
        vzaresize = np.tile(np.resize(vza,[np.size(vza),1]),(1,MatXsize))
#        T_A_Kresize = np.tile(range(270,340,10),(np.size(vza),1))
        Tr_ADD = np.tile(np.transpose(range(0,20,3)),[np.size(hc),1])
        Tr_Kcol = np.resize(Tr_K,[np.size(Tr_K),1])-10.
        T_A_Kresize = Tr_Kcol+Tr_ADD
        uresize = np.tile(np.resize(u,[np.size(u),1]),(1,MatXsize))
        presize = np.tile(np.resize(p,[np.size(p),1]),(1,MatXsize))
        Rs_1resize = np.tile(np.resize(Rs_1,[np.size(Rs_1),1]),(1,MatXsize))
        zsresize = np.tile(np.resize(zs,[np.size(zs),1]),(1,MatXsize))
        aleafvresize = np.tile(np.resize(aleafv,[np.size(hc),1]),(1,MatXsize))
        aleafnresize = np.tile(np.resize(aleafn,[np.size(hc),1]),(1,MatXsize))
        aleaflresize = np.tile(np.resize(aleafl,[np.size(hc),1]),(1,MatXsize))
        adeadvresize = np.tile(np.resize(adeadv,[np.size(hc),1]),(1,MatXsize))
        adeadnresize = np.tile(np.resize(adeadn,[np.size(hc),1]),(1,MatXsize))
        adeadlresize = np.tile(np.resize(adeadl,[np.size(hc),1]),(1,MatXsize))
        albedoresize = np.tile(np.resize(albedo,[np.size(hc),1]),(1,MatXsize))
        ndviresize = np.tile(np.resize(ndvi,[np.size(hc),1]),(1,MatXsize))
        LAIresize = np.tile(np.resize(LAI,[np.size(LAI),1]),(1,MatXsize))
        clumpresize = np.tile(np.resize(clump,[np.size(hc),1]),(1,MatXsize))
        hcresize = np.tile(np.resize(hc,[np.size(hc),1]),(1,MatXsize))
        maskresize = np.tile(np.array(np.resize(mask,[np.size(hc),1])),(1,MatXsize))
        timeresize = np.tile(np.array(np.resize(time,[np.size(hc),1])),(1,MatXsize))
        t_riseresize = np.tile(np.array(np.resize(t_rise,[np.size(hc),1])),(1,MatXsize))
        t_endresize = np.tile(np.array(np.resize(t_end,[np.size(hc),1])),(1,MatXsize))
        leaf_widthresize = np.tile(np.resize(leaf_width,[np.size(hc),1]),(1,MatXsize))
        alpha_PTresize = np.tile(np.resize(alpha_PT,[np.size(hc),1]),(1,MatXsize))

        # run TSEB over TA options
        output = TSEB_PT_usda(
            Tr_Kresize,
            vzaresize,
            T_A_Kresize,
            uresize,
            presize,
            Rs_1resize,
            zsresize,
            aleafvresize, 
            aleafnresize, 
            aleaflresize, 
            adeadvresize, 
            adeadnresize, 
            adeadlresize,
            albedoresize,
            ndviresize,
            LAIresize,
            clumpresize,
            hcresize,
            maskresize,
            timeresize,
            t_riseresize,
            t_endresize,
            leaf_width=leaf_widthresize,
            a_PT_in=alpha_PTresize)
            
        scaling = 1.0
        Fsun =  (output[4]+output[6])/np.resize(Rs_1,[np.size(hc),1])
        EFeq=Fsun*(np.reshape(Rs24in,[np.size(hc),1]))
        et = EFeq/2.45*scaling
        et[et<0.01] = 0.01
        
        
        #=============find Average ETd====================================== 
        
        et_alexi = np.array(np.reshape(ET_ALEXI,[np.size(ET_ALEXI)])*10000, dtype='int')
        etDict = {'ID':et_alexi,
                  'et1':np.reshape(et[:,0],[np.size(ET_ALEXI)]),
                  'et2':np.reshape(et[:,1],[np.size(ET_ALEXI)]),
                  'et3':np.reshape(et[:,2],[np.size(ET_ALEXI)]),
                  'et4':np.reshape(et[:,3],[np.size(ET_ALEXI)]),
                  'et5':np.reshape(et[:,4],[np.size(ET_ALEXI)]),
                  'et6':np.reshape(et[:,5],[np.size(ET_ALEXI)]),
                  'et7':np.reshape(et[:,6],[np.size(ET_ALEXI)])}
        
        etDF = pd.DataFrame(etDict, columns=etDict.keys())
        etDF = pd.DataFrame(etDict)
        group = etDF.groupby(etDF['ID'])
        valMean = group.mean()
        outData = np.zeros(et.shape)
        for i in range(valMean.shape[0]):
            outData[et_alexi==valMean.index[i]]=valMean.iloc[i]
        et = np.reshape(outData,et.shape)
        # ======interpolate over mutiple Ta ===================================

        from scipy.interpolate import interp1d
        x = range(0,20,3)
        ET_ALEXI[mask==0]=-9999.
        et_alexi = np.reshape(ET_ALEXI,[np.size(hc),1])
        bias = et_alexi-et
        # check if all values inrow are nan
        nanIndex = np.sum(np.isnan(bias),axis=1)
        # set all to 1 so it doesnt throw an error below
        bias[np.where(nanIndex==MatXsize),:]=1.
        f_bias = interp1d(x,bias,kind='linear', bounds_error=False)
        f_ta= interp1d(x,T_A_Kresize,kind='linear', bounds_error=False)

        biasInterp = f_bias(np.linspace(0,20,1000))
        TaInterp = f_ta(np.linspace(0,20,1000))
        # extract the Ta based on minimum bias at Fine resolution
        minBiasIndex = np.array(np.nanargmin(abs(biasInterp),axis=1))
        TaExtrap = TaInterp[np.array(range(np.size(hc))),minBiasIndex]
        TaExtrap[np.where(nanIndex==MatXsize)]=np.nan
        Tareshape = np.reshape(TaExtrap,np.shape(hc))
        
        T_A_K = Tareshape
        output ={'T_A_K':T_A_K}
        return output
    
    def smoothTaData(self,ALEXIgeodict):
        
        ALEXILatRes = ALEXIgeodict['ALEXI_LatRes']
        ALEXILonRes = ALEXIgeodict['ALEXI_LonRes']
        sceneID =self.sceneID
        scene = self.scene
        outFN = os.path.join(self.resultsBase,scene,'%s_Ta.tif' % sceneID[:-5])
        inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        #=======================convert fine TA to coarse resolution=========
        outfile = os.path.join(self.resultsBase,scene,'Ta_DisALEXI.tif')
#            outfile = os.path.join(self.resultsBase,scene,'%s_Ta.tif' % sceneID[:-5])

        coarseFile = os.path.join(self.resultsBase,scene,'TaCoarse.tif')
        coarse2fineFile = os.path.join(self.resultsBase,scene,'TaCoarse2Fine.tif')
#            outFN = coarseFile[:-10]+'.tif'
        
        if not os.path.exists(outFN):
            print 'get->Ta'
            # get mask from Landsat LAI
            ls = GeoTIFF(outfile)
#                laiFN = os.path.join(self.landsatDataBase,'LAI',scene,'lndlai.%s.hdf' % sceneID)
#                hdf = SD(laiFN,SDC.READ)
#                data2D = hdf.select('cfmask')
#                cfmask = data2D[:,:].astype(np.double)
            sceneDir = os.path.join(self.satscene_path,'CF_MASK')
            maskFN = os.path.join(sceneDir,'%s_Mask.tiff' % sceneID)
            g = gdal.Open(maskFN,GA_ReadOnly)
            cfmask = g.ReadAsArray()
            g= None
#            g = gdal.Open(outfile,GA_ReadOnly)
#            ta = g.ReadAsArray()
#            ta[cfmask == 1]=np.nan 
#            g= None
#            mask = os.path.join(self.resultsBase,scene,"TafineMask.tif")
#            masked = os.path.join(self.resultsBase,scene,"TafineMasked.tif")
#            ls.clone(mask,ta)
#            subprocess.check_output('gdal_fillnodata.py %s %s -mask %s -of GTiff' % (outfile,masked,mask),shell=True)
            #=============find Average Ta====================================== COMMENTED FOR TESTING
            in_ds = gdal.Open(outfile)
            coarseds = gdal.Translate(coarseFile,in_ds,
                                      options=gdal.TranslateOptions(
                                              resampleAlg='average',
                                              xRes=400,
                                              yRes=400))
            fineds = gdal.Warp(outFN,coarseds,options=gdal.WarpOptions(resampleAlg='average',
                                                                        height=ls.nrow,
                                                                        width=ls.nrow))
            coarseds=None
            #========smooth Ta data========================================
            ta = fineds.ReadAsArray()
            fineRes = ls.Lat[1,0]-ls.Lat[0,0]
            coarseRes = ALEXILatRes
            course2fineRatio = coarseRes**2/fineRes**2
            rid2 = int(np.sqrt(course2fineRatio))
            gauss_kernal = Gaussian2DKernel(rid2)
            ta = convolve_fft(ta, gauss_kernal,allow_huge=True)
            fineds.GetRasterBand(1).WriteArray(ta)
            fineds = None
#            sceneDir = os.path.join(self.ALEXIbase,'%s' % scene)
#            sceneDir = os.path.join(self.satscene_path,'ET','400m')
#            etFN = os.path.join(sceneDir,'%s_alexiET.tiff' % sceneID)         
#            g = gdal.Open(etFN,GA_ReadOnly)
#            ET_ALEXI = g.ReadAsArray()
#            g= None
#            et_alexi = np.array(np.reshape(ET_ALEXI,[np.size(ET_ALEXI)])*10000, dtype='int')
#            ta_masked = np.reshape(ta,[np.size(ta)])
#            taDict = {'ID':et_alexi,'ta':ta_masked}
#            taDF = pd.DataFrame(taDict, columns=taDict.keys())
#            group = taDF['ta'].groupby(taDF['ID'])
#            valMean = group.mean()
#            outData = np.zeros(ta_masked.size)
#            def getHighResMean(i):
#                outData[et_alexi==valMean.index[i]]=valMean.iloc[i]
#                return outData
#            for i in range(valMean.size):
#                outData[et_alexi==valMean.index[i]]=valMean.iloc[i]
#            outData = Parallel(n_jobs=-1, verbose=5)(delayed(getHighResMean)(i) for i in range(valMean.size)) 
#            outData = np.array(taDF.groupby('ID')['ta'].transform('mean'))
#            print ta.shape
#            outData[np.isnan(ta_masked)]=np.nan
#            ta = np.reshape(outData,ta.shape)
            
            ulx = ls.ulx
            uly = ls.uly
            delx = ls.delx
            dely = -ls.dely
            fineRes = ls.Lat[1,0]-ls.Lat[0,0]
            coarseRes = ALEXILatRes
            inUL = [ulx,uly]
            inRes = [delx,dely]
            
#            Ta = interp_ta(ta,coarseRes,fineRes)-273.16
            Ta = ta-273.16 # FOR TESTING!!
            
            outFormat = gdal.GDT_Float32 
            writeArray2Tiff(Ta,inRes,inUL,ls.proj4,outFN,outFormat)
            os.remove(coarseFile)
    
    def runDisALEXI(self,xStart,yStart,xSize,ySize,ALEXIgeodict,TSEB_only):
        # USER INPUT============================================================
        ALEXILatRes = ALEXIgeodict['ALEXI_LatRes']
        ALEXILonRes = ALEXIgeodict['ALEXI_LonRes']
        sceneID =self.sceneID
        scene = self.scene
        productID = self.productID
#        xSize = 200
#        ySize = 200
            
        #-------------pick Landcover map----------------
        if self.isUSA ==1:
            landcover = 'NLCD'
        else:
            landcover = 'GlobeLand30'
        
        yeardoy = sceneID[9:16]
        #-------------get Landsat information-----------
#        ls = GeoTIFF(os.path.join(self.landsatSR,'temp',"%s_band10.tif" % productID))
        sceneDir = os.path.join(self.satscene_path,'LST')
        ls = GeoTIFF(os.path.join(sceneDir,'%s_lstSharp.tiff' % sceneID))
        g = gdal.Open(os.path.join(sceneDir,'%s_lstSharp.tiff' % sceneID))
        solZen = self.meta.SUN_ELEVATION
#        nsamples = int(self.meta.REFLECTIVE_SAMPLES)
#        nlines = int(self.meta.REFLECTIVE_LINES)
        nsamples = g.RasterXSize
        nlines = g.RasterYSize
        if xStart==((nsamples/xSize)*xSize):
            xSize = (nsamples-xStart)
        if yStart==((nlines/ySize)*ySize):
            ySize = (nlines-yStart)
        g =None
        inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        sz = np.radians(90-solZen) # convert sza to radians
    
    
        #===========================get the ETd data==============================
    
#        sceneDir = os.path.join(self.ALEXIbase,'%s' % scene)  
        sceneDir = os.path.join(self.satscene_path,'ET','400m')
        outFN = os.path.join(sceneDir,'%s_alexiET.tiff' % sceneID)         
        g = gdal.Open(outFN,GA_ReadOnly)
        ET_ALEXI = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
           
        
        #=============get MET data================================================
        #get CFSR MET data at overpass time
    
#        sceneDir = os.path.join(self.metBase,'%s' % scene)
        sceneDir = os.path.join(self.satscene_path,'MET')
        #------------get-> surface pressure...
        outFN = os.path.join(sceneDir,'%s_p.tiff' % sceneID) 
        g = gdal.Open(outFN,GA_ReadOnly)
#        print("xsize:%d" % xSize)
#        print("ysize:%d" % ySize)
        p = g.ReadAsArray(xStart,yStart,xSize,ySize)
        p /=100. #convert to mb
        g= None

            
        #------------get-> ea...
        outFN = os.path.join(sceneDir,'%s_q2.tiff' % sceneID) 
        g = gdal.Open(outFN,GA_ReadOnly)
        q2 = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
    
        ea = ((q2*(1000./621.9907))*(p*100.))*0.001                             #kPa
        ea *= 10. #mb
        #====get air temperature===============================================
        outFN = os.path.join(self.resultsBase,scene,'%s_Ta.tif' % sceneID[:-5])
        if (TSEB_only==1):
#            if ((xStart==0) & (yStart==0)):
#                #ls = GeoTIFF(os.path.join(self.landsatSR, scene,'%s_sr_band1.tif' % productID))
#    
#                #=======================convert fine TA to coarse resolution=========
#                outfile = os.path.join(self.resultsBase,scene,'Ta_DisALEXI.tif')
#    #            outfile = os.path.join(self.resultsBase,scene,'%s_Ta.tif' % sceneID[:-5])
#        
#                coarseFile = os.path.join(self.resultsBase,scene,'TaCoarse.tif')
#                coarse2fineFile = os.path.join(self.resultsBase,scene,'TaCoarse2Fine.tif')
#    #            outFN = coarseFile[:-10]+'.tif'
#                
#                if not os.path.exists(outFN):
#                    print 'get->Ta'
#                    # get mask from Landsat LAI
#                    ls = GeoTIFF(outfile)
#    #                laiFN = os.path.join(self.landsatDataBase,'LAI',scene,'lndlai.%s.hdf' % sceneID)
#    #                hdf = SD(laiFN,SDC.READ)
#    #                data2D = hdf.select('cfmask')
#    #                cfmask = data2D[:,:].astype(np.double)
#                    
#                    maskFN = os.path.join(self.landsatDataBase,'Mask',scene,'%s_mask.tiff' % sceneID)
#                    g = gdal.Open(maskFN,GA_ReadOnly)
#                    cfmask = g.ReadAsArray()
#                    g= None
#                    g = gdal.Open(outfile,GA_ReadOnly)
#                    ta = g.ReadAsArray()
#                    ta[cfmask > 0]=0
#                    g= None
#                    mask = os.path.join(self.resultsBase,scene,"TafineMask.tif")
#                    masked = os.path.join(self.resultsBase,scene,"TafineMasked.tif")
#                    ls.clone(mask,ta)
#                    subprocess.check_output('gdal_fillnodata.py %s %s -mask %s -of GTiff' % (outfile,masked,mask),shell=True)
#                    optionList = ['-overwrite', '-s_srs', '%s' % ls.proj4,'-t_srs',
#                                  '%s' % inProj4,'-r', 'average','-tr', 
#                                  '%f' % ALEXILatRes, '%f' % ALEXILonRes,
#                                  '-srcnodata','270.','-dstnodata','0.0',
#                                  '-of','GTiff','%s' % masked, '%s' % coarseFile]
#                    
#                    warp(optionList)
#                    #=======now convert the averaged coarse Ta to fine resolution==
#                    nrow = ls.nrow+100.
#                    ncol = ls.ncol+100.
#                    optionList = ['-overwrite', '-s_srs', '%s' % inProj4, '-t_srs', 
#                                  '%s' % ls.proj4,'-r', 'near','-ts', 
#                                  '%f' % nrow, '%f' % ncol,'-of',
#                                  'GTiff','%s' % coarseFile, '%s' % coarse2fineFile]
#                    warp(optionList)
#                    #========smooth Ta data========================================
#                    ulx = ls.ulx
#                    uly = ls.uly
#                    lrx = ls.lrx
#                    lry = ls.lry
#                    delx = ls.delx
#                    dely = -ls.dely
#                    fineRes = ls.Lat[1,0]-ls.Lat[0,0]
#                    coarseRes = ALEXILatRes
#                    inUL = [ulx,uly]
#                    inRes = [delx,dely]
#                    g = gdal.Open(coarse2fineFile,GA_ReadOnly)
#                    ta = g.ReadAsArray()
#                    g= None
#                    
#                    Ta = interp_ta(ta,coarseRes,fineRes)-273.16
#                    
#                    outFormat = gdal.GDT_Float32 
#                    writeArray2Tiff(Ta,inRes,inUL,ls.proj4,outFN,outFormat)
#    #
#    #                optionList = ['-overwrite','-te', '%f' % ulx, '%f' % lry,
#    #                              '%f' % lrx,'%f' % uly,'-tr',
#    #                              '%f' % delx, '%f' % dely ,'-multi','-of','GTiff',
#    #                              '%s' % coarse2fineFile, '%s' % outFN]
#    #                warp(optionList)
#                    #os.remove(coarseFile)
            g = gdal.Open(outFN,GA_ReadOnly)
            T_A_K = g.ReadAsArray(xStart,yStart,xSize,ySize)+273.16
            g= None
        

#        sceneDir = os.path.join(self.metBase,'%s' % scene)
        outFN = os.path.join(sceneDir,'%s_u.tiff' % sceneID) 
        g = gdal.Open(outFN,GA_ReadOnly)
        u = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
        #===create lat long files==============================================
        
        inUL = [ls.ulx,ls.uly]
        inRes = [ls.delx,-ls.dely]
        lat_fName = os.path.join(self.landsatSR,'temp','lat.tif')
        lon_fName = os.path.join(self.landsatSR,'temp','lon.tif')
        if not os.path.exists(lat_fName):
            lats = ls.Lat_pxcenter
            lons = ls.Lon_pxcenter
            writeArray2Tiff(lats,inRes,inUL,ls.proj4,lat_fName,gdal.GDT_Float32)
            writeArray2Tiff(lons,inRes,inUL,ls.proj4,lon_fName,gdal.GDT_Float32)
        g = gdal.Open(lat_fName,GA_ReadOnly)
        lat = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
        
        g = gdal.Open(lon_fName,GA_ReadOnly)
        lon = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
        
        
        #====get overpass hour insolation======================================
        sceneDir = os.path.join(self.satscene_path,'INSOL')
        outFN = os.path.join(sceneDir,'%s_Insol1.tiff' % sceneID)            
        g = gdal.Open(outFN,GA_ReadOnly)
        Rs_1 = g.ReadAsArray(xStart,yStart,xSize,ySize)*0.042727217
        g= None
    
        #====get daily insolation=========================================
        outFN = os.path.join(sceneDir,'%s_Insol24.tiff' % sceneID)
        g = gdal.Open(outFN,GA_ReadOnly)
        Rs24 = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
    
        #===============get biophysical parameters at overpass time============
        sceneDir = os.path.join(self.satscene_path,'ALBEDO')
#        sceneDir = os.path.join(self.landsatDataBase,'albedo',scene)
        outFN = os.path.join(sceneDir,'%s_albedo.tiff' % sceneID)         
        g = gdal.Open(outFN,GA_ReadOnly)
        albedo = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
    
        #------>get LAI...
#        outFN = os.path.join(self.landsatDataBase,'LAI',scene,'lndlai.%s.hdf' % sceneID)
        sceneDir = os.path.join(self.satscene_path,'LAI')
        outFN = os.path.join(sceneDir,'%s_lai.tiff' % sceneID)
        g = gdal.Open(outFN,GA_ReadOnly)
        LAI = g.ReadAsArray(xStart,yStart,xSize,ySize)#*0.001 # TESTING
        g= None
#        LAI[np.where(LAI==-9.999)]=np.nan
#        LAI[np.where(LAI<=0.)]=0.001
        
#        hdf = SD(outFN,SDC.READ)
#        data2D = hdf.select('LAI')
#        LAI = data2D[yStart:yStart+ySize,xStart:xStart+xSize].astype(np.double)*0.001
#    
#        LAI[np.where(LAI==-9.999)]=np.nan
#        LAI[np.where(LAI<=0.)]=0.001
        
        #------>get ndvi...'
#        sceneDir = os.path.join(self.landsatNDVI,scene)
        sceneDir = os.path.join(self.satscene_path,'NDVI')
        outFN = os.path.join(sceneDir,'%s_ndvi.tiff' % sceneID)
        g = gdal.Open(outFN,GA_ReadOnly)
        ndvi = g.ReadAsArray(xStart,yStart,xSize,ySize)#*0.001 # TESTING
        g= None
#        ndvi[np.where(ndvi==-9999.)]=np.nan
        
#        data2D = hdf.select('NDVI')
#        ndvi = data2D[yStart:yStart+ySize,xStart:xStart+xSize].astype(np.double)*0.001
#        ndvi[np.where(ndvi==-9.999)]=np.nan
        
        #===get cfmask=======
        sceneDir = os.path.join(self.satscene_path,'CF_MASK')
        outFN = os.path.join(sceneDir,'%s_Mask.tiff' % sceneID)
        g = gdal.Open(outFN,GA_ReadOnly)
        cfmask = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
#        laiFN = os.path.join(self.landsatDataBase,'LAI',scene,'lndlai.%s.hdf' % sceneID)
#        hdf = SD(laiFN,SDC.READ)
#        data2D = hdf.select('cfmask')
#        cfmask = data2D[yStart:yStart+ySize,xStart:xStart+xSize].astype(np.double)
    
        
        #---------->get LST...   
        sceneDir = os.path.join(self.satscene_path,'LST')
        outFN = os.path.join(sceneDir,'%s_lstSharp.tiff' % sceneID)
        g = gdal.Open(outFN,GA_ReadOnly)
        # *NOTE: version 0.2.0 forward------>
        # convert from scaled celcius to kelvin int16->float32
        Tr_K = (g.ReadAsArray(xStart,yStart,xSize,ySize)/100.)+273.15 
#        Tr_K = g.ReadAsArray(xStart,yStart,xSize,ySize)+273.16 # TESTING
        g= None    
        Tr_K[np.where(albedo<0)]=np.nan
        #---------->get LC...
        sceneDir = os.path.join(self.satscene_path,'LC')
#        sceneDir = os.path.join(self.landsatDataBase,'LC',scene)
        outFN = os.path.join(sceneDir,'%s_LC.tiff' % sceneID)        
        g = gdal.Open(outFN,GA_ReadOnly)
        LCdata = g.ReadAsArray(xStart,yStart,xSize,ySize)
        g= None
        #---------->get ALEXI mask...
#        ET_ALEXI[np.where(albedo<0)]=-9999
#        mask = ET_ALEXI.copy()
        mask = np.tile(1.,albedo.shape)
        mask[cfmask>0]=0.
        mask[albedo<0]=0.
#        mask[mask==0]=1.
#        mask[mask==-9999.] = 0.
        albedo[np.where(albedo<0.)]=np.nan
        
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
        clump[clump==0]=0.99
        LAI[LCdata==11]=0.01
        ndvi[LCdata==11]=-0.5
        
        
        aleafv[np.isnan(aleafv)]=0.9
        aleafn[np.isnan(aleafn)]=0.9
        aleafl[np.isnan(aleafl)]=0.9
        adeadv[np.isnan(adeadv)]=0.2
        adeadn[np.isnan(adeadn)]=0.2
        adeadl[np.isnan(adeadl)]=0.2
        hc_min[np.isnan(hc_min)]=0.1
        hc_max[np.isnan(hc_max)]=0.5
        xl[np.isnan(xl)]=0.5
        xl[xl == 0.]=0.5


        F = LAI*clump                                 #LAI for leafs spherical distribution 
        f_c = 1-(np.exp(-0.5*F))                          #fraction cover at nadir (view=0)
        f_c[f_c<=0.01]=0.01
        f_c[f_c>=0.9]=0.9

        #************************************************************************
        #Compute Canopy height and Roughness Parameters
        hc = hc_min+((hc_max-hc_min)*f_c)
    
#        LAI[np.where(LAI==0.0)]=0.001        
        vza = np.tile(0.0,np.shape(LAI))
#        Rs24 = Rs24+500. # FOR TESTING ONLY
#        Rs24 = (Rs24*0.0864)/24.0 # MERRA
        Rs24 = Rs24*0.0864 #GSIP

        leaf_width = xl
        alpha_PT = np.tile(1.32,np.shape(LAI))
        time = self.dt.hour+(self.dt.minute/60.)
#        print("time:%f" % time)
        t_rise, t_end, zs = sunset_sunrise(self.dt,np.deg2rad(lon),np.deg2rad(lat),time)
#        print("t_rise:%f, t_end:%f" % (t_rise[0,0],t_end[0,0]))
#        zs = np.tile(sz,np.shape(LAI))
       
    #================RUN DisALEXI=================================
        
        if TSEB_only==1:
            #convert TA from scaled celcius to kelvin
#            T_A_K = (T_A_K/1000.)+273.15  # removed /1000 FOR TESTING!!!!
#            T_A_K = (T_A_K)+273.16  # removed /1000 FOR TESTING!!!!
            nan_check = np.sum(np.isnan(LAI))/LAI.size
            if nan_check==1: # All nans
                ET_24 = np.tile(np.nan,LAI.shape)
            else:
                output = TSEB_PT_usda(
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
                    LAI,
                    clump,
                    hc,
                    mask,
                    time,
                    t_rise,
                    t_end,
                    leaf_width=leaf_width,
                    a_PT_in=alpha_PT)                           #atmospheric emissivity (clear-sly) Idso and Jackson (1969)
    
        
                scaling = 1.0
                Fsun =  (output[4]+output[6])/Rs_1
    #            Rs24 = ndimage.gaussian_filter(Rs24, sigma=5)
                EFeq=Fsun*(Rs24)
    #            ET_24 = EFeq/2.45*scaling
                ET_24 = EFeq*0.408*scaling
                ET_24[ET_24<0.01]=0.01
#            ET_24 = np.array(ET_24*1000.,dtype='uint16')
        else:
            nan_check = np.sum(np.isnan(LAI))/LAI.size
            if nan_check==1: # All nans
                T_A_K = np.tile(np.nan,LAI.shape)
            else:
                output = self.DisALEXI_PT(
                        ET_ALEXI,
                        Rs_1,
                        Rs24,
                        Tr_K,
                        vza,
                        u,
                        p,
                        zs,
                        aleafv, 
                        aleafn, 
                        aleafl, 
                        adeadv, 
                        adeadn, 
                        adeadl,
                        albedo,
                        ndvi,
                        LAI,
                        clump,
                        hc,
                        mask,
                        time,
                        t_rise,
                        t_end,
                        leaf_width=leaf_width,
                        alpha_PT=alpha_PT)
    #            T_A_K= np.array((output['T_A_K']-273.15)*1000.,dtype='uint16')
                T_A_K= np.array(output['T_A_K'],dtype='float32')
            
#        outFormat = gdal.GDT_UInt16
        outFormat = gdal.GDT_Float32
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
            outFormat = gdal.GDT_Float32 # FOR TESTING WE CAN GO BACK TO INT LATER
            ET_24outName = 'ETd_%s_part_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,ET_24outName)
            writeArray2Tiff(ET_24,inRes,inUL,ls.proj4,fName,outFormat)
            
            #======write out fluxes==================================
#            flag, Ts, Tc, Tac, lETc, H_c, lEs, H_s, G0
            outFormat = gdal.GDT_Float32
            #==Ts=====>
            Ts_24outName = 'Ts_%s_part_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,Ts_24outName)
            writeArray2Tiff(output[1],inRes,inUL,ls.proj4,fName,outFormat)
            #==Tc=====>
            Tc_24outName = 'Tc_%s_part_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,Tc_24outName)
            writeArray2Tiff(output[2],inRes,inUL,ls.proj4,fName,outFormat)
            #==Tac=====>
            Tac_24outName = 'Tac_%s_part_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,Tac_24outName)
            writeArray2Tiff(output[3],inRes,inUL,ls.proj4,fName,outFormat)
            #==lETc=====>
            lETc_24outName = 'lETc_%s_part_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,lETc_24outName)
            writeArray2Tiff(output[4],inRes,inUL,ls.proj4,fName,outFormat)
            #==H_c=====>
            H_c_24outName = 'H_c_%s_part_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,H_c_24outName)
            writeArray2Tiff(output[5],inRes,inUL,ls.proj4,fName,outFormat)
            #==lEs=====>
            lEs_24outName = 'lEs_%s_part_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,lEs_24outName)
            writeArray2Tiff(output[6],inRes,inUL,ls.proj4,fName,outFormat)
            #==H_s=====>
            H_s_24outName = 'H_s_%s_part_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,H_s_24outName)
            writeArray2Tiff(output[7],inRes,inUL,ls.proj4,fName,outFormat)
            #==G0=====>
            G0_24outName = 'G0_%s_part_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,G0_24outName)
            writeArray2Tiff(output[8],inRes,inUL,ls.proj4,fName,outFormat)
            
            
            
        else:
            T_A_KoutName = 'Ta_%s_%d_%d.tif' % (yeardoy,xStart,yStart)
            fName = '%s%s%s' % (outET24Path,os.sep,T_A_KoutName)
            writeArray2Tiff(T_A_K,inRes,inUL,ls.proj4,fName,outFormat)