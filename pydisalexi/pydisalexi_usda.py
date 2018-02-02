#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:39:37 2016

@author: mschull
"""
import os
import numpy as np
import warnings
from .disalexi_usda import disALEXI
import argparse
import glob
from .utils import writeArray2Tiff,buildvrt,clean,folders
from joblib import Parallel, delayed
import types
import copy_reg
import pycurl
warnings.simplefilter('ignore', np.RankWarning)
from .landsatTools import landsat_metadata,GeoTIFF
import time as timer
from osgeo import gdal
import pandas as pd
from osgeo.gdalconst import GA_ReadOnly


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


def main():
    # Get time and location from user
    parser = argparse.ArgumentParser()
    parser.add_argument("isUSA", type=float, help="USA=1, non-USA=0")
    parser.add_argument("njobs", type=int, default=-1, help="number of cores to use.  To use all cores available use -1")
    args = parser.parse_args()
    isUSA = args.isUSA
    base = os.getcwd()
    njobs = args.njobs
    subsetSize = 200

   
    Folders = folders(base)    
    landsatSR = Folders['landsatSR']
    resultsBase = Folders['resultsBase']
    landsatTemp = os.path.join(landsatSR,'temp')
    landsatDataBase = Folders['landsatDataBase']
    ALEXIbase = Folders['ALEXIbase']
    #======FIND AVAILABLE FILES FOR PROCESSING=============================


    ####****Global******######
    #---placeholders-----
    ALEXI_ulLon = 0.0 
    ALEXI_ulLat = 0.0
    #-------------------- 375 M
#    ALEXILatRes = 0.004
#    ALEXILonRes = 0.004
    #---------------------roughly 4 KM  FOR TESTING
    ALEXILatRes = 0.04
    ALEXILonRes = 0.04
    
    ALEXIshape = [3750,3750]
    ALEXIgeodict ={'ALEXI_ulLat':ALEXI_ulLat,'ALEXI_ulLon':ALEXI_ulLon,
                   'ALEXI_LatRes':ALEXILatRes,'ALEXI_LonRes':ALEXILonRes,
                   'ALEXIshape': ALEXIshape}
    
    #process scenes that have been preprocessed
    fileList = glob.glob(os.path.join(landsatTemp,"*_MTL.txt"))
    tiffList = glob.glob(os.path.join(landsatTemp,"*lstSharp.tiff"))
#    tiffList = glob.glob(os.path.join(landsatDataBase,"LST",scene,'*lstSharp.tiff'))
    
            
    #USER INPUT END===============================================================
    start = timer.time()
    for i in range(len(fileList)):
        filepath = fileList[i]
        tiff = tiffList[i]
        ll = GeoTIFF(tiff)
        
        meta = landsat_metadata(filepath)
        sceneID = meta.LANDSAT_SCENE_ID        
        scene = sceneID[3:9]
        finalFile = os.path.join(resultsBase,scene,'%s_ETd.tif' % sceneID[:-5])
        dt = meta.DATETIME_OBJ
        if not os.path.exists(finalFile):
                   
            #============Run DisALEXI in parallel======================================
            dd = disALEXI(filepath,dt,isUSA)
            #===COMMENTED FOR TESTING ONLY=================== 
            dd.runDisALEXI(0,0,subsetSize,subsetSize,ALEXIgeodict,0)
            nsamples = ll.nrow
            nlines = ll.ncol
            print 'Running disALEXI...'
            r = Parallel(n_jobs=njobs, verbose=5)(delayed(dd.runDisALEXI)(xStart,yStart,subsetSize,subsetSize,ALEXIgeodict,0) for xStart in range(0,nsamples,subsetSize) for yStart in range(0,nlines,subsetSize))            
            
            # =================merge Ta files============================================
#            print 'merging Ta files...'            
#
##            finalFile = os.path.join(resultsBase,scene,'%s_Ta.tif' % sceneID[:-5])
#            finalFile = os.path.join(resultsBase,scene,'Ta_DisALEXI.tif')
#            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'Ta*'))
#            buildvrt(cmd)
            print("merging Ta files----------------------->")

            tifs = glob.glob(os.path.join(resultsBase,scene,'Ta*'))
            finalFileVRT = os.path.join(resultsBase,scene,'Ta_DisALEXI.vrt')
            finalFile = os.path.join(resultsBase,scene,'Ta_DisALEXI.tif')
            outds = gdal.BuildVRT(finalFileVRT, tifs, options=gdal.BuildVRTOptions(srcNodata=-9999.))
            outds = gdal.Translate(finalFile, outds)
            outds = None
            #=========smooth the TA data=======================================
            print 'Smoothing Ta...'
            dd.smoothTaData(ALEXIgeodict)
            
            # =================run TSEB one last time in parallel=======================
#            print "run one last time in serial"
#            dd.runDisALEXI(0,0,1135,1135,ALEXIgeodict,1)
            print "run TSEB one last time in parallel"
            r = Parallel(n_jobs=njobs, verbose=5)(delayed(dd.runDisALEXI)(xStart,yStart,subsetSize,subsetSize,ALEXIgeodict,1) for xStart in range(0,nsamples,subsetSize) for yStart in range(0,nlines,subsetSize)) 

            #=====================merge all files =====================================
            finalFile = os.path.join(resultsBase,scene,'%s_ETd.tif' % sceneID[:-5])
            print 'merging ETd files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'ETd*'))
            buildvrt(cmd)
            
            
            #=============find Average ET_24======================================
            outFormat = gdal.GDT_Float32
            ls = GeoTIFF(finalFile)
            ulx = ls.ulx
            uly = ls.uly
            delx = ls.delx
            dely = -ls.dely
            inUL = [ulx,uly]
            inRes = [delx,dely]
            sceneDir = os.path.join(ALEXIbase,'%s' % scene)        
            etFN = os.path.join(sceneDir,'%s_alexiETSub.tiff' % sceneID)         
            g = gdal.Open(etFN,GA_ReadOnly)
            ET_ALEXI = g.ReadAsArray()
            g= None
            et_alexi = np.array(np.reshape(ET_ALEXI,[np.size(ET_ALEXI)])*10000, dtype='int')
            
            g = gdal.Open(finalFile,GA_ReadOnly)
            ET_24 = g.ReadAsArray()
            g= None
            
            et_24 = np.reshape(ET_24,[np.size(ET_24)])
            etDict = {'ID':et_alexi,'et':et_24}
            etDF = pd.DataFrame(etDict, columns=etDict.keys())
            group = etDF['et'].groupby(etDF['ID'])
            valMean = group.mean()
            outData = np.zeros(ET_24.size)
            for i in range(valMean.size):
                outData[et_alexi==valMean.index[i]]=valMean.iloc[i]
            et_avg = np.reshape(outData,ET_24.shape)
            et_diff = abs(et_avg-ET_ALEXI)
            outET24Path = os.path.join(resultsBase,scene)
            ET_diff_outName = '%s_ETd_diff.tif' % sceneID[:-5]
            fName = '%s%s%s' % (outET24Path,os.sep,ET_diff_outName)
            writeArray2Tiff(et_diff,inRes,inUL,ls.proj4,fName,outFormat)
            
            finalFile = os.path.join(resultsBase,scene,'%s_G0.tif' % sceneID[:-5])
            print 'merging G0 files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'G0*'))
            buildvrt(cmd)
            
            finalFile = os.path.join(resultsBase,scene,'%s_H_c.tif' % sceneID[:-5])
            print 'merging H_c files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'H_c*'))
            buildvrt(cmd)
            
            finalFile = os.path.join(resultsBase,scene,'%s_H_s.tif' % sceneID[:-5])
            print 'merging H_s files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'H_s*'))
            buildvrt(cmd)
            
            finalFile = os.path.join(resultsBase,scene,'%s_Tac.tif' % sceneID[:-5])
            print 'merging Tac files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'Tac*'))
            buildvrt(cmd)
            
            finalFile = os.path.join(resultsBase,scene,'%s_Tc.tif' % sceneID[:-5])
            print 'merging Tc files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'Tc*'))
            buildvrt(cmd)
            
            finalFile = os.path.join(resultsBase,scene,'%s_Ts.tif' % sceneID[:-5])
            print 'merging Ts files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'Ts*'))
            buildvrt(cmd)
            
            finalFile = os.path.join(resultsBase,scene,'%s_lETc.tif' % sceneID[:-5])
            print 'merging lETc files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'lETc*'))
            buildvrt(cmd)
            
            finalFile = os.path.join(resultsBase,scene,'%s_lEs.tif' % sceneID[:-5])
            print 'merging lEs files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'lEs*'))
            buildvrt(cmd)
            
            #=======================clean up files===================================
            print 'cleaning up...'
            
            clean(os.path.join(resultsBase,scene),"ETd")
#            clean(os.path.join(resultsBase,scene),"Ta")
            clean(os.path.join(resultsBase,scene),"Ts")
            clean(os.path.join(resultsBase,scene),"Tc")
            clean(os.path.join(resultsBase,scene),"G0")
            clean(os.path.join(resultsBase,scene),"H_c")
            clean(os.path.join(resultsBase,scene),"H_s")
            clean(os.path.join(resultsBase,scene),"lETc")
            clean(os.path.join(resultsBase,scene),"lEs")
            lat_fName = os.path.join(landsatSR,'temp','lat.tif')
            lon_fName = os.path.join(landsatSR,'temp','lon.tif')
    end = timer.time()
    print("program duration: %f minutes" % ((end - start)/60.))
#            os.remove(lat_fName)
#            os.remove(lon_fName)
    
if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, pycurl.error):
        exit('Received Ctrl + C... Exiting! Bye.', 1)