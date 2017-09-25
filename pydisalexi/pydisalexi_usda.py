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
from .utils import buildvrt,clean,folders
from joblib import Parallel, delayed
import types
import copy_reg
import pycurl
warnings.simplefilter('ignore', np.RankWarning)
from .landsatTools import landsat_metadata,GeoTIFF


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

   
    Folders = folders(base)    
    landsatSR = Folders['landsatSR']
    resultsBase = Folders['resultsBase']
    landsatTemp = os.path.join(landsatSR,'temp')
    landsatDataBase = Folders['landsatDataBase']

    #======FIND AVAILABLE FILES FOR PROCESSING=============================


    ####****Global******######
    #---placeholders-----
    ALEXI_ulLon = 0.0 
    ALEXI_ulLat = 0.0
    #-------------------- 375 M
    ALEXILatRes = 0.004
    ALEXILonRes = 0.004
    #---------------------roughly 4 KM 
#    ALEXILatRes = 0.032
#    ALEXILonRes = 0.032
    
    ALEXIshape = [3750,3750]
    ALEXIgeodict ={'ALEXI_ulLat':ALEXI_ulLat,'ALEXI_ulLon':ALEXI_ulLon,
                   'ALEXI_LatRes':ALEXILatRes,'ALEXI_LonRes':ALEXILonRes,
                   'ALEXIshape': ALEXIshape}
    
    #process scenes that have been preprocessed
    fileList = glob.glob(os.path.join(landsatTemp,"*_MTL.txt"))
    tiffList = glob.glob(os.path.join(landsatTemp,"*lstSharp.tiff"))
#    tiffList = glob.glob(os.path.join(landsatDataBase,"LST",scene,'*lstSharp.tiff'))
    
            
    #USER INPUT END===============================================================

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
#            dd.runDisALEXI(0,0,ALEXIgeodict,0)
            nsamples = ll.nrow
            nlines = ll.ncol
#            print 'Running disALEXI...'
#            r = Parallel(n_jobs=njobs, verbose=5)(delayed(dd.runDisALEXI)(xStart,yStart,ALEXIgeodict,0) for xStart in range(0,nsamples,200) for yStart in range(0,nlines,200))            
#            
#            # =================merge Ta files============================================
#            print 'merging Ta files...'            
#            intFile = os.path.join(resultsBase,scene,'Taxxxxx.tif')
#            cmd = 'gdal_merge.py -o %s %s' % (intFile,os.path.join(resultsBase,scene,'Ta*'))
#            buildvrt(cmd)
            
            # =================run TSEB one last time in parallel=======================
            print "run one last time in serial"
            dd.runDisALEXI(0,0,ALEXIgeodict,1)
#            print "run TSEB one last time in parallel"
#            r = Parallel(n_jobs=njobs, verbose=5)(delayed(dd.runDisALEXI)(xStart,yStart,ALEXIgeodict,1) for xStart in range(0,nsamples,200) for yStart in range(0,nlines,200)) 

            #=====================merge all files =====================================
            print 'merging ETd files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'ETd*'))
            buildvrt(cmd)
            
            #=======================clean up files===================================
            print 'cleaning up...'
            
            clean(os.path.join(resultsBase,scene),"ETd")
#            clean(os.path.join(resultsBase,scene),"Ta")
            lat_fName = os.path.join(landsatSR,'temp','lat.tif')
            lon_fName = os.path.join(landsatSR,'temp','lon.tif')
            os.remove(lat_fName)
            os.remove(lon_fName)
    
if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, pycurl.error):
        exit('Received Ctrl + C... Exiting! Bye.', 1)