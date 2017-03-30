#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:39:37 2016

@author: mschull
"""
import os
import numpy as np
from .search import Search
import warnings
from .disalexi import disALEXI
import keyring
import getpass
import argparse
from .utils import buildvrt,clean,folders,search
from pydap.cas.urs import setup_session
from joblib import Parallel, delayed
import types
import copy_reg
import pycurl
warnings.simplefilter('ignore', np.RankWarning)


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


def main():
    # Get time and location from user
    parser = argparse.ArgumentParser()
    parser.add_argument("lat", type=float, help="latitude")
    parser.add_argument("lon", type=float, help="longitude")
    parser.add_argument("isUSA", type=float, help="USA=1, non-USA=0")
    parser.add_argument("startDate", type=str, help="Start date yyyy-mm-dd")
    parser.add_argument("endDate", type=str, help="Start date yyyy-mm-dd")
    parser.add_argument("ET_dir", type=str, help="ALEXI ET directory")
    parser.add_argument("njobs", type=int, help="number of cores to use.  To use all cores available use -1")
    args = parser.parse_args()
    GCP = [args.lat,args.lon] 
    startDate = args.startDate
    endDate = args.endDate
    isUSA = args.isUSA
    if isUSA == 0:
        parser.add_argument("LC_dir", type=str, help="Landcover directory")
    #base = args.base
    base = os.getcwd()
    njobs = args.njobs
    LC_dir = args.LC_dir

    
    Folders = folders(base)    
    landsatSR = Folders['landsatSR']
    resultsBase = Folders['resultsBase']
    
    #%%        
     # =====earthData credentials===============
    earthLoginUser = str(getpass.getpass(prompt="earth login username:"))
    if keyring.get_password("nasa",earthLoginUser)==None:
        earthLoginPass = str(getpass.getpass(prompt="earth login password:"))
        keyring.set_password("nasa",earthLoginUser,earthLoginPass)
    else:
        earthLoginPass = str(keyring.get_password("nasa",earthLoginUser)) 

    session = setup_session(earthLoginUser, earthLoginPass)
    #======FIND AVAILABLE FILES FOR PROCESSING=============================


    ####****Global******######
    #---placeholders-----
    ALEXI_ulLon = 0.0 
    ALEXI_ulLat = 0.0
    #--------------------
    ALEXILatRes = 0.004
    ALEXILonRes = 0.004
    ALEXIshape = [3750,3750]
    ALEXIgeodict ={'ALEXI_ulLat':ALEXI_ulLat,'ALEXI_ulLon':ALEXI_ulLon,
                   'ALEXI_LatRes':ALEXILatRes,'ALEXI_LonRes':ALEXILonRes,
                   'ALEXIshape': ALEXIshape}
    
    #find the scenes
    try:
        s = Search()
        scenes = s.search(lat=GCP[0],lon=GCP[1],limit = 100, start_date = startDate,end_date=endDate, cloud_max=5)
        sceneID = str(scenes['results'][0]['sceneID'])
    except:
        sceneIDs = search(GCP[0],GCP[1],startDate, endDate)
        sceneID = sceneIDs[0]
            
    #USER INPUT END===============================================================
    #%% 
    
    scene = sceneID[3:9]
    
    
    #============Run DisALEXI in parallel======================================
    #print 'run DisALEXI once to avoid huge overhead issues in parallel runs'
    fn = os.path.join(landsatSR,scene,"%s.xml" % sceneID)
    dd = disALEXI(fn,session,LC_dir)
    dd.runDisALEXI(0,0,fn,isUSA,ALEXIgeodict,0)
    
    print 'Running disALEXI...'
    r = Parallel(n_jobs=njobs, verbose=5)(delayed(dd.runDisALEXI)(xStart,yStart,fn,isUSA,ALEXIgeodict,0) for xStart in range(0,7600,200) for yStart in range(0,7600,200))            
    
    # =================merge Ta files============================================
    print 'merging Ta files...'
    
    finalFile = os.path.join(resultsBase,scene,'Taxxxxx.tif')
    cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'Ta*'))
    buildvrt(cmd)
    
    # =================run TSEB one last time in parallel=======================
    #print 'run DisALEXI once to avoid huge overhead issues in parallel runs'
    dd.runDisALEXI(0,0,fn,isUSA,ALEXIgeodict,1)
    print "run TSEB one last time in parallel"
    r = Parallel(n_jobs=njobs, verbose=5)(delayed(dd.runDisALEXI)(xStart,yStart,fn,isUSA,ALEXIgeodict,1) for xStart in range(0,7600,200) for yStart in range(0,7600,200)) 
    #=====================merge all files =====================================
    print 'merging ETd files...'
    finalFile = os.path.join(resultsBase,scene,'%s_ETd.tif' % sceneID[:-5])
    cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'ETd*'))
    buildvrt(cmd)
    
    #=======================clean up files===================================
    print 'cleaning up...'
    
    clean(os.path.join(resultsBase,scene),"ETd")
    clean(os.path.join(resultsBase,scene),"Ta")
    
if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, pycurl.error):
        exit('Received Ctrl + C... Exiting! Bye.', 1)