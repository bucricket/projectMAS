#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:39:37 2016

@author: mschull
"""
import os
import numpy as np
from landsat.search import Search
import warnings
from .disalexi import disALEXI
import keyring
import getpass
import argparse
from .utils import buildvrt,clean,folders
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
    parser.add_argument("base", type=str, help="project base directory")
    parser.add_argument("lat", type=float, help="latitude")
    parser.add_argument("lon", type=float, help="longitude")
    parser.add_argument("isUSA", type=float, help="USA=1, non-USA=0")
    parser.add_argument("startDate", type=str, help="Start date yyyy-mm-dd")
    parser.add_argument("endDate", type=str, help="Start date yyyy-mm-dd")
    args = parser.parse_args()
    GCP = [args.lat,args.lon] 
    startDate = args.startDate
    endDate = args.endDate
    isUSA = args.isUSA
    base = args.base
    #base = os.getcwd()
#    if platform.uname()[1].split('.')[0][:3]=='rhw':
#        base = os.path.join(os.sep,'data','smcd4','mschull','pyDisALEXI') #SMCD4   
#    elif platform.uname()[1].split('.')[0]=='alexi1':
#        base = os.path.join(os.sep,'home','mschull','pyDisALEXI') #alexi1
#    else:        
#        base = os.path.join(os.sep,'Users','mschull','umdGD','pyDisALEXI')#mac
       
    
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

    isUSA=0
    if isUSA==1:
            ####****Mead 1 km ******######
        ALEXI_ulLon = -105.0
        ALEXI_ulLat = 45.0
        ALEXILatRes = 0.01
        ALEXILonRes = 0.01
        ALEXIshape = [1100,500]
        ALEXIgeodict ={'ALEXI_ulLat':ALEXI_ulLat,'ALEXI_ulLon':ALEXI_ulLon,
                       'ALEXI_LatRes':ALEXILatRes,'ALEXI_LonRes':ALEXILonRes,
                       'ALEXIshape': ALEXIshape}
    else:
        ####****Egypt******######
        ALEXI_ulLon = 30.0
        ALEXI_ulLat = 40.0
        ALEXILatRes = 0.004
        ALEXILonRes = 0.004
        ALEXIshape = [2500,2500]
        ALEXIgeodict ={'ALEXI_ulLat':ALEXI_ulLat,'ALEXI_ulLon':ALEXI_ulLon,
                       'ALEXI_LatRes':ALEXILatRes,'ALEXI_LonRes':ALEXILonRes,
                       'ALEXIshape': ALEXIshape}
    
    #find the scenes
    s = Search()
    scenes = s.search(lat=GCP[0],lon=GCP[1],limit = 100, start_date = startDate,end_date=endDate, cloud_max=5)
    
#    pathrow = []
#    for i in xrange(len(scenes['results'])):        
#        path = np.str(scenes['results'][i]['path'])
#        row = np.str(scenes['results'][i]['row'])
#        pathrow.append('%02s%02s' % (path,row))
#    pathrow = np.unique(pathrow)
#    
#    sceneIDSRlist = []
#    sceneIDLSTlist = [] 
#    for i in xrange(len(pathrow)):      
#        sceneIDLSTlist.extend(glob.glob(os.path.join(landsatDataBase,'LST','LC8%s%d*' % (pathrow[i],year))))
#        sceneIDSRlist.extend(glob.glob(os.path.join(landsatSR,'LC8%s%d*' % (pathrow[i],year))))
#        
#    sceneIDLST = []
#    for i in xrange(len(sceneIDLSTlist)):
#        sceneIDLST.append(sceneIDLSTlist[i].split(os.sep)[-1][:-9])
#        
#    sceneIDSR = []
#    for i in xrange(len(sceneIDSRlist)):
#        sceneIDSR.append(sceneIDSRlist[i].split(os.sep)[-1]+'LGN00')
#    sceneIDlist = list(set(sceneIDSR) & set(sceneIDLST))
    #USER INPUT END===============================================================
    #%%
#    isUSA = 0
    sceneID = str(scenes['results'][0]['sceneID'])
    #sceneID = 'LC81760392015221LGN00'
    scene = sceneID[3:9]
    
    
    #============Run DisALEXI in parallel======================================
    #print 'run DisALEXI once to avoid huge overhead issues in parallel runs'
    fn = os.path.join(landsatSR,scene,"%s.xml" % sceneID)
    dd = disALEXI(fn,session)
    dd.runDisALEXI(0,0,fn,isUSA,ALEXIgeodict,0)
    
    print 'Running disALEXI...'
    r = Parallel(n_jobs=-1, verbose=5)(delayed(dd.runDisALEXI)(xStart,yStart,fn,isUSA,ALEXIgeodict,0) for xStart in range(0,7600,200) for yStart in range(0,7600,200))            
    
    # =================merge Ta files============================================
    print 'merging Ta files...'
    
    finalFile = os.path.join(resultsBase,scene,'Taxxxxx.tif')
    cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'Ta*'))
    buildvrt(cmd)
    
    # =================run TSEB one last time in parallel=======================
    #print 'run DisALEXI once to avoid huge overhead issues in parallel runs'
    dd.runDisALEXI(0,0,fn,isUSA,ALEXIgeodict,1)
    print "run TSEB one last time in parallel"
    r = Parallel(n_jobs=-1, verbose=5)(delayed(dd.runDisALEXI)(xStart,yStart,fn,isUSA,ALEXIgeodict,1) for xStart in range(0,7600,200) for yStart in range(0,7600,200)) 
    #=====================merge all files =====================================
    print 'merging ETd files...'
    finalFile = os.path.join(resultsBase,scene,'%s_ETd.tif' % sceneID[:-5])
    cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'ETd*'))
    buildvrt(cmd)
    
    #=======================clean up files===================================
    print 'cleaning up...'
    
    clean(os.path.join(resultsBase,scene),"ETd*")
    clean(os.path.join(resultsBase,scene),"Ta*")
    
if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, pycurl.error):
        exit('Received Ctrl + C... Exiting! Bye.', 1)