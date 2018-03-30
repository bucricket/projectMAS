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
from processlai import processlai
from processlst import processlst
import sqlite3
from getlandsatdata import getlandsatdata


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
    parser.add_argument("start_date", type=str, help="Start date yyyy-mm-dd")
    parser.add_argument("end_date", type=str, help="Start date yyyy-mm-dd")
    parser.add_argument("njobs", type=int, default=-1, help="number of cores to use.  To use all cores available use -1")
    parser.add_argument('-s','--sat', nargs='?',type=int, default=8, help='which landsat to search or download, i.e. Landsat 8 = 8')

    args = parser.parse_args()
    isUSA = args.isUSA
    njobs = args.njobs
    loc = [args.lat,args.lon]
    sat = args.sat
    start_date = args.start_date
    end_date = args.end_date
    subsetSize = 200
    base = os.getcwd()
    cacheDir = os.path.abspath(os.path.join(base,os.pardir,"SATELLITE_DATA"))
   
    Folders = folders(base)    
    landsatSR = Folders['landsatSR']
    resultsBase = Folders['resultsBase']
#    landsatTemp = os.path.join(landsatSR,'temp')
#    landsatDataBase = Folders['landsatDataBase']
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
#    fileList = glob.glob(os.path.join(landsatTemp,"*_MTL.txt"))
#    tiffList = glob.glob(os.path.join(landsatTemp,"*lstSharp.tiff"))
    
    
    #===FIND PROCESSED SCENES TO BE PROCESSED================================
    landsatCacheDir = os.path.join(cacheDir,"LANDSAT")
    db_fn = os.path.join(landsatCacheDir,"landsat_products.db")
#    available = 'Y'
    product = 'LST'
#    search_df = getlandsatdata.search(loc[0],loc[1],start_date,end_date,cloud,available,landsatCacheDir,sat)
    search_df = processlst.searchLandsatProductsDB(loc[0],loc[1],start_date,end_date,product,landsatCacheDir)
    productIDs = search_df.LANDSAT_PRODUCT_ID
#    fileList = search_df.local_file_path
    #====check what products are processed against what Landsat data is available===
    product = 'ETd'
    if os.path.exists(db_fn):
        conn = sqlite3.connect( db_fn )
        res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = res.fetchall()[0]
        if (product in tables):
            processedProductIDs = processlst.searchLandsatProductsDB(loc[0],loc[1],start_date,end_date,product,landsatCacheDir)
            df1 = processedProductIDs[["LANDSAT_PRODUCT_ID"]]
            merged = df1.merge(pd.DataFrame(productIDs), indicator=True, how='outer')
            df3 = merged[merged['_merge'] != 'both' ]
            productIDs = df3[["LANDSAT_PRODUCT_ID"]].LANDSAT_PRODUCT_ID
#            if len(productIDs)>0:
#                output_df = pd.DataFrame()
#                for productID in productIDs:
#                    output_df = output_df.append(getlandsatdata.searchProduct(productID,cacheDir,sat),ignore_index=True)
#                productIDs = output_df.LANDSAT_PRODUCT_ID
            
    #USER INPUT END===============================================================
    start = timer.time()
#    for i in range(len(fileList)):
    for productID in productIDs:
        print("productID:%s" % productID)
#        fn = fileList[i]
        out_df = getlandsatdata.searchProduct(productID,landsatCacheDir,sat)
        fn = os.path.join(out_df.local_file_path[0],productID+"_MTL.txt")
        meta = landsat_metadata(fn)
        sceneID = meta.LANDSAT_SCENE_ID
        productID = meta.LANDSAT_PRODUCT_ID
        satscene_path = os.sep.join(fn.split(os.sep)[:-2])
        sceneDir = os.path.join(satscene_path,'LST')
        tiff= os.path.join(sceneDir,'%s_lstSharp.tiff' % sceneID)
#        tiff = tiffList[i]
        g = gdal.Open(tiff)     
        scene = sceneID[3:9]
        sceneDir = os.path.join(satscene_path,'ET','30m')
        if not os.path.exists(sceneDir):
            os.mkdir(sceneDir)
        finalFile = os.path.join(sceneDir,'%s_ETd.tif' % sceneID)
        dt = meta.DATETIME_OBJ
        if not os.path.exists(finalFile):
                   
            #============Run DisALEXI in parallel======================================
            dd = disALEXI(fn,dt,isUSA)
#            #===COMMENTED FOR TESTING ONLY=================== 
            dd.runDisALEXI(0,0,subsetSize,subsetSize,ALEXIgeodict,0)
            print 'Running disALEXI...'
            r = Parallel(n_jobs=njobs, verbose=5)(delayed(dd.runDisALEXI)(xStart,yStart,subsetSize,subsetSize,ALEXIgeodict,0) for xStart in range(0,g.RasterXSize,subsetSize) for yStart in range(0,g.RasterYSize,subsetSize))            
#            
#            # =================merge Ta files============================================
            print("merging Ta files----------------------->")
#
            tifs = glob.glob(os.path.join(resultsBase,scene,'Ta*'))
            finalFileVRT = os.path.join(resultsBase,scene,'Ta_DisALEXI.vrt')
            finalFile = os.path.join(resultsBase,scene,'testTa_DisALEXI.tif')
            outds = gdal.BuildVRT(finalFileVRT, tifs, options=gdal.BuildVRTOptions(srcNodata=-9999.))
            outds = gdal.Translate(finalFile, outds)
            outds = None
#            #=========smooth the TA data=======================================
            print 'Smoothing Ta...'
            dd.smoothTaData(ALEXIgeodict)
#            
##             =================run TSEB one last time in parallel=======================
#            print "run one last time in serial"
#            dd.runDisALEXI(0,0,1135,1135,ALEXIgeodict,1)
            print "run TSEB one last time in parallel"
            r = Parallel(n_jobs=njobs, verbose=5)(delayed(dd.runDisALEXI)(xStart,yStart,subsetSize,subsetSize,ALEXIgeodict,1) for xStart in range(0,g.RasterXSize,subsetSize) for yStart in range(0,g.RasterYSize,subsetSize)) 

            #=====================merge all files =====================================
            finalFile = os.path.join(sceneDir,'%s_ETd.tif' % sceneID[:-5])
            print 'merging ETd files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile,os.path.join(resultsBase,scene,'ETd*'))
            buildvrt(cmd)
            
            #=======================update ETd database========================
#            output_df = processlst.searchLandsatProductsDB(loc[0],loc[1],start_date,end_date,product,landsatCacheDir)
            output_df = getlandsatdata.searchProduct(productID,landsatCacheDir,sat)
            processlai.updateLandsatProductsDB(output_df,finalFile,landsatCacheDir,'ETd')
            
#            #=============find Average ET_24===================================
#            outFormat = gdal.GDT_Float32
#            ls = GeoTIFF(finalFile)
#            ulx = ls.ulx
#            uly = ls.uly
#            delx = ls.delx
#            dely = -ls.dely
#            inUL = [ulx,uly]
#            inRes = [delx,dely]
#            sceneDir = os.path.join(ALEXIbase,'%s' % scene)        
#            etFN = os.path.join(sceneDir,'%s_alexiET.tiff' % sceneID)         
#            g = gdal.Open(etFN,GA_ReadOnly)
#            ET_ALEXI = g.ReadAsArray()
#            g= None
#            et_alexi = np.array(np.reshape(ET_ALEXI,[np.size(ET_ALEXI)])*10000, dtype='int')
#            
#            g = gdal.Open(finalFile,GA_ReadOnly)
#            ET_24 = g.ReadAsArray()
#            g= None
#            
#            et_24 = np.reshape(ET_24,[np.size(ET_24)])
#            etDict = {'ID':et_alexi,'et':et_24}
#            etDF = pd.DataFrame(etDict, columns=etDict.keys())
#            group = etDF['et'].groupby(etDF['ID'])
#            valMean = group.mean()
#            outData = np.zeros(ET_24.size)
#            for i in range(valMean.size):
#                outData[et_alexi==valMean.index[i]]=valMean.iloc[i]
#            et_avg = np.reshape(outData,ET_24.shape)
#            et_diff = abs(et_avg-ET_ALEXI)
#            outET24Path = os.path.join(resultsBase,scene)
#            ET_diff_outName = '%s_ETd_diff.tif' % sceneID[:-5]
#            fName = '%s%s%s' % (outET24Path,os.sep,ET_diff_outName)
#            writeArray2Tiff(et_diff,inRes,inUL,ls.proj4,fName,outFormat)
            
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

            #=======================clean up files=============================
            print 'cleaning up...'
            
            clean(os.path.join(resultsBase,scene),"ETd")
            clean(os.path.join(resultsBase,scene),"Ta")
            clean(os.path.join(resultsBase,scene),"Ts")
            clean(os.path.join(resultsBase,scene),"Tc")
            clean(os.path.join(resultsBase,scene),"G0")
            clean(os.path.join(resultsBase,scene),"H_c")
            clean(os.path.join(resultsBase,scene),"H_s")
            clean(os.path.join(resultsBase,scene),"lETc")
            clean(os.path.join(resultsBase,scene),"lEs")
    end = timer.time()
    print("program duration: %f minutes" % ((end - start)/60.))
    os.removedirs(os.path.join(landsatSR ,'temp'))
#            os.remove(lat_fName)
#            os.remove(lon_fName)
    
if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, pycurl.error):
        exit('Received Ctrl + C... Exiting! Bye.', 1)