#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:30:05 2017

@author: mschull
"""
from .landsatTools import landsat_metadata,GeoTIFF
import os
from datetime import datetime
import subprocess
import numpy as np
import utm
import shutil
from .utils import writeArray2Tiff,warp,interpOverpassHour,folders
from osgeo import gdal
from pydap.client import open_url
   
class Landsat(object):
    def __init__(self, filepath):
        base = os.path.abspath(os.path.join(filepath,os.pardir,os.pardir,os.pardir,
                                            os.pardir,os.pardir))
        Folders = folders(base)    
        self.inputDataBase = Folders['inputDataBase']
        self.landsatLC = Folders['landsatLC']
        self.landsatSR = Folders['landsatSR']
        self.albedoBase = Folders['albedoBase']
        self.sceneID = filepath.split(os.sep)[-1][:21]
        self.scene = self.sceneID[3:9]
        
        meta = landsat_metadata(os.path.join(self.landsatSR, 
                                                          self.scene,'%s_MTL.txt' % self.sceneID))
        ls = GeoTIFF(os.path.join(self.landsatSR, self.scene,'%s_sr_band1.tif' % self.sceneID))
        self.proj4 = ls.proj4
        self.inProj4 = ls.proj4
        self.ulx = meta.CORNER_UL_PROJECTION_X_PRODUCT
        self.uly = meta.CORNER_UL_PROJECTION_Y_PRODUCT
        self.lrx = meta.CORNER_LR_PROJECTION_X_PRODUCT
        self.lry = meta.CORNER_LR_PROJECTION_Y_PRODUCT
        self.ulLat = meta.CORNER_UL_LAT_PRODUCT
        self.ulLon = meta.CORNER_UL_LON_PRODUCT
        self.lrLat = meta.CORNER_LR_LAT_PRODUCT
        self.lrLon = meta.CORNER_LR_LON_PRODUCT
        self.delx = meta.GRID_CELL_SIZE_REFLECTIVE
        self.dely = meta.GRID_CELL_SIZE_REFLECTIVE
        
    def getLC(self,classification):
        scene = self.scene
        sceneID = self.sceneID
        if classification=='NLCD': #NLCD
            outfile = os.path.join(self.landsatLC,'nlcd_2011_landcover_2011_edition_2014_10_10','nlcd_2011_landcover_2011_edition_2014_10_10.img')
            if not os.path.exists(outfile):
                print "download NLCD data at:\n"
                print 'http://www.landfire.gov/bulk/downloadfile.php?TYPE=nlcd2011&FNAME=nlcd_2011_landcover_2011_edition_2014_10_10.zip'
                print "and unzip in %s\n" % self.landsatLC
                raise SystemExit(0)
            self.inProj4 = '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
        else:
            #get UTM info
            utmZone= []
            corners = [[self.ulLat,self.ulLon],[self.ulLat,self.lrLon],
                       [self.lrLat,self.lrLon],[self.lrLat,self.ulLon]]
            for i in xrange(4):
                lat =corners[i][0]
                lon =corners[i][1]
                utmZone.append(utm.from_latlon(lat,lon)[2])
            latNames = [np.floor(self.ulLat/5.)*5,np.floor(self.lrLat/5.)*5.]
            utmzone = np.unique(utmZone)
            latNames = np.unique(latNames)
            tileNames =[]
            for i in xrange(len(utmzone)):
                for j in xrange(len(latNames)):
                    LCdata = os.path.join(self.inputDataBase,'N%d_%d_2010LC030' % (utmzone[i],latNames[j]),'n%d_%d_2010lc030.tif' % (utmZone[i],latNames[j]))
                    tileNames.append(LCdata)
            
            tileNames = np.unique(tileNames)
            
            for i in xrange(len(tileNames)):
                localFN = os.path.join(self.landsatLC,tileNames[i].split(os.sep)[-1])
                LCdata =tileNames[i]
                if os.path.exists(localFN):
                    #print 'file exists already'
                    continue   
                out = subprocess.check_output('rsync -avrz mschull@alexi1.umd.edu:%s %s' % (LCdata,self.landsatLC), shell=True)
                #print out
            #find all LC tiles needed for Landsat image
            
            
            # mosaic dataset if needed
            if len(tileNames)>1:
                inNames = os.path.join(self.landsatLC,'%s' % tileNames[0].split(os.sep)[-1])
                for i in range(1,len(tileNames)):
                    inNames = inNames+' ' + os.path.join(self.landsatLC,'%s' % tileNames[i].split(os.sep)[-1])
            outfile = os.path.join(self.landsatLC,'tempMos.tiff')
            command = 'gdal_merge.py -n 0 -a_nodata 0 -of GTiff -o %s %s' % (outfile,inNames)
            #print command
            #out = subprocess.check_output('gdal_merge.py -n 0 -a_nodata 0 -pct -of GTiff -o %s %s' % (outfile,inNames),shell=True)   
            out = subprocess.check_output('gdal_merge.py -pct -of GTiff -o %s %s' % (outfile,inNames),shell=True)
            #print out
    
        dailyPath = os.path.join(self.landsatLC, '%s' % scene)
        
        if not os.path.exists(dailyPath):
            os.makedirs(dailyPath)
        outfile2=os.path.join(dailyPath,'%s%s' % (sceneID,'_LC.tiff'))
        
        optionList = ['-overwrite', '-s_srs', '%s' % self.inProj4,'-t_srs','%s' % self.proj4,\
        '-te', '%f' % self.ulx, '%f' % self.lry,'%f' % self.lrx,'%f' % self.uly,\
        '-multi','-of','GTiff','%s' % outfile, '%s' % outfile2]
        warp(optionList)

    def getAlbedo(self):
        #print '--------------- process landsat albedo/NDVI -----------------------'
        #yeardoy = sceneID[9:16]
        scene = self.scene
        sceneID = self.sceneID
        #print "scene:%s" % self.scene
        sceneFolderAlbedo = os.path.join(self.albedoBase,scene)
        if not os.path.exists(sceneFolderAlbedo):
            os.makedirs(sceneFolderAlbedo)
        albedoPath = os.path.join(sceneFolderAlbedo,'%s_albedo.tiff' % sceneID)
    
        bands = [1,3,4,5,7] # dont use blue
        # extract the desired surface refelctance bands from landsat
        data = []   
        for i in xrange(len(bands)):
            landsatgeotif = os.path.join(self.landsatSR,scene,'%s_sr_band%d.tif' % (sceneID,bands[i]))
            if os.path.exists(landsatgeotif):
                ls = GeoTIFF(landsatgeotif)
                data.append(ls.data*0.0001)
            else:
                print "no sr file for scenedID: %s, band: %d" % (sceneID,bands[i])
        albedotemp=(0.356*data[0])+(0.130*data[1])+(0.373*data[2])+(0.085*data[3])+(0.072*data[4])-0.0018
        ls.clone(albedoPath,albedotemp)
class ALEXI:
    def __init__(self, filepath):
        base = os.path.abspath(os.path.join(filepath,os.pardir,os.pardir,os.pardir,
                                            os.pardir,os.pardir))
        Folders = folders(base)    
        self.inputDataBase = Folders['inputDataBase']
        self.landsatLC = Folders['landsatLC']
        self.landsatSR = Folders['landsatSR']
        self.ALEXIbase = Folders['ALEXIbase']
        self.sceneID = filepath.split(os.sep)[-1][:21]
        self.scene = self.sceneID[3:9]
        self.yeardoy = self.sceneID[9:16]
        
        meta = landsat_metadata(os.path.join(self.landsatSR, 
                                                          self.scene,'%s_MTL.txt' % self.sceneID))
        ls = GeoTIFF(os.path.join(self.landsatSR, self.scene,'%s_sr_band1.tif' % self.sceneID))
        self.proj4 = ls.proj4
        self.ulx = meta.CORNER_UL_PROJECTION_X_PRODUCT
        self.uly = meta.CORNER_UL_PROJECTION_Y_PRODUCT
        self.lrx = meta.CORNER_LR_PROJECTION_X_PRODUCT
        self.lry = meta.CORNER_LR_PROJECTION_Y_PRODUCT
        self.ulLat = meta.CORNER_UL_LAT_PRODUCT
        self.ulLon = meta.CORNER_UL_LON_PRODUCT
        self.lrLat = meta.CORNER_LR_LAT_PRODUCT
        self.lrLon = meta.CORNER_LR_LON_PRODUCT
        self.delx = meta.GRID_CELL_SIZE_REFLECTIVE
        self.dely = meta.GRID_CELL_SIZE_REFLECTIVE
        self.landsatDate = meta.DATE_ACQUIRED
        self.landsatTime = meta.SCENE_CENTER_TIME[:-2]
        d = datetime.strptime('%s%s' % (self.landsatDate,self.landsatTime),'%Y-%m-%d%H:%M:%S.%f')
        self.hr = d.hour #UTC
        self.year = d.year
        
    def getALEXIdata(self,ALEXIgeodict,isUSA):       
        #ALEXI input info
        ALEXI_ulLon = ALEXIgeodict['ALEXI_ulLon']
        ALEXI_ulLat = ALEXIgeodict['ALEXI_ulLat']
        ALEXILatRes = ALEXIgeodict['ALEXI_LatRes']
        ALEXILonRes = ALEXIgeodict['ALEXI_LonRes']
        ALEXIshape = ALEXIgeodict['ALEXIshape']
        inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        inUL = [ALEXI_ulLon,ALEXI_ulLat]
        inRes = [ALEXILonRes,ALEXILatRes] 
        dailyPath = os.path.join(self.ALEXIbase,'%s' % self.scene)
        if not os.path.exists(dailyPath):
            os.makedirs(dailyPath)
            
        outfile = os.path.join(dailyPath,'%s_alexiET.tiff' % self.sceneID)
        if not os.path.exists(outfile):
            #print 'get->ALEXI ET...'
        #else:
            print 'processing : %s...' % outfile
            subsetFile = outfile[:-5]+'Sub.tiff'
            outFormat = gdal.GDT_Float32
            #********THIS SECTION IS A TEMPERARY FIX
            if isUSA == 0:
                alexiTile = 'T13'
                read_data = np.fromfile(os.path.join(self.ALEXIbase,'%d' % self.year,'%s' % alexiTile ,'FINAL_EDAY_CFSR_%s.dat' % (self.yeardoy)), dtype=np.float32)
            else: # STILL NEED TO DOWNLOAD DATA FROM UMD SERVER
                alexiTile = 'MEAD'
                read_data = np.fromfile(os.path.join(self.ALEXIbase,'%d' % self.year,'%s' % alexiTile ,'FINAL_EDAY_CFSR_FMAX_%s.dat' % (self.yeardoy)), dtype=np.float32) 
            #*******************
            dataset = np.flipud(read_data.reshape([ALEXIshape[1],ALEXIshape[0]]))*0.408 
            writeArray2Tiff(dataset,inRes,inUL,inProj4,outfile,outFormat)
            
            optionList = ['-overwrite', '-s_srs', '%s' % inProj4,'-t_srs','%s' % self.proj4,\
            '-te', '%f' % self.ulx, '%f' % self.lry,'%f' % self.lrx,'%f' % self.uly,'-r', 'near',\
            '-tr', '%f' % self.delx, '%f' % self.dely,'-multi','-of','GTiff','%s' % outfile, '%s' % subsetFile]
            warp(optionList)
class MET:
    def __init__(self, filepath,session):
        base = os.path.abspath(os.path.join(filepath,os.pardir,os.pardir,os.pardir,
                                            os.pardir,os.pardir))
        print base
        Folders = folders(base)    
        self.session = session
        self.inputDataBase = Folders['inputDataBase']
        self.landsatLC = Folders['landsatLC']
        self.landsatSR = Folders['landsatSR']
        self.metBase = Folders['metBase']
        self.sceneID = filepath.split(os.sep)[-1][:21]
        self.scene = self.sceneID[3:9]
        self.yeardoy = self.sceneID[9:16]
        
        meta = landsat_metadata(os.path.join(self.landsatSR, 
                                                          self.scene,'%s_MTL.txt' % self.sceneID))
        ls = GeoTIFF(os.path.join(self.landsatSR, self.scene,'%s_sr_band1.tif' % self.sceneID))
        self.proj4 = ls.proj4
        self.ulx = meta.CORNER_UL_PROJECTION_X_PRODUCT
        self.uly = meta.CORNER_UL_PROJECTION_Y_PRODUCT
        self.lrx = meta.CORNER_LR_PROJECTION_X_PRODUCT
        self.lry = meta.CORNER_LR_PROJECTION_Y_PRODUCT
        self.ulLat = meta.CORNER_UL_LAT_PRODUCT
        self.ulLon = meta.CORNER_UL_LON_PRODUCT
        self.lrLat = meta.CORNER_LR_LAT_PRODUCT
        self.lrLon = meta.CORNER_LR_LON_PRODUCT
        self.delx = meta.GRID_CELL_SIZE_REFLECTIVE
        self.dely = meta.GRID_CELL_SIZE_REFLECTIVE
        self.landsatDate = meta.DATE_ACQUIRED
        self.landsatTime = meta.SCENE_CENTER_TIME[:-2]
        d = datetime.strptime('%s%s' % (self.landsatDate,self.landsatTime),'%Y-%m-%d%H:%M:%S.%f')
        self.year = d.year
        self.month = d.month
        self.day = d.day
        self.hr = d.hour #UTC
        
    def getMetData(self):
        
        # set Met dataset geo info
        CFSR_ulLat = 90.25
        CFSR_ulLon = -180.25
        CFSRLatRes = 0.25
        CFSRLonRes = 0.25
        inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

        dailyPath = os.path.join(self.metBase,'%s' % self.yeardoy)
        dailyInputPath = os.path.join(self.inputDataBase,'output%s' % self.year,'%s' % self.yeardoy)
        if not os.path.exists(dailyPath):
            os.makedirs(dailyPath)
        
        fileNames = ['psfc_series.bin.gz','q2_series.bin.gz','t2_series.bin.gz','wind_surface.bin.gz']    
        outNames = ['p','q2','u']
        inRes = [CFSRLonRes,CFSRLatRes]
        inUL = [CFSR_ulLon,CFSR_ulLat]
        for i in xrange(len(outNames)):
            outfile = os.path.join(dailyPath,'%s_%s.tiff' % (self.sceneID,outNames[i]))
            if os.path.exists(outfile):
                print ' ' 
            else:
                print 'processing : %s...' % outfile
                gzipFile=os.path.join(dailyPath,fileNames[i])
                shutil.copyfile(os.path.join(dailyInputPath,fileNames[i]),gzipFile)
                out = subprocess.check_output('gzip -df %s' % gzipFile, shell=True)
                
                read_data = np.fromfile(os.path.join(dailyPath,fileNames[i][:-3]), dtype=np.float32)
                dataset = np.reshape(read_data.reshape([8,600,1440]),[8,600*1440])
        
                dataSET = interpOverpassHour(dataset,self.hr)
                dataset2 = np.flipud(dataSET.reshape(600,1440))
                outfile = os.path.join(dailyPath,'%s_%s.tiff' % (self.sceneID,outNames[i]))
                outFormat = gdal.GDT_Float32
                writeArray2Tiff(dataset2,inRes,inUL,inProj4,outfile,outFormat)
                subsetFile = outfile[:-5]+'Sub.tiff'
                optionList = ['-overwrite', '-s_srs', '%s' % inProj4,'-t_srs','%s' % self.proj4,\
                '-te', '%f' % self.ulx, '%f' % self.lry,'%f' % self.lrx,'%f' % self.uly,'-r', 'bilinear',\
                '-tr', '%f' % self.delx, '%f' % self.dely ,'-multi','-of','GTiff','%s' % outfile, '%s' % subsetFile]
                warp(optionList)
                
    def getCFSR(self):
        
        # set Met dataset geo info
        CFSR_ulLat = 90.0
        CFSR_ulLon = -180.0
        CFSRLatRes = 0.50
        CFSRLonRes = 0.50
        inRes = [CFSRLonRes,CFSRLatRes]
        inUL = [CFSR_ulLon,CFSR_ulLat]
        inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        #===Open CFSR file=================
        dailyPath = os.path.join(self.metBase,'%s' % self.scene)
        ncdcURL = 'https://nomads.ncdc.noaa.gov/thredds/dodsC/modeldata/cfsv2_analysis_pgbh/'       
        iHour = (int(self.hr/6.)*6.)
        fHour = self.hr-iHour
        hr1file = 'cdas1.t%02dz.pgrbh%02d.grib2' % (iHour,fHour)
        pydapURL = os.path.join(ncdcURL,"%s" % self.year,"%d%02d" % (self.year,self.month),"%d%02d%02d" % (self.year,self.month,self.day),hr1file)
        print pydapURL
        #=====Surface Pressure=============
        d = open_url(pydapURL)
        data = d["Pressure_surface"][:,:,:]
        opdata = data.data[0][0]
        #---flip data so ul -180,90-----------      
        b = opdata[:,:360]
        c = opdata[:,360:]
        flipped = np.hstack((c,b))
        print np.shape(flipped)
        #----write out global dataset to gTiff--------
        outfile = os.path.join(dailyPath,'%s_p.tiff' % (self.sceneID))
        outFormat = gdal.GDT_Float32
        writeArray2Tiff(flipped.astype(float),inRes,inUL,inProj4,outfile,outFormat)
        
        subsetFile = outfile[:-5]+'Sub.tiff'
        optionList = ['-overwrite', '-s_srs', '%s' % inProj4,'-t_srs','%s' % self.proj4,\
                '-te', '%f' % self.ulx, '%f' % self.lry,'%f' % self.lrx,'%f' % self.uly,'-r', 'bilinear',\
                '-tr', '%f' % self.delx, '%f' % self.dely ,'-multi','-of','GTiff','%s' % outfile, '%s' % subsetFile]
        warp(optionList)       
        #====Wind Speed================
        
        Udata = d["U-component_of_wind_sigma"][:,:,:]
        Uopdata = Udata.data[0][0]
        
        Vdata = d["V-component_of_wind_sigma"][:,:,:]
        Vopdata = Vdata.data[0][0]
        
        wind = np.sqrt(Uopdata**2+Vopdata**2)
        #---flip data so ul -180,90-----------      
        b = wind[:,:360]
        c = wind[:,360:]
        flipped = np.hstack((c,b))
        #----write out global dataset to gTiff--------
        
        outfile = os.path.join(dailyPath,'%s_u.tiff' % (self.sceneID))
        #outfile = 'u05NCDC.tif'
        outFormat = gdal.GDT_Float32
        writeArray2Tiff(flipped.astype(float),inRes,inUL,inProj4,outfile,outFormat)
        
        subsetFile = outfile[:-5]+'Sub.tiff'
        optionList = ['-overwrite', '-s_srs', '%s' % inProj4,'-t_srs','%s' % self.proj4,\
                '-te', '%f' % self.ulx, '%f' % self.lry,'%f' % self.lrx,'%f' % self.uly,'-r', 'bilinear',\
                '-tr', '%f' % self.delx, '%f' % self.dely ,'-multi','-of','GTiff','%s' % outfile, '%s' % subsetFile]
        
        warp(optionList)
        
        #===== Specific humidity =============
        d = open_url(pydapURL)
        data = d["Specific_humidity_hybrid"][:,:,:]
        opdata = data.data[0][0][0]
        #---flip data so ul -180,90-----------      
        b = opdata[:,:360]
        c = opdata[:,360:]
        flipped = np.hstack((c,b))
        #----write out global dataset to gTiff--------
        
        outfile = os.path.join(dailyPath,'%s_q2.tiff' % (self.sceneID))
        #outfile = 'q205NCDC.tif'
        outFormat = gdal.GDT_Float32
        writeArray2Tiff(flipped.astype(float),inRes,inUL,inProj4,outfile,outFormat)
        
        subsetFile = outfile[:-5]+'Sub.tiff'
        optionList = ['-overwrite', '-s_srs', '%s' % inProj4,'-t_srs','%s' % self.proj4,\
                '-te', '%f' % self.ulx, '%f' % self.lry,'%f' % self.lrx,'%f' % self.uly,'-r', 'bilinear',\
                '-tr', '%f' % self.delx, '%f' % self.dely ,'-multi','-of','GTiff','%s' % outfile, '%s' % subsetFile]
        warp(optionList)     
        
        #===== Temperature 2m =============
        d = open_url(pydapURL)
        data = d["Temperature_sigma"][:,:,:]
        opdata = data.data[0][0]
        #---flip data so ul -180,90-----------      
        b = opdata[:,:360]
        c = opdata[:,360:]
        flipped = np.hstack((c,b))
        #----write out global dataset to gTiff--------
        
        outfile = os.path.join(dailyPath,'%s_Ta.tiff' % (self.sceneID))
        #outfile = 'q205NCDC.tif'
        outFormat = gdal.GDT_Float32
        writeArray2Tiff(flipped.astype(float),inRes,inUL,inProj4,outfile,outFormat)
        
        subsetFile = outfile[:-5]+'Sub.tiff'
        optionList = ['-overwrite', '-s_srs', '%s' % inProj4,'-t_srs','%s' % self.proj4,\
                '-te', '%f' % self.ulx, '%f' % self.lry,'%f' % self.lrx,'%f' % self.uly,'-r', 'bilinear',\
                '-tr', '%f' % self.delx, '%f' % self.dely ,'-multi','-of','GTiff','%s' % outfile, '%s' % subsetFile]
        warp(optionList) 
                
    def getInsolation(self):
        MERRA2_ulLat = 90.0
        MERRA2_ulLon = -180.0
        MERRA2LatRes = 0.5
        MERRA2LonRes = 0.625
        inProj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        #========get MERRA2 Insolation data at overpass time====================== 
        inRes = [MERRA2LonRes,MERRA2LatRes]
        inUL = [MERRA2_ulLon,MERRA2_ulLat]
    
        if self.year <1992:
            fileType = 100
        elif self.year >1991 and self.year < 2001:
            fileType=200
        elif self.year > 2000 and self.year<2011:
            fileType = 300
        else:
            fileType = 400
    
        opendap_url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/'
        product = 'M2T1NXRAD.5.12.4'
        
        filename = 'MERRA2_%d.tavg1_2d_rad_Nx.%04d%02d%02d.nc4' % (fileType,self.year,self.month,self.day)
        fullUrl =os.path.join(opendap_url,product,'%04d'% self.year,'%02d'% self.month,filename)
    
        dailyPath = os.path.join(self.metBase,'%s' % self.scene)
        if not os.path.exists(dailyPath):
            os.makedirs(dailyPath)
        #====get overpass hour insolation=========================================
        outFN = os.path.join(dailyPath,'%s_Insol1Sub.tiff' % self.sceneID) 
        if not os.path.exists(outFN):
            #print ''
        #else:
            print 'processing : %s...' % outFN
            d = open_url(fullUrl,session=self.session)
            Insol = d.SWGDNCLR
    
        # wv_mmr = 1.e-6 * wv_ppmv_layer * (Rair / Rwater)
        # wv_mmr in kg/kg, Rair = 287.0, Rwater = 461.5
            dataset = np.squeeze(Insol[self.hr,:,:,:])*1.
            
            outfile = os.path.join(dailyPath,'%s_%s.tiff' % (self.sceneID,'Insol1'))
            subsetFile = outfile[:-5]+'Sub.tiff'
            outFormat = gdal.GDT_Float32
            writeArray2Tiff(dataset,inRes,inUL,inProj4,outfile,outFormat)
            optionList = ['-overwrite', '-s_srs', '%s' % inProj4,'-t_srs','%s' % self.proj4,\
            '-te', '%f' % self.ulx, '%f' % self.lry,'%f' % self.lrx,'%f' % self.uly,'-r', 'bilinear',\
            '-tr', '%f' % self.delx, '%f' % self.dely,'-multi','-of','GTiff','%s' % outfile, '%s' % subsetFile]
            warp(optionList)
            
                #====get daily insolation=========================================
        outFN = os.path.join(dailyPath,'%s_Insol24Sub.tiff' % self.sceneID)
        if not os.path.exists(outFN):
            #print ''
        #else:
            print 'processing : %s...' % outFN
            dataset2 = np.flipud(np.sum(np.squeeze(Insol[:,:,:]),axis=0))
            outfile = os.path.join(dailyPath,'%s_%s.tiff' % (self.sceneID,'Insol24'))
            subsetFile = outfile[:-5]+'Sub.tiff'
            outFormat = gdal.GDT_Float32
            writeArray2Tiff(dataset2,inRes,inUL,inProj4,outfile,outFormat)
            optionList = ['-overwrite', '-s_srs', '%s' % inProj4,'-t_srs','%s' % self.proj4,\
            '-te', '%f' % self.ulx, '%f' % self.lry,'%f' % self.lrx,'%f' % self.uly,'-r', 'bilinear',\
            '-tr', '%f' % self.delx, '%f' % self.dely,'-multi','-of','GTiff','%s' % outfile, '%s' % subsetFile]
            warp(optionList)