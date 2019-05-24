#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part PyDisALEXI, consisting of of high level PyDisALEXI scripting
# Copyright 2018 Mitchell A. Schull and contributors listed in the README.md file.
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

"""
Created on 
@author: Mitch Schull (mschull@umd.edu)

Modified on 
@author: Mitch Schull (mschull@umd.edu)

DESCRIPTION
===========


"""
import os
import numpy as np
import warnings
from .disalexi_usda import disALEXI
import argparse
import glob
from .utils import writeArray2Tiff, buildvrt, clean, folders
from joblib import Parallel, delayed
import types
import copy_reg
from .landsatTools import landsat_metadata, GeoTIFF
import time as timer
from osgeo import gdal
from .database_tools import *
from pyproj import Proj
import logging
import sys

warnings.simplefilter('ignore', np.RankWarning)


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copy_reg.pickle(types.MethodType, _pickle_method)


def search(lat, lon, start_date, end_date, cloud, cacheDir, sat):
    columns = ['acquisitionDate', 'acquisitionDate', 'upperLeftCornerLatitude', 'upperLeftCornerLongitude',
               'lowerRightCornerLatitude', 'lowerRightCornerLongitude', 'cloudCover', 'sensor', 'LANDSAT_PRODUCT_ID']
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    # this is a landsat-util work around when it fails
    if sat == 7:
        metadataUrl = 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_ETM_C1.csv'
    else:
        metadataUrl = 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_8_C1.csv'

    fn = os.path.join(cacheDir, metadataUrl.split(os.sep)[-1])
    # looking to see if metadata CSV is available and if its up to the date needed
    if os.path.exists(fn):
        d = datetime.datetime.fromtimestamp(os.path.getmtime(fn))
        if (end.year > d.year) and (end.month > d.month) and (end.day > d.day):
            wget.download(metadataUrl, out=fn)
            df = pd.read_csv(fn, usecols=columns)
            df.to_csv(fn)
        df = pd.read_csv(fn)
        index = ((df.acquisitionDate >= start_date) & (df.acquisitionDate < end_date) & (
                df.upperLeftCornerLatitude > lat) & (df.upperLeftCornerLongitude < lon) & (
                         df.lowerRightCornerLatitude < lat) & (df.lowerRightCornerLongitude > lon) & (
                         df.cloudCover <= cloud) & (df.sensor == 'OLI_TIRS'))
        df = df[index]

    else:
        wget.download(metadataUrl, out=fn)
        df = pd.read_csv(fn, usecols=columns)
        df.to_csv(fn)
        index = ((df.acquisitionDate >= start_date) & (df.acquisitionDate < end_date) & (
                df.upperLeftCornerLatitude > lat) & (df.upperLeftCornerLongitude < lon) & (
                         df.lowerRightCornerLatitude < lat) & (df.lowerRightCornerLongitude > lon) & (
                         df.cloudCover <= cloud) & (df.sensor == 'OLI_TIRS'))
        df = df[index]

    return df


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def find_already_downloaded(df, cache_dir):
    usgs_available = list(df.LANDSAT_PRODUCT_ID.values)
    # find sat
    sat = usgs_available[0].split("_")[0][-1]
    # find scenes
    scenes = [x.split("_")[2] for x in usgs_available]
    scenes = list(set(scenes))
    available_list = []
    for scene in scenes:
        path_to_search = os.path.join(cache_dir, 'L%s/%s/RAW_DATA/*MTL*' % (sat, scene))
        available = [os.path.basename(x) for x in
                     glob.glob(path_to_search)]
        available = [x[:-8] for x in available]
        available_list = available_list + available
    return intersection(usgs_available, available_list)


def find_not_processed(downloaded, cache_dir):
    """finds the files that are downloaded but still need to process ALBEDO data and thus the rest of the inputs"""
    # find sat
    sat = downloaded[0].split("_")[0][-1]
    # find scenes
    scenes = [x.split("_")[2] for x in downloaded]
    scenes = list(set(scenes))
    available_list = []
    for scene in scenes:
        path_to_search = os.path.join(cache_dir, 'L%s/%s/ALBEDO/*_l.tif' % (sat, scene))
        available = [os.path.basename(x) for x in
                     glob.glob(path_to_search)]
        available = [x[:-8] for x in available]
        available_list = available_list + available
    for x in available_list:
        if x in downloaded:
            downloaded.remove(x)
    return downloaded


def main():
    """ This is the main function for the PyDisALEXI program """
    args = arg_parse()
    lat = args.lat
    lon = args.lon
    is_usa = args.is_usa
    start_date = args.start_date
    end_date = args.end_date
    n_jobs = args.n_jobs
    sample_size = args.sample_size
    sat = args.sat

    loc = [lat, lon]
    subset_size = 200
    base = os.getcwd()
    cache_dir = os.path.abspath(os.path.join(base, "SATELLITE_DATA"))

    Folders = folders(base)
    landsatSR = Folders['landsatSR']
    resultsBase = Folders['resultsBase']

    # ======FIND AVAILABLE FILES FOR PROCESSING=============================

    ####****Global******######
    # ---placeholders-----
    ALEXI_ulLon = 0.0
    ALEXI_ulLat = 0.0
    # -------------------- 375 M
    #    ALEXILatRes = 0.004
    #    ALEXILonRes = 0.004
    # ---------------------roughly 4 KM  FOR TESTING
    ALEXILatRes = 0.04
    ALEXILonRes = 0.04

    ALEXIshape = [3750, 3750]
    ALEXIgeodict = {'ALEXI_ulLat': ALEXI_ulLat, 'ALEXI_ulLon': ALEXI_ulLon,
                    'ALEXI_LatRes': ALEXILatRes, 'ALEXI_LonRes': ALEXILonRes,
                    'ALEXIshape': ALEXIshape}

    # ===FIND PROCESSED SCENES TO BE PROCESSED================================
    landsatCacheDir = os.path.join(cache_dir, "LANDSAT")
    # ====check what products are processed against what Landsat data is available===
    cloud = 5  # FIX THIS LATER!
    output_df = search(loc[0], loc[1], start_date, end_date, cloud, landsatCacheDir, sat)
    downloaded = find_already_downloaded(output_df, landsatCacheDir)
    productIDs = find_not_processed(downloaded, landsatCacheDir)

    # USER INPUT END===============================================================
    start = timer.time()
    #    for i in range(len(fileList)):
    for productID in productIDs:
        print("productID:%s" % productID)
        sat_str = productID.split("_")[0][-1]
        scene = productID.split("_")[2]
        path = os.path.join(landsatCacheDir, 'L%s/%s/RAW_DATA/' % (sat_str, scene))
        fn = os.path.join(path, productID + "_MTL.txt")
        # fn = os.path.join(out_df.local_file_path[0], productID + "_MTL.txt")
        meta = landsat_metadata(fn)
        sceneID = meta.LANDSAT_SCENE_ID
        productID = meta.LANDSAT_PRODUCT_ID
        satscene_path = os.sep.join(fn.split(os.sep)[:-2])
        sceneDir = os.path.join(satscene_path, 'LST')
        tiff = os.path.join(sceneDir, '%s_lstSharp.tiff' % sceneID)
        #        tiff = tiffList[i]
        ls = GeoTIFF(tiff)
        lrx = ls.lrx
        lry = ls.lry
        ulx = ls.ulx
        uly = ls.uly
        g = gdal.Open(tiff)

        scene = sceneID[3:9]
        sceneDir = os.path.join(satscene_path, 'ET', '30m')
        if not os.path.exists(sceneDir):
            os.mkdir(sceneDir)
        finalFile = os.path.join(sceneDir, '%s_ETd.tif' % sceneID)
        dt = meta.DATETIME_OBJ

        # to subset data
        if sample_size is None:
            start_x_loc = 0
            x_size = g.RasterXSize
            start_y_loc = 0
            y_size = g.RasterYSize
        else:
            # =======find subset========
            myProj = Proj(ls.proj4)
            UTMx, UTMy = myProj(loc[1], loc[0])
            start_utm_x = UTMx - (sample_size / 2)
            start_utm_y = UTMy + (sample_size / 2)
            end_utm_x = start_utm_x + sample_size
            end_utm_y = start_utm_y - sample_size
            point_x, point_y = ls.xy2ij(start_utm_x, start_utm_y)
            point_x_end, point_y_end = ls.xy2ij(end_utm_x, end_utm_y)
            print("UTM_x: %d" % start_utm_x)
            print("UTM_y: %d" % start_utm_y)
            print("point_x: %d" % point_x)
            print("point_y: %d" % point_y)

            print("end_utm_x: %d" % end_utm_x)
            print("end_utm_y: %d" % end_utm_y)
            print("point_x_end: %d" % point_x_end)
            print("point_y_end: %d" % point_y_end)
            start_x_loc = point_x
            x_size = abs(point_x - point_x_end)
            start_y_loc = point_y
            y_size = abs(point_y - point_y_end)
            if x_size < subset_size:
                subset_size = x_size + 1
            if y_size < subset_size:
                subset_size = y_size + 1
        # create coordinates if it is needed
        dd = disALEXI(fn, dt, is_usa)
        dd.create_coordinates()
        if not os.path.exists(finalFile):

            # ============Run DisALEXI in parallel======================================
            # ===COMMENTED FOR TESTING ONLY===================
            # dd.runDisALEXI(0, 0, subset_size, subset_size, 0)
            print('Running disALEXI...')
            r = Parallel(n_jobs=n_jobs, verbose=5, prefer="threads")(
                delayed(dd.runDisALEXI)(xStart, yStart, subset_size, subset_size, 0) for xStart in
                range(start_x_loc, start_x_loc + x_size, subset_size) for yStart in
                range(start_y_loc, start_y_loc + y_size, subset_size))
            #
            # =================merge Ta files============================================
            print("merging Ta files----------------------->")
            # TODO: make Ta subset fit into the Landsat scene so we can use the whole scene later
            tifs = glob.glob(os.path.join(resultsBase, scene, 'Ta*'))
            finalFileVRT = os.path.join(resultsBase, scene, 'Ta_DisALEXI.vrt')
            finalFile = os.path.join(resultsBase, scene, 'Ta_DisALEXI.tif')
            outds = gdal.BuildVRT(finalFileVRT, tifs, options=gdal.BuildVRTOptions(srcNodata=-9999.))
            outputBounds = [ulx, uly, lrx, lry]
            outds = gdal.Translate(finalFile, outds, options=gdal.TranslateOptions(outputBounds=outputBounds))
            outds = None
            # =========smooth the TA data=======================================
            print 'Smoothing Ta...'
            dd.smoothTaData(ALEXIgeodict)
            #
            # =================run TSEB one last time in parallel=======================
            print "run TSEB one last time in parallel"
            r = Parallel(n_jobs=n_jobs, verbose=5, prefer="threads")(
                delayed(dd.runDisALEXI)(xStart, yStart, subset_size, subset_size, 1) for xStart in
                range(start_x_loc, start_x_loc + x_size, subset_size) for yStart in
                range(start_y_loc, start_y_loc + y_size, subset_size))

            # =====================merge all files =====================================
            finalFile = os.path.join(sceneDir, '%s_ETd.tif' % sceneID[:-5])
            print 'merging ETd files...'
            tifs = glob.glob(os.path.join(resultsBase, scene, 'ETd*'))
            finalFileVRT = os.path.join(resultsBase, scene, 'ETd_DisALEXI.vrt')
            outds = gdal.BuildVRT(finalFileVRT, tifs, options=gdal.BuildVRTOptions(srcNodata=-9999.))
            outds = gdal.Translate(finalFile, outds)
            outds = None
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

            finalFile = os.path.join(resultsBase, scene, '%s_G0.tif' % sceneID[:-5])
            print 'merging G0 files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile, os.path.join(resultsBase, scene, 'G0*'))
            buildvrt(cmd)

            finalFile = os.path.join(resultsBase, scene, '%s_H_c.tif' % sceneID[:-5])
            print 'merging H_c files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile, os.path.join(resultsBase, scene, 'H_c*'))
            buildvrt(cmd)

            finalFile = os.path.join(resultsBase, scene, '%s_H_s.tif' % sceneID[:-5])
            print('merging H_s files...')
            cmd = 'gdal_merge.py -o %s %s' % (finalFile, os.path.join(resultsBase, scene, 'H_s*'))
            buildvrt(cmd)

            finalFile = os.path.join(resultsBase, scene, '%s_Tac.tif' % sceneID[:-5])
            print 'merging Tac files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile, os.path.join(resultsBase, scene, 'Tac*'))
            buildvrt(cmd)

            finalFile = os.path.join(resultsBase, scene, '%s_Tc.tif' % sceneID[:-5])
            print 'merging Tc files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile, os.path.join(resultsBase, scene, 'Tc*'))
            buildvrt(cmd)

            finalFile = os.path.join(resultsBase, scene, '%s_Ts.tif' % sceneID[:-5])
            print 'merging Ts files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile, os.path.join(resultsBase, scene, 'Ts*'))
            buildvrt(cmd)

            finalFile = os.path.join(resultsBase, scene, '%s_lETc.tif' % sceneID[:-5])
            print 'merging lETc files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile, os.path.join(resultsBase, scene, 'lETc*'))
            buildvrt(cmd)

            finalFile = os.path.join(resultsBase, scene, '%s_lEs.tif' % sceneID[:-5])
            print 'merging lEs files...'
            cmd = 'gdal_merge.py -o %s %s' % (finalFile, os.path.join(resultsBase, scene, 'lEs*'))
            buildvrt(cmd)

            # =======================clean up files=============================
            print 'cleaning up...'

            clean(os.path.join(resultsBase, scene), "ETd")
            clean(os.path.join(resultsBase, scene), "Ta")
            clean(os.path.join(resultsBase, scene), "Ts")
            clean(os.path.join(resultsBase, scene), "Tc")
            clean(os.path.join(resultsBase, scene), "G0")
            clean(os.path.join(resultsBase, scene), "H_c")
            clean(os.path.join(resultsBase, scene), "H_s")
            clean(os.path.join(resultsBase, scene), "lETc")
            clean(os.path.join(resultsBase, scene), "lEs")
            # ===remove all files in temp folder================================
            folder = os.path.join(landsatSR, 'temp')
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    # elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
    #            shutil.rmtree(os.path.join(landsatSR ,'temp'),ignore_errors=True)
    end = timer.time()
    print("program duration: %f minutes" % ((end - start) / 60.))


def arg_parse():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("lat", type=float, help="latitude")
    parser.add_argument("lon", type=float, help="longitude")
    parser.add_argument("is_usa", type=float, help="USA=1, non-USA=0")
    parser.add_argument("start_date", type=str, help="Start date yyyy-mm-dd")
    parser.add_argument("end_date", type=str, help="Start date yyyy-mm-dd")
    parser.add_argument("n_jobs", type=int, default=-1,
                        help="number of cores to use.  To use all cores available use -1")
    parser.add_argument("-ss", "--sample_size", type=int, default=None,
                        help="Square size in meters ex. 1000 for a 1000 x 1000 plot")
    parser.add_argument('-s', '--sat', nargs='?', type=int, default=8,
                        help='which landsat to search or download, i.e. Landsat 8 = 8')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{0}'.format('#' * 80))
    logging.info('{0:<20s} {1}'.format(
        'Run Time Stamp:', datetime.datetime.now().isoformat(' ')))
    logging.info('{0:<20s} {1}'.format('Current Directory:', os.getcwd()))
    logging.info('{0:<20s} {1}'.format(
        'Script:', os.path.basename(sys.argv[0])))
    try:
        # main(year=args.year, start_doy=args.start_doy, end_doy=args.end_doy, region=args.region)
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(0)
