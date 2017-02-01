#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:03:17 2017

@author: mschull
"""
from __future__ import division, print_function, absolute_import
__author__ = 'jwely'
__all__ = ["landsat_metadata"]

# standard imports

import os.path
import numpy as np
import logging
from .utils import RasterError,_test_outside
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger('pydisalexi.geotiff')

from osgeo import gdal, osr
try:
    from pyproj import Proj
except ImportError:
    LOGGER.warning(
        "PROJ4 is not available. " +
        "Any method requiring coordinate transform will fail.")
from datetime import datetime
import inspect


class landsat_metadata:
    """
    A landsat metadata object. This class builds is attributes
    from the names of each tag in the xml formatted .MTL files that
    come with landsat data. So, any tag that appears in the MTL file
    will populate as an attribute of landsat_metadata.

    You can access explore these attributes by using, for example

    .. code-block:: python

        from dnppy import landsat
        meta = landsat.landsat_metadata(my_filepath) # create object

        from pprint import pprint                    # import pprint
        pprint(vars(m))                              # pretty print output
        scene_id = meta.LANDSAT_SCENE_ID             # access specific attribute

    :param filename: the filepath to an MTL file.
    """

    def __init__(self, filename):
        """
        There are several critical attributes that keep a common
        naming convention between all landsat versions, so they are
        initialized in this class for good record keeping and reference
        """

        # custom attribute additions
        self.FILEPATH           = filename
        self.DATETIME_OBJ       = None

        # product metadata attributes
        self.LANDSAT_SCENE_ID   = None
        self.DATA_TYPE          = None
        self.ELEVATION_SOURCE   = None
        self.OUTPUT_FORMAT      = None
        self.SPACECRAFT_ID      = None
        self.SENSOR_ID          = None
        self.WRS_PATH           = None
        self.WRS_ROW            = None
        self.NADIR_OFFNADIR     = None
        self.TARGET_WRS_PATH    = None
        self.TARGET_WRS_ROW     = None
        self.DATE_ACQUIRED      = None
        self.SCENE_CENTER_TIME  = None

        # image attributes
        self.CLOUD_COVER        = None
        self.IMAGE_QUALITY_OLI  = None
        self.IMAGE_QUALITY_TIRS = None
        self.ROLL_ANGLE         = None
        self.SUN_AZIMUTH        = None
        self.SUN_ELEVATION      = None
        self.EARTH_SUN_DISTANCE = None    # calculated for Landsats before 8.

        # read the file and populate the MTL attributes
        self._read(filename)

    def _read(self, filename):
        """ reads the contents of an MTL file """

        # if the "filename" input is actually already a metadata class object, return it back.
        if inspect.isclass(filename):
            return filename

        fields = []
        values = []

        metafile = open(filename, 'r')
        metadata = metafile.readlines()

        for line in metadata:
            # skips lines that contain "bad flags" denoting useless data AND lines
            # greater than 1000 characters. 1000 character limit works around an odd LC5
            # issue where the metadata has 40,000+ characters of whitespace
            bad_flags = ["END", "GROUP"]
            if not any(x in line for x in bad_flags) and len(line) <= 1000:
                try:
                    line = line.replace("  ", "")
                    line = line.replace("\n", "")
                    field_name, field_value = line.split(' = ')
                    fields.append(field_name)
                    values.append(field_value)
                except:
                    pass

        for i in range(len(fields)):

            # format fields without quotes,dates, or times in them as floats
            if not any(['"' in values[i], 'DATE' in fields[i], 'TIME' in fields[i]]):
                setattr(self, fields[i], float(values[i]))
            else:
                values[i] = values[i].replace('"', '')
                setattr(self, fields[i], values[i])

        # create datetime_obj attribute (drop decimal seconds)
        dto_string          = self.DATE_ACQUIRED + self.SCENE_CENTER_TIME
        self.DATETIME_OBJ   = datetime.strptime(dto_string.split(".")[0], "%Y-%m-%d%H:%M:%S")

        # only landsat 8 includes sun-earth-distance in MTL file, so calculate it
        # for the Landsats 4,5,7 using solar module.
#        if not self.SPACECRAFT_ID == "LANDSAT_8":
#
#            # use 0s for lat and lon, sun_earth_distance is not a function of any one location on earth.
#            s = solar(0, 0, self.DATETIME_OBJ, 0)
#            self.EARTH_SUN_DISTANCE = s.get_rad_vector()

        print("Scene {0} center time is {1}".format(self.LANDSAT_SCENE_ID, self.DATETIME_OBJ))


class GeoTIFF(object):
    """
    Represents a GeoTIFF file for data access and processing and provides
    a number of useful methods and attributes.

    Arguments:
      filepath (str): the full or relative file path
    """
    def __init__(self, filepath):
        try:
            self.dataobj = gdal.Open(filepath)
        except RuntimeError as err:
            LOGGER.error("Could not open %s: %s" % (filepath, err.message))
            raise
        self.filepath = filepath
        self.ncol = self.dataobj.RasterXSize
        self.nrow = self.dataobj.RasterYSize
        self.nbands = self.dataobj.RasterCount
        self._gtr = self.dataobj.GetGeoTransform()
        # see http://www.gdal.org/gdal_datamodel.html
        self.ulx = self._gtr[0]
        self.uly = self._gtr[3]
        self.lrx = (self.ulx + self.ncol * self._gtr[1]
                    + self.nrow * self._gtr[2])
        self.lry = (self.uly + self.ncol * self._gtr[4]
                    + self.nrow * self._gtr[5])
        if self._gtr[2] != 0 or self._gtr[4] != 0:
            LOGGER.warning(
                "The dataset is not north-up. The geotransform is given "
                + "by: (%s). " % ', '.join([str(item) for item in self._gtr])
                + "Northing and easting values will not have expected meaning."
                )
        self.dataobj = None

    @property
    def data(self):
        """2D numpy array for single-band GeoTIFF file data. Otherwise, 3D. """
        if not self.dataobj:
            self.dataobj = gdal.Open(self.filepath)
        dat = self.dataobj.ReadAsArray()
        self.dataobj = None
        return dat

    @property
    def projection(self):
        """The dataset's coordinate reference system as a Well-Known String"""
        if not self.dataobj:
            self.dataobj = gdal.Open(self.filepath)
        dat = self.dataobj.GetProjection()
        self.dataobj = None
        return dat

    @property
    def proj4(self):
        """The dataset's coordinate reference system as a PROJ4 string"""
        osrref = osr.SpatialReference()
        osrref.ImportFromWkt(self.projection)
        return osrref.ExportToProj4()

    @property
    def coordtrans(self):
        """A PROJ4 Proj object, which is able to perform coordinate
        transformations"""
        return Proj(self.proj4)

    @property
    def delx(self):
        """The sampling distance in x-direction, in physical units
        (eg metres)"""
        return self._gtr[1]

    @property
    def dely(self):
        """The sampling distance in y-direction, in physical units
        (eg metres). Negative in northern hemisphere."""
        return self._gtr[5]

    @property
    def easting(self):
        """The x-coordinates of first row pixel corners,
        as a numpy array: upper-left corner of upper-left pixel
        to upper-right corner of upper-right pixel (ncol+1)."""
        delta = np.abs(
            (self.lrx-self.ulx)/self.ncol
            - self.delx
            )
        if delta > 10e-2:
            LOGGER.warn(
                "GeoTIFF issue: E-W grid step differs from "
                + "deltaX by more than 1% ")
        return np.linspace(self.ulx, self.lrx, self.ncol+1)

    @property
    def northing(self):
        """The y-coordinates of first column pixel corners,
        as a numpy array: lower-left corner of lower-left pixel to
        upper-left corner of upper-left pixel (nrow+1)."""
        # check if data grid step is consistent
        delta = np.abs(
            (self.lry-self.uly)/self.nrow
            - self.dely
            )
        if delta > 10e-2:
            LOGGER.warn(
                "GeoTIFF issue: N-S grid step differs from "
                + "deltaY by more than 1% ")
        return np.linspace(self.lry, self.uly, self.nrow+1)

    @property
    def x_pxcenter(self):
        """The x-coordinates of pixel centers, as a numpy array ncol."""
        return np.linspace(
            self.ulx + self.delx/2,
            self.lrx - self.delx/2,
            self.ncol)

    @property
    def y_pxcenter(self):
        """y-coordinates of pixel centers, nrow."""
        return np.linspace(
            self.lry - self.dely/2,
            self.uly + self.dely/2,
            self.nrow)

    @property
    def _XY(self):
        """Meshgrid of nrow+1, ncol+1 corner xy coordinates"""
        return np.meshgrid(self.easting, self.northing)

    @property
    def _XY_pxcenter(self):
        """Meshgrid of nrow, ncol center xy coordinates"""
        return np.meshgrid(self.x_pxcenter, self.y_pxcenter)

    @property
    def _LonLat_pxcorner(self):
        """Meshgrid of nrow+1, ncol+1 corner Lon/Lat coordinates"""
        return self.coordtrans(*self._XY, inverse=True)

    @property
    def _LonLat_pxcenter(self):
        """Meshgrid of nrow, ncol center Lon/Lat coordinates"""
        return self.coordtrans(*self._XY_pxcenter, inverse=True)

    @property
    def Lon(self):
        """Longitude coordinate of each pixel corner, as an array"""
        return self._LonLat_pxcorner[0]

    @property
    def Lat(self):
        """Latitude coordinate of each pixel corner, as an array"""
        return self._LonLat_pxcorner[1]

    @property
    def Lon_pxcenter(self):
        """Longitude coordinate of each pixel center, as an array"""
        return self._LonLat_pxcenter[0]

    @property
    def Lat_pxcenter(self):
        """Latitude coordinate of each pixel center, as an array"""
        return self._LonLat_pxcenter[1]

    def ij2xy(self, i, j):
        """
        Converts array index pair(s) to easting/northing coordinate pairs(s).

        NOTE: array coordinate origin is in the top left corner whereas
        easting/northing origin is in the bottom left corner. Easting and
        northing are floating point numbers, and refer to the top-left corner
        coordinate of the pixel. i runs from 0 to nrow-1, j from 0 to ncol-1.
        For i=nrow and j=ncol, the bottom-right corner coordinate of the
        bottom-right pixel will be returned. This is identical to the bottom-
        right corner.

        Arguments:
            i (int): scalar or array of row coordinate index
            j (int): scalar or array of column coordinate index

        Returns:
            x (float): scalar or array of easting coordinates
            y (float): scalar or array of northing coordinates
        """
        if (_test_outside(i, 0, self.nrow)
                or _test_outside(j, 0, self.ncol)):
            raise RasterError(
                "Coordinates %d, %d out of bounds" % (i, j))
        x = self.easting[0] + j * self.delx
        y = self.northing[-1] + i * self.dely
        return x, y

    def xy2ij(self, x, y, precise=False):
        """
        Convert easting/northing coordinate pair(s) to array coordinate
        pairs(s).

        NOTE: see note at ij2xy()

        Arguments:
            x (float): scalar or array of easting coordinates
            y (float): scalar or array of northing coordinates
            precise (bool): if true, return fractional array coordinates

        Returns:
            i (int, or float): scalar or array of row coordinate index
            j (int, or float): scalar or array of column coordinate index
        """
        if (_test_outside(x, self.easting[0], self.easting[-1]) or
                _test_outside(y, self.northing[0], self.northing[-1])):
            raise RasterError("Coordinates out of bounds")
        i = (1 - (y  - self.northing[0]) /
             (self.northing[-1] - self.northing[0])) * self.nrow
        j = ((x - self.easting[0]) /
             (self.easting[-1] - self.easting[0])) * self.ncol
        if precise:
            return i, j
        else:
            return int(np.floor(i)), int(np.floor(j))

    def simpleplot(self):
        """Quick and dirty plot of each band (channel, dataset) in the image.
        Requires Matplotlib."""
        import matplotlib.pyplot as plt
        numbands = self.nbands
        if numbands == 1:
            plt.figure(figsize=(15, 10))
            plt.imshow(self.data[:, :], cmap='bone')
        elif numbands > 1:
            for idx in range(numbands):
                plt.figure(figsize=(15, 10))
                plt.imshow(self.data[idx, :, :], cmap='bone')
        return True

    def clone(self, newpath, newdata):
        """
        Creates new GeoTIFF object from existing: new data, same georeference.

        Arguments:
            newpath: valid file path
            newdata: numpy array, 2 or 3-dim

        Returns:
            A raster.GeoTIFF object
        """
        # convert Numpy dtype objects to GDAL type codes
        # see https://gist.github.com/chryss/8366492

        NPDTYPE2GDALTYPECODE = {
            "uint8": 1,
            "int8": 1,
            "uint16": 2,
            "int16": 3,
            "uint32": 4,
            "int32": 5,
            "float32": 6,
            "float64": 7,
            "complex64": 10,
            "complex128": 11,
        }
        # check if newpath is potentially a valid file path to save data
        dirname, fname = os.path.split(newpath)
        if dirname:
            if not os.path.isdir(dirname):
                print("%s is not a valid directory to save file to " % dirname)
        if os.path.isdir(newpath):
            LOGGER.warning(
                "%s is a directory." % dirname + " Choose a name "
                + "that is suitable for writing a dataset to.")
        if (newdata.shape != self.data.shape
                and newdata.shape != self.data[0, ...].shape):
            raise RasterError(
                "New and cloned GeoTIFF dataset must be the same shape.")
        dims = newdata.ndim
        if dims == 2:
            bands = 1
        elif dims > 2:
            bands = newdata.shape[0]
        else:
            raise RasterError(
                "New data array has only %s dimensions." % dims)
        try:
            LOGGER.info(newdata.dtype.name)
            LOGGER.info(NPDTYPE2GDALTYPECODE)
            LOGGER.info(NPDTYPE2GDALTYPECODE[newdata.dtype.name])
            gdaltype = NPDTYPE2GDALTYPECODE[newdata.dtype.name]
        except KeyError as err:
            raise RasterError(
                "Data type in array %s " % newdata.dtype.name
                + "cannot be converted to GDAL data type: \n%s" % err.message)
        proj = self.projection
        geotrans = self._gtr
        gtiffdr = gdal.GetDriverByName('GTiff')
        gtiff = gtiffdr.Create(newpath, self.ncol, self.nrow, bands, gdaltype)
        gtiff.SetProjection(proj)
        gtiff.SetGeoTransform(geotrans)
        if dims == 2:
            gtiff.GetRasterBand(1).WriteArray(newdata)
        else:
            for idx in range(dims):
                gtiff.GetRasterBand(idx+1).WriteArray(newdata[idx, :, :])
        gtiff = None
        return GeoTIFF(newpath)