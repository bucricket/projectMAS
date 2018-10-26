import os
import sqlite3
import pandas as pd
import datetime
import numpy as np
import wget


def updateModisDB(filenames, cacheDir):
    if len(filenames) > 0:
        db_fn = os.path.join(cacheDir, "modis_db.db")
        fn = filenames[0].split(os.sep)[-1]
        product = fn.split('.')[0]
        years = []
        doys = []
        tiles = []
        fns = []
        for filename in filenames:
            fn = filename.split(os.sep)[-1]
            fns.append(filename)
            years.append(fn.split('.')[1][1:5])
            doys.append(fn.split('.')[1][5:9])
            tiles.append(fn.split('.')[2])
        if not os.path.exists(db_fn):
            conn = sqlite3.connect(db_fn)
            modis_dict = {"TILE": tiles, "YEAR": years, "DOY": doys, "filename": fns}
            modis_df = pd.DataFrame.from_dict(modis_dict)
            modis_df.to_sql("%s" % product, conn, if_exists="replace", index=False)
            conn.close()
        else:
            conn = sqlite3.connect(db_fn)
            orig_df = pd.read_sql_query("SELECT * from %s" % product, conn)
            modis_dict = {"TILE": tiles, "YEAR": years, "DOY": doys, "filename": fns}
            modis_df = pd.DataFrame.from_dict(modis_dict)
            orig_df = orig_df.append(modis_df, ignore_index=True)
            orig_df = orig_df.drop_duplicates(keep='last')
            orig_df.to_sql("%s" % product, conn, if_exists="replace", index=False)
            conn.close()


def searchModisDB(tiles, start_date, end_date, product, cacheDir):
    db_fn = os.path.join(cacheDir, "modis_db.db")
    conn = sqlite3.connect(db_fn)
    startdd = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    enddd = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    numDays = (enddd - startdd).days
    laidates = np.array(range(1, 366, 4))
    df1 = pd.DataFrame.from_dict({"TILE": [], "YEAR": [], "DOY": [], "filename": []})
    df2 = pd.DataFrame.from_dict({"TILE": [], "YEAR": [], "DOY": []})
    if isinstance(tiles, basestring):
        tiles = [tiles]
    for tile in tiles:
        for i in range(numDays + 1):
            dd = startdd + datetime.timedelta(days=i)
            year = dd.year
            doy = (dd - datetime.datetime(year, 1, 1, 0, 0)).days + 1
            rday = laidates[laidates >= doy][0]
            if (doy == rday):
                dd = datetime.datetime(year, 1, 1, 0, 0) + datetime.timedelta(days=rday - 1)
                year = dd.year
                df = pd.read_sql_query("SELECT * from %s WHERE (TILE = '%s')"
                                       "AND (YEAR =  '%d') AND (DOY = '%03d' )" %
                                       (product, tile, year, rday), conn)
                df1 = df1.append(df, ignore_index=True)
                df1 = df1[["DOY", "TILE", "YEAR"]]
                row = pd.Series({"TILE": "%s" % tile, "YEAR": "%d" % year, "DOY": "%03d" % rday})
                df2 = df2.append(row, ignore_index=True)
    merged = df2.merge(df1, indicator=True, how='outer')
    df3 = merged[merged['_merge'] != 'both']
    out_df = df3[["DOY", "TILE", "YEAR"]]
    conn.close()
    return out_df


def search(lat, lon, start_date, end_date, cloud, available, cacheDir, sat):
    """ Search the USGS Landsat database """
    end = datetime.strptime(end_date, '%Y-%m-%d')
    # this is a landsat-util work around when it fails
    if sat == 7:
        metadataUrl = 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_ETM_C1.csv'
    else:
        metadataUrl = 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_8_C1.csv'

    fn = os.path.join(cacheDir, metadataUrl.split(os.sep)[-1])
    # looking to see if metadata CSV is available and if its up to the date needed
    if os.path.exists(fn):
        d = datetime.fromtimestamp(os.path.getmtime(fn))
        db_name = os.path.join(cacheDir, fn.split(os.sep)[-1][:-4] + '.db')
        if not os.path.exists(db_name):
            orig_df = pd.read_csv(fn)
            orig_df['sr'] = pd.Series(np.tile('N', len(orig_df)))
            orig_df['bt'] = pd.Series(np.tile('N', len(orig_df)))
            orig_df['local_file_path'] = ''
            conn = sqlite3.connect(db_name)
            orig_df.to_sql("raw_data", conn, if_exists="replace", index=False)
            conn.close()
        #            orig_df = pd.read_sql_query("SELECT * from raw_data",conn)

        if ((end.year > d.year) and (end.month > d.month) and (end.day > d.day)):
            wget.download(metadataUrl, out=fn)
            metadata = pd.read_csv(fn)
            metadata['sr'] = pd.Series(np.tile('N', len(metadata)))
            metadata['bt'] = pd.Series(np.tile('N', len(metadata)))
            orig_df = pd.read_sql_query("SELECT * from raw_data", conn)
            orig_df = orig_df.append(metadata, ignore_index=True)
            orig_df = orig_df.drop_duplicates(subset='sceneID', keep='first')
            orig_df.to_sql("raw_data", conn, if_exists="replace", index=False)
    else:
        wget.download(metadataUrl, out=fn)
        db_name = os.path.join(cacheDir, fn.split(os.sep)[-1][:-4] + '.db')
        conn = sqlite3.connect(db_name)
        metadata = pd.read_csv(fn)
        metadata['sr'] = pd.Series(np.tile('N', len(metadata)))
        metadata['bt'] = pd.Series(np.tile('N', len(metadata)))
        metadata['local_file_path'] = ''
        metadata.to_sql("raw_data", conn, if_exists="replace", index=False)
        conn.close()
    conn = sqlite3.connect(db_name)
    if sat == 8:
        output = pd.read_sql_query("SELECT * from raw_data WHERE (acquisitionDate >= '%s')"
                                   "AND (acquisitionDate < '%s') AND (upperLeftCornerLatitude > %f )"
                                   "AND (upperLeftCornerLongitude < %f ) AND "
                                   "(lowerRightCornerLatitude < %f) AND "
                                   "(lowerRightCornerLongitude > %f) AND "
                                   "(cloudCoverFull <= %d) AND (sr = '%s') AND "
                                   "(sensor = 'OLI_TIRS')" %
                                   (start_date, end_date, lat, lon, lat, lon, cloud, available), conn)
    else:
        output = pd.read_sql_query("SELECT * from raw_data WHERE (acquisitionDate >= '%s')"
                                   "AND (acquisitionDate < '%s') AND (upperLeftCornerLatitude > %f )"
                                   "AND (upperLeftCornerLongitude < %f ) AND "
                                   "(lowerRightCornerLatitude < %f) AND "
                                   "(lowerRightCornerLongitude > %f) AND "
                                   "(cloudCoverFull <= %d) AND (sr = '%s')" %
                                   (start_date, end_date, lat, lon, lat, lon, cloud, available), conn)
    conn.close()
    return output

def searchProduct(productID,db_path,sat):
    """ search Landsat database by ProductID """
    if sat==7:
        metadataUrl = 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_ETM_C1.csv'
        db_name = os.path.join(db_path,'LANDSAT_ETM_C1.db')
    else:
        metadataUrl = 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_8_C1.csv'
        db_name = os.path.join(db_path,'LANDSAT_8_C1.db')

    fn  = os.path.join(db_path,metadataUrl.split(os.sep)[-1])
    if not os.path.exists(db_name):
        if not os.path.exists(fn):
            wget.download(metadataUrl,out=fn)
        conn = sqlite3.connect( db_name )
        orig_df= pd.read_csv(fn)
        orig_df['sr'] = pd.Series(np.tile('N',len(orig_df)))
        orig_df['bt'] = pd.Series(np.tile('N',len(orig_df)))
        orig_df['local_file_path'] = ''
        orig_df.to_sql("raw_data", conn, if_exists="replace", index=False)
        conn.close()

    conn = sqlite3.connect( db_name )
    output = pd.read_sql_query("SELECT * from raw_data WHERE (LANDSAT_PRODUCT_ID == '%s')" %  productID,conn)
    conn.close()
    return output

def updateLandsatProductsDB(landsatDB, filenames, cacheDir, product):
    db_fn = os.path.join(cacheDir, "landsat_products.db")

    date = landsatDB.acquisitionDate
    ullat = landsatDB.upperLeftCornerLatitude
    ullon = landsatDB.upperLeftCornerLongitude
    lllat = landsatDB.lowerRightCornerLatitude
    lllon = landsatDB.lowerRightCornerLongitude
    productIDs = landsatDB.LANDSAT_PRODUCT_ID

    if not os.path.exists(db_fn):
        conn = sqlite3.connect(db_fn)
        landsat_dict = {"acquisitionDate": date, "upperLeftCornerLatitude": ullat,
                        "upperLeftCornerLongitude": ullon,
                        "lowerRightCornerLatitude": lllat,
                        "lowerRightCornerLongitude": lllon,
                        "LANDSAT_PRODUCT_ID": productIDs, "filename": filenames}
        landsat_df = pd.DataFrame.from_dict(landsat_dict)
        landsat_df.to_sql("%s" % product, conn, if_exists="replace", index=False)
        conn.close()
    else:
        conn = sqlite3.connect(db_fn)
        res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = res.fetchall()[0]
        if (product in tables):
            orig_df = pd.read_sql_query("SELECT * from %s" % product, conn)
        else:
            orig_df = pd.DataFrame()

        landsat_dict = {"acquisitionDate": date, "upperLeftCornerLatitude": ullat,
                        "upperLeftCornerLongitude": ullon,
                        "lowerRightCornerLatitude": lllat,
                        "lowerRightCornerLongitude": lllon,
                        "LANDSAT_PRODUCT_ID": productIDs, "filename": filenames}
        landsat_df = pd.DataFrame.from_dict(landsat_dict)
        orig_df = orig_df.append(landsat_df, ignore_index=True)
        orig_df = orig_df.drop_duplicates(keep='last')
        orig_df.to_sql("%s" % product, conn, if_exists="replace", index=False)
        conn.close()


def searchLandsatProductsDB(lat, lon, start_date, end_date, product, cacheDir):
    db_fn = os.path.join(cacheDir, "landsat_products.db")
    conn = sqlite3.connect(db_fn)

    out_df = pd.read_sql_query("SELECT * from %s WHERE (acquisitionDate >= '%s')"
                               "AND (acquisitionDate < '%s') AND (upperLeftCornerLatitude > %f )"
                               "AND (upperLeftCornerLongitude < %f ) AND "
                               "(lowerRightCornerLatitude < %f) AND "
                               "(lowerRightCornerLongitude > %f)" %
                               (product, start_date, end_date, lat, lon, lat, lon), conn)
    conn.close()
    return out_df
