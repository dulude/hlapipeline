"""Wrappers for astroquery-related functionality"""
import shutil
import os
import pdb
from astroquery.mast import Observations
from astropy.table import Table


def retrieve_observation(obsid, suffix=['FLC']):
    """Simple interface for retrieving an observation from the MAST archive

    If the input obsid is for an association, it will request all members with
    the specified suffixes.

    Parameters
    -----------
    obsid : string
        ID for observation to be retrieved from the MAST archive.  Only the
        IPPSSOOT (rootname) of exposure or ASN needs to be provided; eg., ib6v06060.

    suffix : list
        List containing suffixes of files which should be requested from MAST.

    path : string
        Directory to use for writing out downloaded files.  If `None` (default),
        the current working directory will be used.

    """
    local_files = []

    # Query MAST for the data with an observation type of either "science" or "calibration"
    obsTable = Observations.query_criteria(obs_id=obsid, obstype='all')

    # Catch the case where no files are found for download
    if len(obsTable) == 0:
        print("WARNING: Query for {} returned NO RESULTS!".format(obsid))
        return local_files

    dpobs = Observations.get_product_list(obsTable)
    dataProductsByID = Observations.filter_products(dpobs,
                                              productSubGroupDescription=suffix,
                                              extension='fits',
                                              mrp_only=False)

    # After the filtering has been done, ensure there is still data in the table for download.
    # If the table is empty, look for FLT images in lieu of FLC images. Only want one
    # or the other (not both!), so just do the filtering again.
    if len(dataProductsByID) == 0:
        print("WARNING: No FLC files found for {} - will look for FLT files instead.".format(obsid))
        suffix = ['FLT']
        dataProductsByID = Observations.filter_products(dpobs,
                                              productSubGroupDescription=suffix,
                                              extension='fits',
                                              mrp_only=False)

        # If still no data, then return.  An exception will eventually be thrown in
        # the higher level code.
        if len(dataProductsByID) == 0:
            print("WARNING: No FLC or FLT files found for {}.".format(obsid))
            return local_files
        
    # if clobber == False:    #TODO: Finish clobber section
    #     rowsToRemove = []
    #     for rowCtr in range(0,len(dataProductsByID)):
    #         if os.path.exists(dataProductsByID[rowCtr]['productFilename']):
    #             print("{} already exists. File download skipped.".format(dataProductsByID[rowCtr]['productFilename']))
    #             rowsToRemove.append(rowCtr)
    #     if rowsToRemove:
    #         rowsToRemove.reverse()
    #         for rowNum in rowsToRemove:
    #             dataProductsByID.remove_row(rowNum)


    manifest = Observations.download_products(dataProductsByID, mrp_only=False)
    download_dir = None
    for file in manifest['Local Path']:
        # Identify what sub-directory was created by astroquery for the download
        if download_dir is None:
            file_path = file.split(os.sep)
            file_path.remove('.')
            download_dir = file_path[0]
        # Move downloaded file to current directory
        local_file = os.path.abspath(os.path.basename(file))
        shutil.move(file, local_file)
        # Record what files were downloaded and their current location
        local_files.append(local_file)
    # Remove astroquery created sub-directories
    shutil.rmtree(download_dir)
    return local_files
