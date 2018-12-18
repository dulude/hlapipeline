#!/usr/bin/env python

"""This script is a modernized replacement of tweakreg.

"""

from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict
from drizzlepac import updatehdr
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import pdb
from stsci.tools import fileutil
from stwcs.wcsutil import HSTWCS
import sys
import tweakwcs
try:
    from hlapipeline.utils import astrometric_utils as amutils
    from hlapipeline.utils import astroquery_utils as aqutils
    from hlapipeline.utils import filter
except:
    from utils import astrometric_utils as amutils
    from utils import astroquery_utils as aqutils
    from utils import filter

MIN_CATALOG_THRESHOLD = 3
MIN_OBSERVABLE_THRESHOLD = 10
MIN_CROSS_MATCHES = 3
MIN_FIT_MATCHES = 6
MAX_FIT_RMS = 1.0

# Module-level dictionary contains instrument/detector-specific parameters used later on in the script.
detector_specific_params = {"acs":
                                {"hrc":
                                     {"fwhmpsf": 0.073,
                                      "classify": True,
                                      "threshold": None},
                                 "sbc":
                                     {"fwhmpsf": 0.065,
                                      "classify": False,
                                      "threshold": 10},
                                 "wfc":
                                     {"fwhmpsf": 0.076,
                                      "classify": True,
                                      "threshold": -1.1}},
                            "wfc3":
                                {"ir":
                                     {"fwhmpsf": 0.14,
                                      "classify": False,
                                      "threshold": None},
                                 "uvis":
                                     {"fwhmpsf": 0.076,
                                      "classify": True,
                                      "threshold": None}}} # fwhmpsf in units of arcsec


# ----------------------------------------------------------------------------------------------------------------------


def check_and_get_data(input_list,**pars):
    """Verify that all specified files are present. If not, retrieve them from MAST.

    Parameters
    ----------
    imglist : list
        List of one or more calibrated fits images that will be used for catalog generation.

    Returns
    =======
    input_file_list : list
        list of full filenames

    """
    totalInputList=[]
    for input_item in input_list:
        if input_item.endswith("0"): #asn table
            totalInputList += aqutils.retrieve_observation(input_item,**pars)

        else: #single file rootname.
            fitsfilename = glob.glob("{}_flc.fits".format(input_item))
            if not fitsfilename:
                fitsfilename = glob.glob("{}_flt.fits".format(input_item))
            fitsfilename = fitsfilename[0]

            if not os.path.exists(fitsfilename):
                imghdu = fits.open(fitsfilename)
                imgprimaryheader = imghdu[0].header
                try:
                    asnid = imgprimaryheader['ASN_ID'].strip().lower()
                except:
                    asnid = 'NONE'
                if asnid[0] in ['i','j']:
                    totalInputList += aqutils.retrieve_observation(asnid,**pars)
                else:
                    totalInputList += aqutils.retrieve_observation(input_item, **pars) #try with ippssoot instead

            else: totalInputList.append(fitsfilename)
    print("TOTAL INPUT LIST: ",totalInputList)
    # TODO: add trap to deal with non-existent (incorrect) rootnames
    # TODO: Address issue about how the code will retrieve association information if there isn't a local file to get 'ASN_ID' header info
    return(totalInputList)


# ----------------------------------------------------------------------------------------------------------------------

def convert_string_tf_to_boolean(invalue):
    """Converts string 'True' or 'False' value to Boolean True or Boolean False.

    :param invalue: string
        input true/false value

    :return: Boolean
        converted True/False value
    """
    outvalue = False
    if invalue == 'True':
        outvalue = True
    return(outvalue)


# ----------------------------------------------------------------------------------------------------------------------


def perform_align(input_list, archive=False, clobber=False, makeplots=False, plotdest='screen',update_hdr_wcs=False):
    """Main calling function.

    Parameters
    ----------
    input_list : list
        List of one or more IPPSSOOTs (rootnames) to align.

    archive : Boolean
        Retain copies of the downloaded files in the astroquery created sub-directories?

    clobber : Boolean
        Download and overwrite existing local copies of input files?

    makeplots : Boolean
        Generate 2-d vector plots?

    update_hdr_wcs : Boolean
        Write newly computed WCS information to image image headers?

    Returns
    -------
    int value 0 if successful, int value 1 if unsuccessful

    """

    # Define astrometric catalog list in priority order
    catalogList = ['GAIADR2', 'GSC241']
    numCatalogs = len(catalogList)

    # 1: Interpret input data and optional parameters
    print("-------------------- STEP 1: Get data --------------------")
    imglist = check_and_get_data(input_list, archive=archive, clobber=clobber)
    print("\nSUCCESS")

    # 2: Apply filter to input observations to insure that they meet minimum criteria for being able to be aligned
    print("-------------------- STEP 2: Filter data --------------------")
    filteredTable = filter.analyze_data(imglist)

    # Check the table to determine if there is any viable data to be aligned.  The
    # 'doProcess' column (bool) indicates the image/file should or should not be used
    # for alignment purposes.
    if filteredTable['doProcess'].sum() == 0:
        print("No viable images in filtered table - no processing done.\n")
        return(1)

    # Get the list of all "good" files to use for the alignment
    processList = filteredTable['imageName'][np.where(filteredTable['doProcess'])]
    processList = list(processList) #Convert processList from numpy list to regular python list
    print("\nSUCCESS")

    # 3: Build WCS for full set of input observations
    print("-------------------- STEP 3: Build WCS --------------------")
    refwcs = amutils.build_reference_wcs(processList)
    print("\nSUCCESS")

    # 4: Retrieve list of astrometric sources from database
    # While loop to accommodate using multiple catalogs
    doneFitting = False
    catalogIndex = 0
    extracted_sources = None
    while not doneFitting:
        skip_all_other_steps = False
        retry_fit = False
        print("-------------------- STEP 4: Detect astrometric sources --------------------")
        print("Astrometric Catalog: ",catalogList[catalogIndex])
        reference_catalog = generate_astrometric_catalog(processList, catalog=catalogList[catalogIndex])
        # The table must have at least MIN_CATALOG_THRESHOLD entries to be useful
        if len(reference_catalog) >= MIN_CATALOG_THRESHOLD:
            print("\nSUCCESS")
        else:
            if catalogIndex < numCatalogs - 1:
                print("Not enough sources found in catalog " + catalogList[catalogIndex])
                print("Try again with the next catalog")
                catalogIndex += 1
                retry_fit = True
                skip_all_other_steps = True
            else:
                print("Not enough sources found in any catalog - no processing done.")
                return (1)
        if not skip_all_other_steps:
        # 5: Extract catalog of observable sources from each input image
            print("-------------------- STEP 5: Source finding --------------------")
            if not extracted_sources:
                # extracted_sources = generate_source_catalogs(processList) # TODO: uncomment this once debugging is done

                pickle_filename = "{}.source_catalog.pickle".format(processList[0]) # TODO: All this pickle stuff is only here for debugging. <START>
                if os.path.exists(pickle_filename):
                    pickle_in = open(pickle_filename, "rb")
                    extracted_sources = pickle.load(pickle_in)
                    print("Using sourcelist extracted from {} generated during the last run to save time.".format(pickle_filename))
                else:
                    extracted_sources = generate_source_catalogs(processList,output=True) #TODO: ADD TO INPUT PARAMS!
                    pickle_out = open(pickle_filename, "wb")
                    pickle.dump(extracted_sources, pickle_out)
                    pickle_out.close()
                    print("Wrote ",pickle_filename)# TODO: All this pickle stuff is only here for debugging. <END>

                for imgname in extracted_sources.keys():
                    table=extracted_sources[imgname]["catalog_table"]
                    # The catalog of observable sources must have at least MIN_OBSERVABLE_THRESHOLD entries to be useful
                    total_num_sources = 0
                    for chipnum in table.keys():
                        total_num_sources += len(table[chipnum])
                    if total_num_sources < MIN_OBSERVABLE_THRESHOLD:
                        print("Not enough sources ({}) found in image {}".format(total_num_sources,imgname))
                        return(1)
            # Convert input images to tweakwcs-compatible NDData objects and
            # attach source catalogs to them.
            imglist = []
            for group_id, image in enumerate(processList):
                imglist.extend(amutils.build_nddata(image, group_id,
                                                    extracted_sources[image]['catalog_table']))
            print("\nSUCCESS")

        # 6: Cross-match source catalog with astrometric reference source catalog, Perform fit between source catalog and reference catalog
            print("-------------------- STEP 6: Cross matching and fitting --------------------")
            # Specify matching algorithm to use
            tol = 100.0
            match = tweakwcs.TPMatch(searchrad=250, separation=0.1,
                                     tolerance=tol, use2dhist=False)
            # Align images and correct WCS
            tweakwcs.tweak_image_wcs(imglist, reference_catalog, match=match)

            tweakwcs_info_keys = OrderedDict(imglist[0].meta['tweakwcs_info']).keys()
            imgctr=0
            for item in imglist:
                retry_fit = False
                #Handle fitting failures (no matches found)
                if item.meta['tweakwcs_info']['status'].startswith("FAILED") == True:
                    if catalogIndex < numCatalogs - 1:
                        print("No cross matches found between astrometric catalog and sources found in images")
                        print("Try again with the next catalog")
                        catalogIndex += 1
                        retry_fit = True
                        break
                    else:
                        print("No cross matches found in any catalog - no processing done.")
                        return (1)
                max_rms_val = max(item.meta['tweakwcs_info']['rms'])
                num_xmatches = item.meta['tweakwcs_info']['nmatches']
                radial_shift = math.sqrt(item.meta['tweakwcs_info']['shift'][0] ** 2 + item.meta['tweakwcs_info']['shift'][1] ** 2)
                # print fit params to screen
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIT PARAMETERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                if item.meta['chip'] == 1:
                    image_name = processList[imgctr]
                    imgctr += 1
                print("image: {}".format(image_name))
                print("chip: {}".format(item.meta['chip']))
                print("group_id: {}".format(item.meta['group_id']))
                for tweakwcs_info_key in tweakwcs_info_keys:
                    if not tweakwcs_info_key.startswith("matched"):
                        print("{} : {}".format(tweakwcs_info_key,item.meta['tweakwcs_info'][tweakwcs_info_key]))
                print("tweakwcs.TPMatch tolerance: {}".format(tol))
                print("Radial shift: {}".format(radial_shift))
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                if makeplots == True and num_xmatches >= MIN_CROSS_MATCHES: #make vector plots
                    generate_vector_plot(item,image_name+"[SCI,{}]".format(item.meta['chip']),plotDest=plotdest)
                if num_xmatches < MIN_CROSS_MATCHES:
                    if catalogIndex < numCatalogs-1:
                        print("Not enough cross matches found between astrometric catalog and sources found in images")
                        print("Try again with the next catalog")
                        catalogIndex += 1
                        retry_fit = True
                        break
                    else:
                        print("Not enough cross matches found in any catalog - no processing done.")
                        return(1)
                elif max_rms_val > MAX_FIT_RMS:
                    if catalogIndex < numCatalogs-1:
                        print("Fit RMS value(s) X_rms= {}, Y_rms = {} greater than the maximum threshold value {}.".format(item.meta['tweakwcs_info']['rms'][0], item.meta['tweakwcs_info']['rms'][1],MAX_FIT_RMS))
                        print("Try again with the next catalog")
                        catalogIndex += 1
                        retry_fit = True
                        break
                    else:
                        print("Fit RMS values too large using any catalog - no processing done.")
                        return(1)
                else:
                    print("Fit calculations successful.")
        if not retry_fit:
            print("\nSUCCESS")

            # 7: Write new fit solution to input image headers
            print("-------------------- STEP 7: Update image headers with new WCS information --------------------")
            if update_hdr_wcs:
                update_image_wcs_info(imglist, processList)
                print("\nSUCCESS")
            else:
                print("\n STEP SKIPPED")
            return (0)



# ----------------------------------------------------------------------------------------------------------------------


def generate_astrometric_catalog(imglist, **pars):
    """Generates a catalog of all sources from an existing astrometric catalog are in or near the FOVs of the images in
        the input list.

    Parameters
    ----------
    imglist : list
        List of one or more calibrated fits images that will be used for catalog generation.

    Returns
    =======
    ref_table : object
        Astropy Table object of the catalog

    """
    # generate catalog
    out_catalog = amutils.create_astrometric_catalog(imglist,**pars)

    # if the catalog has contents, write the catalog to ascii text file
    if len(out_catalog) > 0:
        catalog_filename = "refcatalog.cat"
        out_catalog.write(catalog_filename, format="ascii.fast_commented_header")
        print("Wrote reference catalog {}.".format(catalog_filename))

    return(out_catalog)


# ----------------------------------------------------------------------------------------------------------------------


def generate_source_catalogs(imglist, **pars):
    """Generates a dictionary of source catalogs keyed by image name.

    Parameters
    ----------
    imglist : list
        List of one or more calibrated fits images that will be used for source detection.

    Returns
    -------
    sourcecatalogdict : dictionary
        a dictionary (keyed by image name) of two element dictionaries which in tern contain 1) a dictionary of the
        detector-specific processing parameters and 2) an astropy table of position and photometry information of all
        detected sources
    """
    output = pars.get('output', False)
    sourcecatalogdict = {}
    for imgname in imglist:
        print("Image name: ", imgname)

        sourcecatalogdict[imgname] = {}

        # open image
        imghdu = fits.open(imgname)
        imgprimaryheader = imghdu[0].header
        instrument = imgprimaryheader['INSTRUME'].lower()
        detector = imgprimaryheader['DETECTOR'].lower()

        # get instrument/detector-specific image alignment parameters
        if instrument in detector_specific_params.keys():
            if detector in detector_specific_params[instrument].keys():
                detector_pars = detector_specific_params[instrument][detector]
                sourcecatalogdict[imgname]["params"] = detector_pars
                # to allow generate_source_catalog to get detector specific parameters
                pars.update(detector_pars)
            else:
                sys.exit("ERROR! Unrecognized detector '{}'. Exiting...".format(detector))
        else:
            sys.exit("ERROR! Unrecognized instrument '{}'. Exiting...".format(instrument))

        # Identify sources in image, convert coords from chip x, y form to reference WCS sky RA, Dec form.
        imgwcs = HSTWCS(imghdu, 1)
        fwhmpsf_pix = sourcecatalogdict[imgname]["params"]['fwhmpsf']/imgwcs.pscale #Convert fwhmpsf from arsec to pixels
        sourcecatalogdict[imgname]["catalog_table"] = amutils.generate_source_catalog(imghdu, fwhm=fwhmpsf_pix, **pars)

        # write out coord lists to files for diagnostic purposes. Protip: To display the sources in these files in DS9,
        # set the "Coordinate System" option to "Physical" when loading the region file.
        imgroot = os.path.basename(imgname).split('_')[0]
        numSci = amutils.countExtn(imghdu)
        # Allow user to decide when and how to write out catalogs to files
        if output:
            for chip in range(1,numSci+1):
                regfilename = "{}_sci{}_src.reg".format(imgroot, chip)
                out_table = Table(sourcecatalogdict[imgname]["catalog_table"][chip])
                out_table.write(regfilename, include_names=["xcentroid", "ycentroid"], format="ascii.fast_commented_header")
                print("Wrote region file {}\n".format(regfilename))
        imghdu.close()
    return(sourcecatalogdict)


# ----------------------------------------------------------------------------------------------------------------------

def generate_vector_plot(tweakwcs_output,imagename,**pars):
    """Performs all nessessary coord transforms and array generations in preparation for the call of subroutine
        makeVectorPLot().

    tweakwcs_output : list
        a single entry from the tweakwcs output list "imglist".

    imagename: string
        name of the image being plotted

    Returns
    =======
    Nothing.
    """
    # 1: extract x and y values from catalog
    raw_x_coords = np.asarray(tweakwcs_output.meta['catalog']['x'].data)
    raw_y_coords = np.asarray(tweakwcs_output.meta['catalog']['y'].data)
    # 2: list of indicies of matched catalog entries, extract them from the catalog
    good_idx_list = list(range(len(raw_x_coords))) # TODO: this is just a place holder until I nail down the mappings.
    good_raw_x_coords = np.take(raw_x_coords,good_idx_list)
    good_raw_y_coords = np.take(raw_y_coords,good_idx_list)

    fake_x_coords = good_raw_x_coords+1.0 #TODO: Figure out transfomrations
    fake_y_coords = good_raw_y_coords + 2.0 #TODO: Figure out transfomrations

    #get the x and y values into the form used by the makeVectorPlot() subroutine.
    plot_x_ra = np.stack((good_raw_x_coords,fake_x_coords))
    plot_y_ra = np.stack((good_raw_y_coords,fake_y_coords))

    makeVectorPlot(plot_x_ra,plot_y_ra,imagename,**pars)



# ----------------------------------------------------------------------------------------------------------------------


def makeVectorPlot(x,y,imagename,plotDest="screen",binThresh = 10000,binSize=250):
    """Generate vector plot of dx and dy values vs. reference (x,y) positions

    x : numpy.ndarray
        A 2 x n sized numpy array. Column 1: matched reference X values. Column 2: The corresponding matched
        comparision X values

    y : numpy.ndarray
        A 2 x n sized numpy array. Column 1: matched reference Y values. Column 2: The corresponding matched
        comparision Y values

    imagename : string
        Image name and chip number whose fit residuals are being plotted, using the following syntax:
        <IMAGE NAME>[SCI,<CHIP NUMBER>]

    plot : string
        plot destination: 'screen' or 'file'; 'screen' displays the plot in an interactive plotting window; 'file'
        saves the plot as a pdf file with the following file naming syntax:
        <IPPSSOOT>_sci<CHIP NUMBER>_xy_vector_plot.pdf. Default is 'screen'.

    binThresh : integer
        Minimum size of list *x* and *y* that will trigger generation of a binned vector plot. Default value = 10000.

    binSize : integer
        Size of binning box in pixels. When generating a binned vector plot, mean dx and dy values are computed by
        taking the mean of all points located within the box. Default value = 250.

    Returns
    =======
    Nothing.
    """
    dx = x[1, :] - x[0, :]
    dy = y[1, :] - y[0, :]
    if len(dx)>binThresh:# if the input list is larger than binThresh, a binned vector plot will be generated.
        binStatus = "%d x %d Binning"%(binSize,binSize)
        print("Input list length greater than threshold length value. Generating binned vector plot using {} pixel x {} pixel bins".format(binSize,binSize))
        if min(x[0,:])<0.0: xmin=min(x[0,:])
        else: xmin = 0.0
        if min(y[0,:])<0.0: ymin=min(y[0,:])
        else: ymin = 0.0

        p_x=np.empty(shape=[0])
        p_y=np.empty(shape=[0])
        p_dx=np.empty(shape=[0])
        p_dy=np.empty(shape=[0])
        color_ra=[]
        for xBinCtr in range(int(round2ArbatraryBase(xmin,"down",binSize)),int(round2ArbatraryBase(max(x[0,:]),"up",binSize)),binSize):
            for yBinCtr in range(int(round2ArbatraryBase(ymin, "down", binSize)),
                                 int(round2ArbatraryBase(max(y[0, :]), "up", binSize)), binSize):
                #define bin box x,y upper and lower bounds
                xBinMin=xBinCtr
                xBinMax=xBinMin+binSize
                yBinMin=yBinCtr
                yBinMax=yBinMin+binSize
                #get indicies of x and y withen bounding box
                ix0 = np.where((x[0,:] >= xBinMin) & (x[0,:] < xBinMax) & (y[0,:] >= yBinMin) & (y[0,:] < yBinMax))
                if len(dx[ix0]) > 0 and len(dy[ix0]) > 0: #ignore empty bins
                    p_x=np.append(p_x, xBinCtr + 0.5 * binSize) #X and Y posotion at center of bin.
                    p_y=np.append(p_y, yBinCtr + 0.5 * binSize)
                    mean_dx=np.mean(dx[ix0])
                    p_dx=np.append(p_dx, mean_dx) #compute mean dx, dy values
                    mean_dy = np.mean(dy[ix0])
                    p_dy=np.append(p_dy,mean_dy)
                    avg_npts=(float(len(dx[ix0]))+float(len(dy[ix0])))/2.0 #keep an eye out for mean values computed from less than 10 samples.
                    if (avg_npts<10.0): #if less than 10 samples were used in mean calculation, color the vector red.
                        color_ra.append('r')
                    else:
                        color_ra.append('k')
        lowSampleWarning = ""
        if "r" in color_ra: lowSampleWarning = "; Red Vectors were computed with less than 10 values"
    else:
        print("Generating unbinned vector plot")
        binStatus = "Unbinned"
        lowSampleWarning = ""
        color_ra=["k"]
        p_x=x[0,:]
        p_y = y[0, :]
        p_dx=dx
        p_dy=dy
    plt_mean = np.mean(np.hypot(p_dx, p_dy))
    e = np.log10(5.0*plt_mean).round()
    plt_scaleValue=10**e
    if len(dx) > binThresh: Q = plt.quiver(p_x, p_y, p_dx, p_dy,color=color_ra,units="xy")
    else: Q = plt.quiver(p_x, p_y, p_dx, p_dy)
    plt.quiverkey(Q, 0.75, 0.05, plt_scaleValue, r'%5.3f'%(plt_scaleValue), labelpos='S', coordinates='figure', color="k")
    plot_title="%s fit residuals vs. $(X_{ref}, Y_{ref})$ positions\n%s%s"%(imagename,binStatus,lowSampleWarning)
    plt.title(plot_title)
    plt.xlabel(r"$X_{ref}$")
    plt.ylabel(r"$Y_{ref}$")
    if plotDest == "screen":
        plt.show()
    if plotDest == "file":
        plot_file_name = "{}_sci{}_xy_vector_plot.pdf".format(imagename[:9], imagename.split("[SCI,")[1][0])
        plt.savefig(plot_file_name)
        plt.close()
        print("Vector plot saved to file {}".format(plot_file_name))

# ----------------------------------------------------------------------------------------------------------------------



def update_image_wcs_info(tweakwcs_output,imagelist):
    """Write newly computed WCS information to image headers

    Parameters
    ----------
    tweakwcs_output : list
        output of tweakwcs. Contains sourcelist tables, newly computed WCS info, etc. for every chip of every valid
        input image.

    imagelist : list
        list of valid processed images to be updated

    Returns
    -------
    Nothing!
    """
    imgctr = 0
    for item in tweakwcs_output:
        if item.meta['chip'] == 1:  # to get the image name straight regardless of the number of chips
            image_name = imagelist[imgctr]
            if imgctr > 0: #close previously opened image
                print("CLOSE {}".format(hdulist[0].header['FILENAME'])) #TODO: Remove before deployment
                hdulist.flush()
                hdulist.close()
            hdulist = fits.open(image_name, mode='update')
            sciExtDict = {}
            for sciExtCtr in range(1, amutils.countExtn(hdulist) + 1): #establish correct mapping to the science extensions
                sciExtDict["{}".format(sciExtCtr)] = fileutil.findExtname(hdulist,'sci',extver=sciExtCtr)
            imgctr += 1
        updatehdr.update_wcs(hdulist, sciExtDict["{}".format(item.meta['chip'])], item.wcs, wcsname='TWEAKDEV', reusename=True, verbose=True) #TODO: May want to settle on a better name for 'wcsname'
        print()
    print("CLOSE {}".format(hdulist[0].header['FILENAME'])) #TODO: Remove before deployment
    hdulist.flush() #close last image
    hdulist.close()



# ======================================================================================================================


if __name__ == '__main__':
    import argparse
    PARSER = argparse.ArgumentParser(description='Align images')
    PARSER.add_argument('raw_input_list', nargs='+', help='A space-separated list of fits files to align, or a simple '
                    'text file containing a list of fits files to align, one per line')

    PARSER.add_argument( '-a', '--archive', required=False,choices=['True','False'],default='False',help='Retain '
                    'copies of the downloaded files in the astroquery created sub-directories? Unless explicitly set, '
                    'the default is "False".')

    PARSER.add_argument( '-c', '--clobber', required=False,choices=['True','False'],default='False',help='Download and '
                    'overwrite existing local copies of input files? Unless explicitly set, the default is "False".')

    PARSER.add_argument( '-p', '--makeplots', required=False,choices=['True','False'],default='False',help='Generate 2-d'
                    ' vector plots? Unless explicitly set, the default is "False".')

    PARSER.add_argument('-pd', '--plotdest', required=False, choices=['screen', 'file'], default='screen',
                        help='Vector plot destination: "screen" or "file"; "screen" displays the plot in an interactive'
                             ' plotting window; "file" saves the plot as a pdf file with the following file naming '
                             'syntax: <IPPSSOOT>_sci<CHIP NUMBER>_xy_vector_plot.pdf. Default is "screen". '
                             'NOTE: This parameter is only relevant if plots are going to be generated.')

    PARSER.add_argument( '-u', '--update_hdr_wcs', required=False,choices=['True','False'],default='False',help='Write '
                    'newly computed WCS information to image image headers? Unless explicitly set, the default is '
                    '"False".')
    ARGS = PARSER.parse_args()

    # Build list of input images
    input_list = []
    for item in ARGS.raw_input_list:
        if os.path.exists(item):
            with open(item, 'r') as infile:
                fileLines = infile.readlines()
            for fileLine in fileLines:
                input_list.append(fileLine.strip())
        else:
            input_list.append(item)

    # convert text true/false values to Booleans
    archive = convert_string_tf_to_boolean(ARGS.archive)

    clobber = convert_string_tf_to_boolean(ARGS.clobber)

    makeplots = convert_string_tf_to_boolean(ARGS.makeplots)

    update_hdr_wcs = convert_string_tf_to_boolean(ARGS.update_hdr_wcs)

    # Get to it!
    return_value = perform_align(input_list,archive,clobber,makeplots,ARGS.plotdest,update_hdr_wcs)

