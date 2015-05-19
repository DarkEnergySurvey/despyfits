#!/usr/bin/env python
# $Id: special_metadata_funcs.py 38203 2015-05-17 14:28:09Z mgower $
# $Rev:: 38203                            $:  # Revision of last commit.
# $LastChangedBy:: mgower                 $:  # Author of last commit.
# $LastChangedDate:: 2015-05-17 09:28:09 #$:  # Date of last commit.

"""
Specialized functions for computing metadata
"""

import calendar
import re

import despyfits.fitsutils as fitsutils
import despymisc.create_special_metadata as spmeta


######################################################################
# !!!! Function name must be all lowercase
# !!!! Function name must be of pattern func_<header key>
######################################################################


######################################################################
def func_band(filename, hdulist=None, whichhdu=None):
    """ Create band from the filter keyword """

    if hdulist is None:
        hdulist = fits_open(filename, 'readonly')

    filter = fitsutils.get_hdr_value(hdulist, 'FILTER') 
    return spmeta.create_band(filter)


######################################################################
def func_camsym(filename, hdulist=None, whichhdu=None):
    """ Create camsys from the INSTRUME keyword """
    if hdulist is None:
        hdulist = fits_open(filnam,'readonly')

    instrume = fitsutils.get_hdr_value(hdulist, 'INSTRUME')
    
    return spmeta.create_camsym(intrume)


######################################################################
def func_nite(filename, hdulist=None, whichhdu=None):
    """ Create nite from the DATE-OBS keyword """

    if hdulist is None:
        hdulist = fits_open(filename, 'readonly')

    date_obs = fitsutils.get_hdr_value(hdulist, 'DATE-OBS')
    return spmeta.create_nite(date_obs)


######################################################################
def func_objects(filename, hdulist=None, whichhdu=None):
    """ return the number of objects in fits catalog """

    if hdulist is None:
        hdulist = fits_open(filename, 'readonly')

    objects = fitsutils.get_hdr_value(hdulist, 'NAXIS2', 'LDAC_OBJECTS')

    return objects


######################################################################
def func_field(filename, hdulist=None, whichhdu=None):
    """ return the field from OBJECT fits header value """

    if hdulist is None:
        hdulist = fits_open(filename, 'readonly')

    try:
        object = fitsutils.get_hdr_value(hdulist, 'OBJECT')
    except:
        object = fitsutils.get_hdr_value(hdulist, 'OBJECT', 'LDAC_IMHEAD')

    return spmeta.create_field(object)
