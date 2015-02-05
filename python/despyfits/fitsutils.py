#!/usr/bin/env python
# $Id: fitsutils.py 18485 2014-01-29 15:57:50Z mgower $
# $Rev:: 18485                            $:  # Revision of last commit.
# $LastChangedBy:: mgower                 $:  # Author of last commit.
# $LastChangedDate:: 2014-01-29 09:57:50 #$:  # Date of last commit.

import re
import os
import sys
import pyfits

""" Miscellaneous generic support functions for fits files """

class makeMEF(object):

    """
    A Class to create a MEF fits files using pyfits, we might want to
    migrated this to use fitsio in the future.

    Felipe Menanteau, NCSA Aug 2014.
    """

    # -----------------------------------
    # Translator for DES_EXT, being nice.
    DES_EXT = {}
    DES_EXT['SCI'] = 'IMAGE'
    DES_EXT['WGT'] = 'WEIGHT'
    DES_EXT['MSK'] = 'MASK'
    # -----------------------

    def __init__(self, **kwargs):


        self.filenames = kwargs.pop('filenames',False)
        self.outname   = kwargs.pop('outname',False)
        self.clobber   = kwargs.pop('clobber',False)
        self.extnames  = kwargs.pop('extnames',None)

        # Make sure that filenames and outname are defined
        if not self.filenames: sys.exit("ERROR: must provide input file names")
        if not self.outname:   sys.exit("ERROR: must provide output file name")

        # Output file exits
        if os.path.isfile(self.outname) and self.clobber is False:
            raise Warning("Output file exists, try --clobber option, no file was created")
            return 1
        
        # Get the Pyfits version as a float
        self.pyfitsVersion = float(".".join(pyfits.__version__.split(".")[0:2]))

        self.read()
        if self.extnames:
            self.addEXTNAME()
        self.write()

        return

    def addEXTNAME(self,**kwargs):

        """Add a user-provided list of extension names to the MEF"""

        if len(self.extnames) != len(self.filenames):
            sys.exit("ERROR: number of extension names doesn't match filenames")
            return

        k = 0
        for extname,hdu in zip(self.extnames,self.HDU):

            print "# Adding EXTNAME=%s to HDU %s" % (extname,k)
            # Method for pyfits < 3.1
            if self.pyfitsVersion < 3.1: 
                hdu[0].header.update('EXTNAME',extname, 'Extension Name' ,after='NAXIS2')
                if extname in makeMEF.DES_EXT.keys():
                    hdu[0].header.update('DES_EXT',makeMEF.DES_EXT[extname], 'DESDM Extension Name' ,after='EXTNAME')
            else:   
                hdu[0].header.set('EXTNAME', extname, 'Extension Name', after='NAXIS2') 
                if extname in makeMEF.DES_EXT.keys():
                    hdu[0].header.set('DES_EXT',makeMEF.DES_EXT[extname], 'DESDM Extension Name' ,after='EXTNAME')

            k = k + 1
        return
    
    def read(self,**kwargs):

        """ Read in the HDUs using pyfits """
        self.HDU = []
        k = 0
        for fname in self.filenames:
            print "# Reading %s --> HDU %s" % (fname,k)
            self.HDU.append(pyfits.open(fname))
            k = k + 1
        return

    def write(self,**kwargs):

        """ Write MEF file with no Primary HDU """
        newhdu = pyfits.HDUList()

        for hdu in self.HDU:
            newhdu.append(hdu[0])# ,hdu[0].header)
        print "# Writing to: %s" % self.outname
        newhdu.writeto(self.outname,clobber=self.clobber)
        return



#######################################################################
def get_hdr(hdulist, whichhdu):

    if whichhdu is None:
        whichhdu = 'Primary'

    try:
        whichhdu = int(whichhdu)  # if number, convert type
    except ValueError:
        whichhdu = whichhdu.upper()

    hdr = None
    if whichhdu == 'LDAC_IMHEAD':
        hdr = get_ldac_imhead_as_hdr(hdulist['LDAC_IMHEAD'])
    else:
        try:
            hdr = hdulist[whichhdu].header
        except KeyError:
            # certain versions of pyfits always refer to Primary HDU only as Primary regardless of extname
            if hdulist[0].header['EXTNAME'] == whichhdu:
                hdr = hdulist[0].header
    return hdr


#######################################################################
def get_hdr_value(hdulist, key, whichhdu=None):
    ukey = key.upper()

    hdr = get_hdr(hdulist, whichhdu)
    val = hdr[ukey]

    return val

#######################################################################
def get_ldac_imhead_as_cardlist(imhead):
    data = imhead.data
    cards = []
    for cd in data[0][0]:
        cards.append(pyfits.Card.fromstring(cd))
    return cards


#######################################################################
def get_ldac_imhead_as_hdr(imhead):
    hdr = pyfits.Header(get_ldac_imhead_as_cardlist(imhead))
    return hdr
