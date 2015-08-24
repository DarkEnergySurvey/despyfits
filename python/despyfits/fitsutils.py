#!/usr/bin/env python
# $Id$
# $Rev::                                  $:  # Revision of last commit.
# $LastChangedBy::                        $:  # Author of last commit.
# $LastChangedDate::                      $:  # Date of last commit.

import re
import os
import sys
import pyfits

import despymisc.miscutils as miscutils

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
        self.verb      = kwargs.pop('verb',False)

        # Make sure that filenames and outname are defined
        if not self.filenames: sys.exit("ERROR: must provide input file names")
        if not self.outname:   sys.exit("ERROR: must provide output file name")

        # Output file exits
        if os.path.isfile(self.outname) and self.clobber is False:
            print " [WARNING]: Output file exists, try --clobber option, no file was created"
            return
        
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

            if self.verb: print "# Adding EXTNAME=%s to HDU %s" % (extname,k)
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
            if self.verb: print "# Reading %s --> HDU %s" % (fname,k)
            self.HDU.append(pyfits.open(fname))
            k = k + 1
        return

    def write(self,**kwargs):

        """ Write MEF file with no Primary HDU """
        newhdu = pyfits.HDUList()

        for hdu in self.HDU:
            newhdu.append(hdu[0])# ,hdu[0].header)
        if self.verb: print "# Writing to: %s" % self.outname
        newhdu.writeto(self.outname,clobber=self.clobber)
        return


#######################################################################
def combine_cats(incats, outcat):
    """
    Combine all input catalogs (each with 3 hdus) into a single fits file
    """

    # if incats is comma-separated list, split into python list 
    comma_re = re.compile("\s*,\s*")
    incat_lst = comma_re.split(incats)

    miscutils.fwdebug(3, 'FITSUTILS_DEBUG', "Constructing hdulist object for single fits file")
    # Construct hdulist object to append hdus from individual catalogs to
    hdulist = pyfits.HDUList()

    # Now append the hdus from each input catalog file to the hdulist
    for incat in incat_lst:
        miscutils.fwdebug(3, 'FITSUTILS_DEBUG', "Appending 3 HDUs from cat --> %s" % incat)
        hdulist1 = pyfits.open(incat, mode='readonly')
        hdulist.append(hdulist1[0])
        hdulist.append(hdulist1[1])
        hdulist.append(hdulist1[2])
        #hdulist1.close()

    # And write the full hdulist to the output file
    if os.path.exists(outcat):
        os.remove(outcat)
        miscutils.fwdebug(0, 'FITSUTILS_DEBUG', "Removing pre-existing version of fullcat %s" % outcat)

    miscutils.fwdebug(3, 'FITSUTILS_DEBUG', "Writing results to fullcat --> %s" % outcat)
    hdulist.writeto(outcat)

    miscutils.fwdebug(6, 'FITSUTILS_DEBUG', "Using fits_close to close fullcat --> %s" % outcat)
    hdulist.close()



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
