#!/usr/bin/env python

import fitsio
import numpy 
import time
import argparse
import logging
import os,sys

import despyfits
from despyfits import maskbits
from despyfits import DESImage
from despyfits import compressionhdu as chdu
from despyastro import astrometry
from despyastro import zipper_interp as zipp
from despyastro import CCD_corners 



# Translator for DES_EXT, being nice.
DES_EXT = {
    'SCI' : 'IMAGE',
    'WGT' : 'WEIGHT',
    'MSK' : 'MASK',
    }

def build_parser():

    desc = """
    Creates a co-added MEF fits file from a set of single-plane fits
    containing the SCI/MSK/WGT planes. Interpolates the SCI plane
    using information in the 'custom'-weight mask, and also creates the MSK
    plane to be used by SExtractor for IMAFLAG_ISO.
    """

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--sci_file", default=None,required=True,
                        help="Input SCI fits file")
    parser.add_argument("--msk_file", default=None,required=False,
                        help="Input WGT fits file")
    parser.add_argument("--wgt_file", default=None,required=True,
                        help="Input WGT fits file")
    parser.add_argument("-o","--outname", default=None, required=True,
                        help="Name of output FITS file.")
    parser.add_argument("--clobber", action='store_true', default=False,
                        help="Clobber output MEF fits file")
    parser.add_argument("--add_noise", action='store_true', default=False,
                        help="Add Poisson Noise to the zipper")
    parser.add_argument("--xblock", default=1, type=int,
                        help="Block size of zipper in x-direction")
    # Header options
    parser.add_argument("--band", default=None, type=str, required=False,
                        help="Add (optional) BAND to SCI header if not present")
    parser.add_argument("--magzero", default=None, type=float, required=False,
                        help="Add (optional) MAGZERO to SCI header")
    parser.add_argument("--tilename", default=None, type=str, required=False,
                        help="Add (optional) TILENAME to SCI header")
    parser.add_argument("--tileid", default=None, type=int, required=False,
                        help="Add (optional) TILE_ID to SCI header")

    return parser

def cmdline():
    
    parser = build_parser()
    args = parser.parse_args()

    # Sanity ckecks for inputs...
    # Make sure that outname is defined
    if not args.outname:
        raise ValueError('Undefined outname as %s' % args.outname)
    
    # Output file exits
    if os.path.isfile(args.outname) and args.clobber is False:
        raise ValueError("Output file exists, try --clobber option, no files created")
    return args

def create_logger(level=logging.NOTSET):

    logging.basicConfig(level=level,
                        format='[%(asctime)s] [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger('Merge')
    return logger

def merge(**kwargs):

    """
    Merge the coadded science (SCI), mask (MSK) and weight (WGT) files
    into a single MEF file using fitsio, and perform zipper
    interpolation along columns on the SCI plane using the 'custom'
    MSK weight plane created by SWarp.

    Felipe Menanteau

    """

    sci_file    = kwargs.get('sci_file')
    msk_file    = kwargs.get('msk_file',None)
    wgt_file    = kwargs.get('wgt_file')
    logger      = kwargs.get('logger',None)
    outname     = kwargs.get('outname',False)
    interp_mask = kwargs.get('interp_mask',1)
    #BADPIX_INTERP = kwargs.get('BADPIX_INTERP',maskbits.BADPIX_INTERP)
    BADPIX_INTERP = kwargs.get('BADPIX_INTERP',None)
    # Header options (all optional)
    BAND        = kwargs.get('band',None)
    MAGZERO     = kwargs.get('magzero',None)
    TILENAME    = kwargs.get('tilename',None)
    TILEID      = kwargs.get('tileid',None)

    if not logger:
        logger = create_logger(level=logging.NOTSET)
        
    logger.info("Reading in %s" % sci_file)
    SCI,sci_hdr = fitsio.read(sci_file, ext=0, header=True)
    logger.info("Reading in %s" % wgt_file)
    WGT,wgt_hdr = fitsio.read(wgt_file, ext=0, header=True)
    
    if msk_file:
        logger.info("Reading in %s" % msk_file)
        MSK,msk_hdr = fitsio.read(msk_file, ext=0, header=True)
        # Make the MSK for MSK-like weight
        MSK = numpy.where(MSK == 0,1,0)
    else: # Create mask from WGT plane
        MSK = numpy.copy(WGT)
        MSK = numpy.where(MSK == 0,1,0)
        msk_hdr = wgt_hdr

    # Make sure that we do not interpolate over zeroes
    MSK  = numpy.where(SCI == 0,0,MSK)

    # Perform column interpolation -- Axis=2
    if BADPIX_INTERP:
        SCI,MSK = zipp.zipper_interp(SCI,MSK,interp_mask,axis=2, **kwargs)
        # Interpolate the WGT plane if we don't have a msk_file
        if not msk_file:
            WGT,MSK = zipp.zipper_interp(WGT,MSK,interp_mask,axis=2, ydilate=10, **kwargs)
    else:
        SCI     = zipp.zipper_interp(SCI,MSK,interp_mask,axis=2, **kwargs)
        # Interpolate the WGT plane if we don't have a msk_file
        if not msk_file:
            WGT     = zipp.zipper_interp(WGT,MSK,interp_mask,axis=2, ydilate=10,**kwargs)

    # Update compression settings
    sci_hdr = DESImage.update_hdr_compression(sci_hdr,'SCI')
    msk_hdr = DESImage.update_hdr_compression(msk_hdr,'MSK')
    wgt_hdr = DESImage.update_hdr_compression(wgt_hdr,'WGT')

    # Add corners, centers and extend
    sci_hdr = CCD_corners.update_DESDM_corners(sci_hdr,get_extent=True, verb=False)
    msk_hdr = CCD_corners.update_DESDM_corners(msk_hdr,get_extent=True, verb=False)

    # Add BAND if present
    if BAND:
        record={'name':'BAND', 'value':BAND, 'comment':'Short name for filter'}
        sci_hdr.add_record(record)

    # Add MAGZERO if present
    if MAGZERO:
        record={'name':'MAGZERO', 'value':MAGZERO, 'comment':'Mag Zero-point in magnitudes/s'}
        sci_hdr.add_record(record)

    # Add TILENAME if present
    if TILENAME:
        record={'name':'TILENAME', 'value':TILENAME, 'comment':'DES Tilename'}
        sci_hdr.add_record(record)

    # Add TILEID if present
    if TILEID:
        record={'name':'TILEID', 'value':TILEID, 'comment':'Tile ID for DES Tilename'}
        sci_hdr.add_record(record)


    # Add to image history
    sci_hdr['HISTORY'] = time.asctime(time.localtime()) + \
                         ' column_interp over mask 0x{:04X}'.format(interp_mask)
    # Write it out now
    logger.info("Writing %s" % outname)
    ofits = fitsio.FITS(outname,'rw',clobber=True)
    ofits.write(SCI,extname='SCI',header=sci_hdr)
    ofits.write(MSK,extname='MSK',header=msk_hdr)
    ofits.write(WGT,extname='WGT',header=wgt_hdr)
    ofits.close()

    return

