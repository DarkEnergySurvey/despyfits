#!/usr/bin/env python
"""Class to manage DES images

In this context, "image" means the data from one CCD from one exposure,
possibly accompanied by mask and weight maps.
"""

# imports

import logging
import platform
import ctypes
import re
from collections import namedtuple
from tempfile import mkdtemp
from os import path
import shutil
import os

import numpy as np

import fitsio
from fitsio import FITSHDR
from despyfits import maskbits
from despyfits import compressionhdu as chdu
from despyfits.DESFITSInventory import DESFITSInventory
from despyastro.CCD_corners import update_DESDM_corners

# constants

image_hdu = None
mask_hdu = 1
weight_hdu = 2
weight_dtype = np.dtype(np.float32)
variance_dtype = np.dtype(np.float32)
data_dtype = np.dtype(np.float32)
pass_fortran = False
# Indirect-write behaviour based on the enviroment variables
# 1. For use_indirect_write
if os.environ.get('DESPYFITS_USE_INDIRECT_WRITE'):
    use_indirect_write = True
else:
    use_indirect_write = False
# 2. For indirect_write_prefix
if os.environ.get('DESPYFITS_INDIRECT_WRITE_PREFIX'):
    indirect_write_prefix = os.environ.get('DESPYFITS_INDIRECT_WRITE_PREFIX')
else:
    indirect_write_prefix = '/tmp/desimage-'
# 3. For adding DESDM_PIPEPROD/DESDM_PIPERVER keys
if os.environ.get('DESPYFITS_PIPEKEYS_WRITE'):
    pipekeys_write = True
else:
    pipekeys_write = False

mask_is_unsigned = False
if mask_is_unsigned:
    mask_dtype = np.dtype(np.uint16)
    mask_ctype = ctypes.c_ushort
else:
    mask_dtype = np.dtype(np.int16)
    mask_ctype = ctypes.c_short


logger = logging.getLogger('DESImage')
if len(logger.handlers) < 0:
    logger.addHandler(logging.StreamHandler())


# exception classes

class MissingData(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class MissingMask(Exception):
    pass

class MissingWeight(Exception):
    pass

class UnexpectedDesExt(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class MissingFITSKeyword(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class BadFITSSectionSpec(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class TooManyDataHDUs(Exception):
    pass

class TooManyMaskHDUs(Exception):
    pass

class TooManyWeightHDUs(Exception):
    pass

# interface functions

# decorators

def indirect_write(fname_idx):
    # fname_idx indicate which argument of the wrapped
    # method specifices the filename
    def wrap(fn):
        def writeme(*param_tuple, **kwds_dict):
            if not use_indirect_write:
                return fn(*param_tuple, **kwds_dict)

            param_list = list(param_tuple)
            dest_fname = param_tuple[fname_idx]
            tmp_dir = mkdtemp(prefix=indirect_write_prefix)
            tmp_fname = path.join(tmp_dir, path.basename(dest_fname))
            param_list[fname_idx] = tmp_fname
            param_tuple = tuple(param_list)
            try:
                # If the file exists and we are trying to update it
                # we need to copy the file
                shutil.copy2(dest_fname, tmp_fname)
            except IOError:
                pass

            try:
                # Do it
                result = fn(*param_tuple, **kwds_dict)
                shutil.move(tmp_fname, dest_fname)
                shutil.rmtree(tmp_dir)
                return result
            except:
                # If we fail, we need to clean up after ourselves
                shutil.rmtree(tmp_dir)
                raise
        return writeme
    return wrap

# classes

class HeaderDifference(object):
    def __init__(self, im1, im2):
        self.match = {}
        self.diff = {}

        for k in set(im1.header.keys()) | set(im2.header.keys()):
            if len(k)>8:
                # Invalid FITS keyword
                continue

            v = [None, None]

            if k in im1.header.keys():
                v[0] = im1.header[k]
            if k in im2.header.keys():
                v[1] = im2.header[k]


            if not v[0]==v[1]:
                self.diff[k] = v
            else:
                self.match[k] = v[0]

    def __str__(self):
        s = ''
        for k in self.diff:
            s += "Difference: %s\n" % k
            s += "       im1=>" + str(self.diff[k][0]) + "<=\n"
            s += "       im2=>" + str(self.diff[k][1]) + "<=\n"
        return s

class DESSingleImage(object):

    # If we index this object, assume we are after keywords in the header
    def __getitem__(self, key):
        if key in self.header.keys():
            return self.header[key]
        else:
            return self.pri_hdr[key]

    def __setitem__(self, key, value):
        self.header[key] = value

    def write_key(self, key, value, comment=None):
        """
        Add or alter a header keyword/value pair, with optional comment field
        """
        if comment is None:
            self.header[key] = value
        else:
            self.header.add_record( {'name':key, 'value':value, 'comment':comment})

class DESDataImage(DESSingleImage):

    def __init__(self, data, header={}, pri_hdr=None, sourcefile=None):
        """Create a new DESDataImage

        :Parameters:
            - `data`: the numpy data array
            - `header`: the header
            - `pri_hdr`: the primary header of the FITS file

        """
        self.data = data
        self.sourcefile = sourcefile

        if getattr(header, "add_record", None) is None:
            self.header = FITSHDR(header)
        else:
            self.header = header

        if pri_hdr is None:
            self.pri_hdr = self.header

        if getattr(pri_hdr, "add_record", None) is None:
            self.pri_hdr = FITSHDR(pri_hdr)
        else:
            self.pri_hdr = pri_hdr

    @classmethod
    def load(cls, filename, image_hdu=None):
        """Load from a FITS file

        :Parameters:
            - `filename`: the name of the FITS file from which to load
            - `image_hdu`: the HDU index with the data image

        """
        if image_hdu is None:
            ext = 1 if filename.endswith('.fz') else 0
        else:
            ext = image_hdu

        data, header = fitsio.read(filename, ext=ext, header=True)

        im = cls(data, header, sourcefile=filename)
        return im

    @classmethod
    def load_from_open(cls, fits, image_hdu):
        """Load from a FITS file

        :Parameters:
            - `fits`: an already-opened FITS object
            - `image_hdu`: the HDU index with the data image

        """
        header = fits[image_hdu].read_header()
        data = fits[image_hdu].read()
        pri_hdr = fits[0].read_header()
        im = cls(data, header, pri_hdr)
        return im

    @indirect_write(1)
    def save(self, filename, ext=None):
        """Save to a FITS file


        :Parameters:
            - `filename`: the name of the FITS file
            - `ext`: the index of the FITS hdu into which to save the image

        """
        if ext is None:
            fitsio.write(filename, self.data, header=self.header,
                         clobber=True)
        else:
            with fitsio.FITS(filename, fitsio.READWRITE) as fits:
                while len(fits) <= ext:
                    fits.create_image_hdu(self.data)
                fits.write(self.data, header=self.header, ext=ext, clobber=True)

    def copy_header_info(self, source, keywords, require=False):
        """
        Copy keyword/value pairs from source into header of this object.
        :Parameters:
            - `source`: source of key/value pairs with dict-like behavior
            - `keywords`: iterable list of the keywords to copy from source
            - `require`: if True, exception is generated if a desired keyword is
                         absent from source.
        """
        for kw in keywords:
            try:
                value = source[kw]
                self.header[kw] = value
            except (ValueError,KeyError):
                if require:
                    raise KeyError('copy_header_info did not find required keyword ' + kw)
        return

    @property
    def cstruct(self):
        """Return a structure passable to C libraries using ctypes
        """

        if pass_fortran:
            self.data = np.asfortranarray(localize_numpy_array(self.data, data_dtype))
        else:
            self.data = localize_numpy_array(self.data, data_dtype)

        self._cstruct = DESImageCStruct(self)
        return self._cstruct

class DESImage(DESDataImage):

    def __init__(self, init_data=False, init_mask=False, init_weight=False,
                 shape=(4096, 2048), sourcefile=None):
        """Create a new DESImage

        Create an empty DESImage

        In mast cases, DESImage.create or DESImage.load will be safer
        and more covenient.
        """

        self.data = np.zeros(shape, dtype=data_dtype) if init_data else None

        self.mask = None
        if init_mask:
            self.init_mask()

        self.weight = None
        if init_weight:
            self.init_weight()

        self.variance = None
        self.header = FITSHDR({})
        self.pri_hdr = self.header
        self.mask_hdr = FITSHDR({})
        self.weight_hdr = FITSHDR({})
        self.sourcefile = sourcefile

    @classmethod
    def create(cls, data_im, mask=None, weight=None):
        """Create a new DESImage from a numpy array with the image data
        and (optionally) a header, mask, and/or weight map.

        :Parameters:
            - `data_im`: the data image (a DESDataImage object is expected)
            - `mask`: the mask (a numpy array)
            - `weight`: the weight map (a numpy array)

        @returns: a new DESImage object

        """
        im = cls()
        im.data = data_im.data
        im.header = data_im.header
        im.pri_hdr = data_im.pri_hdr
        im.mask = mask
        im.weight = weight
        return im

    def init_mask(self, mask=None):
        self.mask = mask.astype(mask_dtype) if mask is not None \
                    else np.zeros_like(self.data, dtype=mask_dtype)

    def init_weight(self, weight=None):
        self.weight = weight.astype(weight_dtype) if weight is not None \
                      else np.ones_like(self.data, dtype=weight_dtype)

    def get_weight(self):
        """Return a weight map, convernting from variance if necessary
        """
        if self.weight is None and self.variance is not None:
            w = 1.0/self.variance
            self.variance = None
            self.weight = w
        return self.weight

    def get_variance(self):
        """Return a variance map, concerting from weight if necessary
        """
        if self.variance is None and self.weight is not None:
            v = 1.0/self.weight
            self.weight = None
            self.variance = v
        return self.variance

    @classmethod
    def load(cls, filename,
             assign_default_mask=False,
             assign_default_weight=False,
             wgt_is_variance=False,
             ccdnum=None):
        """Load from a FITS file

        :Parameters:
            - `filename`: the name of the FITS file from which to load
            - `assign_default_mask`: create a default empty mask if none present
            - `assign_default_weight`: create default weights if none present
            - `ccdnum`: which CCD

        @returns: a new DESImage object with data from the FITS file
        """
        fits_inventory = DESFITSInventory(filename)

        # Find and load the data HDU

        data_hdus = fits_inventory.scis
        if ccdnum is not None:
            ccd_hdus = set(fits_inventory.ccd_hdus(ccdnum))
            data_hdus = sorted(set(data_hdus) & ccd_hdus)

        if len(data_hdus)==0:
            data_hdus = fits_inventory.raws
            if ccdnum is not None:
                data_hdus = sorted(set(data_hdus) & ccd_hdus)

        # Create the new object
        if len(data_hdus)>0:
            ext = data_hdus[0]
            logger.info("Loading data from HDU %d" % ext)
            data_im = DESDataImage.load(filename, image_hdu=ext)
            im = cls.create(data_im)
        else:
            im = cls()
            im.pri_hdr = fits_inventory.hdr[0]

        im.sourcefile = filename

        # Find and load the mask HDU

        if len(fits_inventory.masks) > 1:
            raise TooManyMaskHDUs
        elif len(fits_inventory.masks) == 1:
            ext = fits_inventory.masks[0]
            logger.info("Loading mask from HDU %d" % ext)
            im.mask, im.mask_hdr = fitsio.read(
                filename, ext=ext, header=True)
        elif assign_default_mask:
            im.init_mask()
        if im.mask is not None:
            im.mask = localize_numpy_array(im.mask, new_dtype=mask_dtype)

        # Find and load the weight HDU

        if len(fits_inventory.weights) > 1:
            raise TooManyWeightHDUs
        elif len(fits_inventory.weights) == 1:
            ext = fits_inventory.weights[0]
            logger.info("Loading weights from HDU %d" % ext)
            if wgt_is_variance:
                im.variance, im.weight_hdr = fitsio.read(
                    filename, ext=ext, header=True)
            else:
                im.weight, im.weight_hdr = fitsio.read(
                    filename, ext=ext, header=True)

        elif assign_default_weight:
            im.init_weight()

        # Initialze the data if there isn't any yet
        if im.data is None:
            try:
                shape = im.mask.shape
            except AttributeError:
                shape = im.weight.shape
            im.data = np.zeros(shape, dtype=data_dtype)

        return im

    @indirect_write(1)
    def save(self, filename, save_data=True, save_mask=None, save_weight=None):
        """Save to a FITS file

        If the file exists and only some of the FITS extensions need to be
        written, a little time can be saved by setting the save_xxx element(s)
        of those HDUs that need not be saved to False. By default, all are
        written.

        :Parameters:
            - `filename`: the name of the FITS file
            - `save_data`: true if the data array is to be saved
            - `save_mask`: true if the mask array is to be saved
            - `save_weight`: true if the weight array is to be saved

        """

        try:
            has_mask = self.mask is not None
        except AttributeError:
            has_mask = False

        try:
            has_weight = self.get_weight() is not None
        except AttributeError:
            has_weight = False

        save_mask = save_mask if save_mask is not None else has_mask
        if save_mask and not has_mask:
            raise MissingMask

        save_weight = save_weight if save_weight is not None else has_weight
        if save_weight and not has_weight:
            raise MissingWeight

        if image_hdu is None:
            data_hdu = 1 if filename.endswith('.fz') else 0
        else:
            data_hdu = image_hdu

        try:
            with fitsio.FITS(filename, fitsio.READONLY) as fits:
                max_init_hdu = len(fits)
        except:
            max_init_hdu = 0

        with fitsio.FITS(filename, fitsio.READWRITE) as fits:

            # Make sure the file exists and has the expected HDUs
            # Create any HDUs that are needed but don't already exist
            hdu_list = [data_hdu]
            if has_mask:
                hdu_list.append(mask_hdu)
            if has_weight:
                hdu_list.append(weight_hdu)

            max_hdu = max(hdu_list)

            for hdu in range(max_init_hdu, max_hdu+1):

                if hdu==data_hdu and save_data:
                    # Update the hdr with the proper FZ keywords
                    logger.info("Creating SCI HDU %d and relevant FZ*/DES_EXT/EXTNAME keywords" % data_hdu)
                    self.header = update_hdr_compression(self.header,'SCI')
                    # Calculate coordinates for ccd center and corners and update the header
                    logger.info("Calculating CCD corners/center/extern keywords for SCI HDU %d " % data_hdu)
                    self.header = update_DESDM_corners(self.header,get_extent=True, verb=False)
                    if pipekeys_write:
                        logger.info("Inserting EUPS PIPEPROD and PIPEVER to SCI HDU")
                        self.header = insert_eupspipe(self.header)
                    fits.write(self.data,extname='SCI',header=self.header)
                    save_data = False
                elif has_mask and hdu==mask_hdu and save_mask:
                    # Update the hdr with the proper FZ keywords
                    logger.info("Creating MSK HDU %d and relevant FZ*/DES_EXT/EXTNAME keywords" % mask_hdu)
                    self.mask_hdr = update_hdr_compression(self.mask_hdr,'MSK')
                    # Calculate coordinates for ccd center and corners and update the header
                    logger.info("Calculating CCD corners/center/extern keywords for MSK HDU %d " % mask_hdu)
                    self.mask_hdr = update_DESDM_corners(self.mask_hdr,get_extent=True, verb=False)
                    fits.write(self.mask,extname='MSK',header=self.mask_hdr)
                    save_mask = False
                elif has_weight and hdu==weight_hdu and save_weight:
                    logger.info("Creating WGT HDU %d and relevant FZ*/DES_EXT/EXTNAME keywords" % weight_hdu)
                    # Update the hdr with the proper FZ keywords
                    self.weight_hdr = update_hdr_compression(self.weight_hdr,'WGT')
                    fits.write(self.weight,extname='WGT',header=self.weight_hdr)
                    save_weight = False
                else:
                    fits.create_image_hdu(np.zeros((1,1)), extname='DUMMY')

            if save_data:
                hdr = fits[data_hdu].read_header()
                if hdr['DES_EXT'].rstrip() != 'IMAGE':
                    raise UnexpectedDesExt('image not in expected HDU')
                fits[data_hdu].write(self.data)
                fits[data_hdu].write_keys(self.header)

            if save_mask:
                hdr = fits[mask_hdu].read_header()
                if hdr['DES_EXT'].rstrip() != 'MASK':
                    raise UnexpectedDesExt('mask not in expected HDU')

                if 'DES_EXT' in self.mask_hdr:
                    if self.mask_hdr['DES_EXT'].rstrip() != 'MASK':
                        raise UnexpectedDesExt('mask not in expected HDU')
                else:
                    self.mask_hdr['DES_EXT']='MASK'

                fits[mask_hdu].write(self.mask)
                fits[mask_hdu].write_keys(self.mask_hdr)

            if save_weight:
                hdr = fits[weight_hdu].read_header()
                if hdr['DES_EXT'].rstrip() != 'WEIGHT':
                    raise UnexpectedDesExt('weight not in expected HDU')

                if 'DES_EXT' in self.weight_hdr:
                    if self.weight_hdr['DES_EXT'].rstrip() != 'WEIGHT':
                        raise UnexpectedDesExt('weight not in expected HDU')
                else:
                    self.weight_hdr['DES_EXT']='WEIGHT'

                fits[weight_hdu].write(self.weight)
                fits[weight_hdu].write_keys(self.weight_hdr)


    @property
    def cstruct(self):
        """Return a structure passable to C libraries using ctypes
        """


        self._cstruct = DESImageCStruct(self)
        return self._cstruct

    def compare(self, im):
        """Show differences between images

        :Parameters:
            - `im`: the comparison image

        @Returns: a difference image

        """
        header_diff = HeaderDifference(self, im)
        hdus = (('science', self.data, im.data),
                ('mask', self.mask, im.mask),
                ('weight', self.weight, im.weight))

        diff_im = DESImage()
        for ext, im1, im2 in hdus:
            if ext=='science':
                diff_im.data = im1-im2
            elif ext=='mask':
                diff_im.mask = np.bitwise_xor(im1, im2)
            elif ext=='weight':
                diff_im.weight = im1-im2

        comparison = DESImageComparison(header_diff, diff_im)
        return comparison


class DESBPMImage(DESSingleImage):

    def __init__(self, bpm, header={}, pri_hdr=None, sourcefile=None):
        """Create a new DESBPMImage

        :Parameters:
            - `bpm`: the numpy data array
            - `header`: the header
            - `pri_hdr`: the primary header of the FITS file

        """
        self.mask = bpm
        self.sourcefile = sourcefile

        if getattr(header, "add_record", None) is None:
            self.header = FITSHDR(header)
        else:
            self.header = header

        if pri_hdr is None:
            self.pri_hdr = self.header

        if getattr(pri_hdr, "add_record", None) is None:
            self.pri_hdr = FITSHDR(pri_hdr)
        else:
            self.pri_hdr = pri_hdr

    @classmethod
    def load(cls, filename, bpm_hdu=None):
        """Load from a FITS file

        :Parameters:
            - `filename`: the name of the FITS file from which to load
            - `image_hdu`: the HDU index with the data image

        """

        if bpm_hdu is None:
            fits_inventory = DESFITSInventory(filename)
            hdus = fits_inventory.bpms
            if len(hdus)>1:
                raise TooManyMaskHDUs
            ext = hdus[0]
        else:
            ext = dbm_hdu

        data, header = fitsio.read(filename, ext=ext, header=True)

        bpm = cls(data, header, sourcefile=filename)
        return bpm

    @indirect_write(1)
    def save(self, filename):
        """Save to a FITS file

        :Parameters:
            - `filename`: the name of the FITS file
        """
        fitsio.write(filename, self.mask, header=self.header,
                     clobber=True)

    @property
    def cstruct(self):
        """Return a structure passable to C libraries using ctypes
        """

        if pass_fortran:
            self.mask = np.asfortranarray(localize_numpy_array(self.mask, mask_dtype))
        else:
            self.mask = localize_numpy_array(self.mask, mask_dtype)

        self._cstruct = DESImageCStruct(self)
        return self._cstruct


# internal functions & classes

DESImageComparisonNT = namedtuple('DESImageComparisonNT',
                                  ('header', 'diff_im'))
class DESImageComparison(DESImageComparisonNT):
    @property
    def mismatched_keywords(self):
        mk = set(
            [k for k in self.header.diff] )
        return mk

    @property
    def data_match(self):
        m = np.count_nonzero(self.diff_im.data) == 0
        return m

    @property
    def mask_match(self):
        m = np.count_nonzero(self.diff_im.mask) == 0
        return m

    @property
    def weight_match(self):
        m = np.count_nonzero(self.diff_im.weight) == 0
        return m

    def match(self, ignore=set()):
        differing_keywords = self.mismatched_keywords - set(ignore)

        m = len(differing_keywords)==0 \
            and self.data_match and self.weight_match and self.mask_match

        return m

    def log(self, logger, ref):
        logger.debug('Image size: %d', self.diff_im.data.size)
        logger.debug('Data differences: %d',
                     np.count_nonzero(self.diff_im.data))
        logger.debug('Weight differences: %d',
                     np.count_nonzero(self.diff_im.weight))
        logger.debug(
            'BADPIX_BPM differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_BPM),
            np.count_nonzero(ref.mask & maskbits.BADPIX_BPM))
        logger.debug(
            'BADPIX_SATURATE differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_SATURATE),
            np.count_nonzero(ref.mask & maskbits.BADPIX_SATURATE))
        logger.debug(
            'BADPIX_INTERP differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_INTERP),
            np.count_nonzero(ref.mask & maskbits.BADPIX_INTERP))
        logger.debug(
            'BADPIX_THRESHOLD differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_THRESHOLD),
            np.count_nonzero(ref.mask & maskbits.BADPIX_THRESHOLD))
        logger.debug(
            'BADPIX_LOW differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_LOW),
            np.count_nonzero(ref.mask & maskbits.BADPIX_LOW))
        logger.debug(
            'BADPIX_CRAY differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_CRAY),
            np.count_nonzero(ref.mask & maskbits.BADPIX_CRAY))
        logger.debug(
            'BADPIX_STAR differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_STAR),
            np.count_nonzero(ref.mask & maskbits.BADPIX_STAR))
        logger.debug(
            'BADPIX_TRAIL differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_TRAIL),
            np.count_nonzero(ref.mask & maskbits.BADPIX_TRAIL))
        logger.debug(
            'BADPIX_EDGEBLEED differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_EDGEBLEED),
            np.count_nonzero(ref.mask & maskbits.BADPIX_EDGEBLEED))
        logger.debug(
            'BADPIX_SSXTALK differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_SSXTALK),
            np.count_nonzero(ref.mask & maskbits.BADPIX_SSXTALK))
        logger.debug(
            'BADPIX_EDGE differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_EDGE),
            np.count_nonzero(ref.mask & maskbits.BADPIX_EDGE))
        logger.debug(
            'BADPIX_STREAK differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_STREAK),
            np.count_nonzero(ref.mask & maskbits.BADPIX_STREAK))
        logger.debug(
            'BADPIX_FIX differences: %d/%d',
            np.count_nonzero(self.diff_im.mask & maskbits.BADPIX_FIX),
            np.count_nonzero(ref.mask & maskbits.BADPIX_FIX))


def scan_fits_section(hdr, keyword):
    str_value = hdr[keyword]
    pattern = r"\[(\d+):(\d+),(\d+):(\d+)\]"
    m = re.match(pattern, str_value)
    if len(m.groups()) != 4:
        raise BadFITSSectionSpec("%s %s" % (keyword, str_value))

    values = [int(s) for s in m.groups()]
    return values

def section2slice(section, reorder=False):
    """
    Parse an IRAF/FITS section specification string and convert to a numpy slice
    specification (2-element tuple of slices).  Converts from 1-indexed to 0-indexed,
    end from last element to 1-past-last element, and swaps index order.

    :Parameters:
      - `section`: string-valued IRAF/FITS section specification
      - `reorder`: if True, will swap start/stop to insure start<=stop.

    :Returns:
      - 2-element tuple of slices, which can index numpy arrays
    """
    pattern = r"\[(\d+):(\d+),(\d+):(\d+)\]"
    m = re.match(pattern, section)
    if len(m.groups()) != 4:
        raise BadFITSSectionSpec("%s" % section)
    values = [int(s) for s in m.groups()]
    if reorder and values[1]<values[0]:
        values[0],values[1] = values[1],values[0]
    if reorder and values[3]<values[2]:
        values[2],values[3] = values[3],values[2]

    return (slice(values[2]-1,values[3]),
            slice(values[0]-1,values[1]))


def slice2section(s):
    """
    Convert numpy 2d slice specification into an IRAF/FITS section specification string.
    Converts from 0-indexed to 1-indexed, 1-past-last to last at end, and swaps index order.
    Requires explicit non-negative start and stop values in the slice.  Only

    :Parameters:
      - `s`: 2-element tuple of slices, which can index numpy arrays

    :Returns:
      - string-valued IRAF/FITS section specification
    """
    values = (s[1].start, s[1].stop, s[0].start, s[0].stop)
    if None in values or np.any(np.array(values)<0):
        raise BadFITSSectionSpec("Bad slice2section input: " + str(s))
    return "[%d:%d,%d:%d]" % values

lib_ext = {'Linux': 'so',
           'Darwin': 'dylib'}
try:
    libdesimage = ctypes.CDLL(
        'libdesimage.' + lib_ext[platform.system()])
except KeyError:
    raise RuntimeError, ("Unknown platform: " + platform.system())


set_desimage = libdesimage.set_desimage
CCDNUM2 = ctypes.c_int.in_dll(libdesimage, 'ccdnum2').value

SevenLongs = ctypes.c_long * 7
FourInts = ctypes.c_int * 4


class DESImageCStruct(ctypes.Structure):
    """Partially simulate desimage from imsupport
    """

    #
    # This needs to match ../include/desimage.h
    #  see desimage from imsupport/include/imreadsubs.h
    #  for guidance, but this does not attempt to be compatible
    _fields_ = [
        ('npixels', ctypes.c_long),
        ('axes', SevenLongs),
        ('exptime', ctypes.c_float),
        ('ampsecan', FourInts),
        ('ampsecbn', FourInts),
        ('saturateA', ctypes.c_float),
        ('saturateB', ctypes.c_float),
        ('gainA', ctypes.c_float),
        ('gainB', ctypes.c_float),
        ('rdnoiseA', ctypes.c_float),
        ('rdnoiseB', ctypes.c_float),
        ('image', ctypes.POINTER(ctypes.c_float)),
        ('variance', ctypes.POINTER(ctypes.c_float)),
        ('weight', ctypes.POINTER(ctypes.c_float)),
        ('mask', ctypes.POINTER(ctypes.c_short))
    ]

    def __init__(self, im=None):
        if im is not None:
            self.create(im)
        else:
            self.image = None
            self.variance = None
            self.weight = None
            self.mask = None
            self.npixels = 0

    def create(self, im):
        if isinstance(im, DESBPMImage):
            im_shape = im.mask.shape
            self.npixels = im.mask.size
        else:
            im_shape = im.data.shape
            self.npixels = im.data.size
        self.axes = SevenLongs(im_shape[1], im_shape[0], 0, 0, 0, 0, 0)

        try:
            self.ampsecan = FourInts(*scan_fits_section(im, 'AMPSECA'))
        except ValueError:
            logger.warning("Keyword AMPSECA not defined")

        try:
            self.ampsecbn = FourInts(*scan_fits_section(im, 'AMPSECB'))
        except ValueError:
            logger.warning("Keyword AMPSECB not defined")

        def get_header_value(keyword, default):
            missing = (keyword not in im.header) and (keyword not in im.pri_hdr)
            if missing:
                logging.warning("Keyword " + keyword + " not defined")

            value = default if missing else im[keyword]
            return value

        self.exptime = get_header_value('EXPTIME', 0.0)
        self.saturateA = get_header_value('SATURATA', 0.0)
        self.saturateB = get_header_value('SATURATB', 0.0)
        self.gainA = get_header_value('GAINA', 0.0)
        self.gainB = get_header_value('GAINB', 0.0)
        self.rdnoiseA = get_header_value('RDNOISEA', 0.0)
        self.rdnoiseB = get_header_value('RDNOISEB', 0.0)

        # Match argtypes to data provided to self.axes
        if pass_fortran:
            npflags = 'aligned, f_contiguous, writeable'
        else:
            npflags = 'aligned, c_contiguous, writeable'
        set_desimage.restype = ctypes.c_int

        try:
            has_data = im.data is not None
        except AttributeError:
            has_data = False


        try:
            has_mask = im.mask is not None
        except AttributeError:
            has_mask = False

        try:
            has_weight = im.weight is not None
        except AttributeError:
            has_weight = False

        try:
            has_variance = im.variance is not None
        except AttributeError:
            has_variance = False

        # Test and correct data types if necessary

        if has_data:
            im.data = localize_numpy_array(im.data, data_dtype)

        if has_mask:
            im.mask = localize_numpy_array(im.mask, mask_dtype)

        if has_weight:
            im.weight = localize_numpy_array(im.weight, weight_dtype)

        if has_variance:
            im.variance = localize_numpy_array(im.variance, variance_dtype)

        if pass_fortran:
            if has_data:
                im.data = np.asfortranarray(im.data)
            if has_mask:
                im.mask = np.asfortranarray(im.mask)
            if has_weight:
                im.weight = np.asfortranarray(im.weight)
            if has_variance:
                im.variance = np.asfortranarray(im.variance)

        # Set the call signature according to what we have, and call

        set_desimage.argtypes = [
            ctypes.POINTER(ctypes.c_float) if not has_data else
            np.ctypeslib.ndpointer(ctypes.c_float, ndim=2, shape=im_shape,
                                   flags = npflags),
            ctypes.POINTER(ctypes.c_float) if not has_variance else
            np.ctypeslib.ndpointer(ctypes.c_float, ndim=2, shape=im_shape,
                                   flags = npflags),
            ctypes.POINTER(ctypes.c_float) if not has_weight else
            np.ctypeslib.ndpointer(ctypes.c_float, ndim=2, shape=im_shape,
                                   flags = npflags),
            ctypes.POINTER(mask_ctype) if not has_mask else
            np.ctypeslib.ndpointer(mask_ctype, ndim=2, shape=im_shape,
                                   flags = npflags),
            ctypes.POINTER(DESImageCStruct)
        ]

        set_desimage(im.data if has_data else None,
                     im.variance if has_variance else None,
                     im.weight if has_weight else None,
                     im.mask if has_mask else None,
                     ctypes.byref(self))

def localize_numpy_array(data, new_dtype=None):
    if new_dtype is None:
        native_dtype = data.dtype.newbyteorder('N')
    else:
        native_dtype = new_dtype.newbyteorder('N')

    try:
        local_data = data.astype(native_dtype, casting='equiv', copy=False)
    except TypeError:
        # Older version of numpy
        local_data = data.astype(native_dtype)
    return local_data


def update_hdr_compression(hdr,extname):

    # Translator for DES_EXT -- FM says: This sets of definitions should be centralized.
    DES_EXT = {
        'SCI' : 'IMAGE',
        'WGT' : 'WEIGHT',
        'MSK' : 'MASK',
        }

    # Create FITSHDR records with comments to insert into the header
    records = [
        {'name': 'EXTNAME', 'value':extname,         'comment':'Name of the extension'},
        {'name': 'DES_EXT', 'value':DES_EXT[extname],'comment':'DES name of the extension'},
        {'name': 'FZALGOR', 'value':chdu.get_FZALGOR(extname), 'comment':'Compression type'},
        {'name': 'FZDTHRSD','value':chdu.get_FZDTHRSD(extname),'comment':'Dithering seed value'},
        {'name': 'FZQVALUE','value':chdu.get_FZQVALUE(extname),'comment':'Compression quantization factor'},
        ]
    # We only update FZQMETHD if not NONE
    if chdu.get_FZQMETHD(extname) != "NONE":
        rec = {'name': 'FZQMETHD','value':chdu.get_FZQMETHD(extname),'comment':'Compression quantization method'}
        records.append(rec)

    # Now we add them to the header
    [hdr.add_record(rec) for rec in records]
    return hdr


def insert_eupspipe(hdr):
    import os

    try:
        EUPSPROD = os.environ['DESDM_PIPEPROD']
        EUPSVER  = os.environ['DESDM_PIPEVER']
    except:
        logger.info("WARNING: Could not find DESDM_PIPEPROD and DESDM_PIPEPVER in the environment")
        return hdr

    records = [
        {'name': 'EUPSPROD','value':EUPSPROD,'comment':'eups pipeline meta-package name'},
        {'name': 'EUPSVER', 'value':EUPSVER, 'comment':'eups pipeline meta-package version'},
        ]
    # Now we add them to the header
    [hdr.add_record(rec) for rec in records]
    return hdr

