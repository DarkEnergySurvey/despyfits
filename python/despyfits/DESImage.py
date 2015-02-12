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

import numpy as np

import fitsio
from fitsio import FITSHDR
from despyfits import maskbits
from despyfits.DESFITSInventory import DESFITSInventory

# constants

image_hdu = None
mask_hdu = 1
weight_hdu = 2
weight_dtype = np.dtype(np.float32)
data_dtype = np.dtype(np.float32)
pass_fortran = False

mask_is_unsigned = True
if mask_is_unsigned:
    mask_dtype = np.dtype(np.uint16)
    mask_ctype = ctypes.c_ushort
else:
    mask_dtype = np.dtype(np.int16)
    mask_ctype = ctypes.c_short
    

logger = logging.getLogger('DESImage')
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


class DESDataImage(DESSingleImage):

    def __init__(self, data, header={}, pri_hdr=None):
        """Create a new DESDataImage

        :Parameters:
            - `data`: the numpy data array
            - `header`: the header
            - `pri_hdr`: the primary header of the FITS file

        """
        self.data = data

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

        im = cls(data, header)
        return im

    @classmethod
    def load_from_open(cls, fits, image_hdu):
        """Load from a FITS file

        :Parameters:
            - `filename`: the name of the FITS file from which to load
            - `image_hdu`: the HDU index with the data image

        """
        header = fits[image_hdu].read_header()
        data = fits[image_hdu].read()
        pri_hdr = fits[0].read_header()
        im = cls(data, header, pri_hdr)
        return im

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
                 shape=(4096, 2048)):
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
            
        self.header = FITSHDR({})
        self.pri_hdr = self.header
        self.mask_hdr = FITSHDR({})
        self.weight_hdr = FITSHDR({})

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
        
    @classmethod
    def load(cls, filename, 
             assign_default_mask=True, 
             assign_default_weight=True,
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
            has_weight = self.weight is not None
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
                pass
            file_exists = True
        except ValueError:
            file_exists = False

        with fitsio.FITS(filename, fitsio.READWRITE) as fits:

            # Make sure the file exists and has the expected HDUs
            # Create any HDUs that are needed but don't already exist
            max_hdu = max([data_hdu, mask_hdu, weight_hdu])
            max_init_hdu = len(fits) if file_exists else 0

            for hdu in range(max_init_hdu, max_hdu+1):
                if has_mask and hdu==mask_hdu:
                    fits.create_image_hdu(self.mask)
                    if save_mask:
                        # If we can, write the HDU as we create it
                        # because insertion into the middle of a
                        # FITS file is slow.
                        fits[mask_hdu].write_keys(self.mask_hdr)
                        fits[mask_hdu].write_key('DES_EXT','MASK')
                        fits[mask_hdu].write_key('EXTNAME','MSK')
                        fits[mask_hdu].write(self.mask)
                        save_mask = False
                elif has_weight and hdu==weight_hdu:
                    fits.create_image_hdu(self.weight)
                    if save_weight:
                        fits[weight_hdu].write_keys(self.weight_hdr)
                        fits[weight_hdu].write_key('DES_EXT','WEIGHT')
                        fits[weight_hdu].write_key('EXTNAME','WGT')
                        fits[weight_hdu].write(self.weight)
                        save_weight = False
                elif hdu==data_hdu:
                    fits.create_image_hdu(self.data)
                    if save_data:
                        fits[data_hdu].write_keys(self.header)
                        fits[data_hdu].write_key('DES_EXT','IMAGE')
                        fits[data_hdu].write_key('EXTNAME','SCI')
                        fits[data_hdu].write(self.data)
                        save_data = False
                else:
                    fits.create_image_hdu(self.data)

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

    def __init__(self, bpm, header={}, pri_hdr=None):
        """Create a new DESBPMImage

        :Parameters:
            - `bpm`: the numpy data array
            - `header`: the header
            - `pri_hdr`: the primary header of the FITS file

        """
        self.mask = bpm

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

        bpm = cls(data, header)
        return bpm

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

lib_ext = {'Linux': 'so',
           'Darwin': 'dylib'}
try:
    libdesimage = ctypes.CDLL(
        'libdesimage.' + lib_ext[platform.system()])
except KeyError:
    raise RuntimeError, ("Unknown platform: " + platform.system())


set_desimage = libdesimage.set_desimage
set_data_desimage = libdesimage.set_data_desimage
set_bpm_desimage = libdesimage.set_bpm_desimage
set_weightless_desimage = libdesimage.set_weightless_desimage
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
        ('varim', ctypes.POINTER(ctypes.c_float)),
        ('mask', ctypes.POINTER(ctypes.c_short))
    ]

    def __init__(self, im=None):
        if im is not None:
            self.create(im)
        else:
            self.image = None
            self.varim = None
            self.mask = None
            self.npixels = 0

    def create(self, im, data_only=False, weightless=False):
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

        # Test and correct data types if necessary

        if has_data:
            im.data = localize_numpy_array(im.data, data_dtype)

        if has_mask:
            im.mask = localize_numpy_array(im.mask, mask_dtype)

        if has_weight:
            im.weight = localize_numpy_array(im.weight, weight_dtype)

        if pass_fortran:
            if has_data:
                im.data = np.asfortranarray(im.data)
            if has_mask:
                im.mask = np.asfortranarray(im.mask)
            if has_weight:
                im.weight = np.asfortranarray(im.weight)


        # Set the call signature according to what we have, and call
        if isinstance(im, DESBPMImage):
            set_bpm_desimage.argtypes = [
                np.ctypeslib.ndpointer(mask_ctype, ndim=2, shape=im_shape,
                                       flags = npflags),
                ctypes.POINTER(DESImageCStruct)
            ]
            set_bpm_desimage(im.mask, ctypes.byref(self))
        elif not (has_mask or has_weight):
            set_data_desimage.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, ndim=2, shape=im_shape,
                                       flags = npflags),
                ctypes.POINTER(DESImageCStruct)
            ]
            set_data_desimage(im.data, ctypes.byref(self))
        elif has_mask and not has_weight:
            set_weightless_desimage.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, ndim=2, shape=im_shape,
                                       flags = npflags),
                np.ctypeslib.ndpointer(mask_ctype, ndim=2, shape=im_shape,
                                       flags = npflags),
                ctypes.POINTER(DESImageCStruct)
            ]
            set_weightless_desimage(im.data, im.mask, ctypes.byref(self))
        else:
            set_desimage.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, ndim=2, shape=im_shape,
                                       flags = npflags),
                np.ctypeslib.ndpointer(ctypes.c_float, ndim=2, shape=im_shape,
                                       flags = npflags),
                np.ctypeslib.ndpointer(mask_ctype, ndim=2, shape=im_shape,
                                       flags = npflags),
                ctypes.POINTER(DESImageCStruct)
            ]
            set_desimage(im.data, im.weight, im.mask, ctypes.byref(self))
            

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
