#!/usr/bin/env python
"""Class to manage DES images

In this context, "image" means the data from one CCD from one exposure,
possibly accompanied by mask and weight maps.
"""

# imports

import fitsio
import logging
from fitsio import FITSHDR
import numpy as np
import ctypes
import re

# constants

image_hdu = None
mask_hdu = 1
weight_hdu = 2
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

# interface functions
# classes

class HeaderDifference(object):
    def __init__(self, im1, im2):
        self.match = {}
        self.diff = {}
        for k in set(im1.header.keys()) | set(im2.header.keys()):
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

class DESDataImage(object):

    data_dtype = np.dtype(np.float32)

    def __init__(self, data, header={}, pri_hdr=None):
        """Create a new DESDataImage

        :Parameters:
            - `data`: the numpy data array
            - `header`: the header
            - `pri_hdr`: the primary header of the FITS file

        """
        self.data = localize_numpy_array(data, self.data_dtype)

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
    def load(cls, filename, ext=None):
        """Load from a FITS file

        :Parameters:
            - `filename`: the name of the FITS file from which to load
            - `ext`: the HDU index with the data image

        """
        if image_hdu is None:
            ext = 1 if filename.endswith('.fz') else 0
        else:
            ext = image_hdu

        data, header = fitsio.read(filename, ext=ext, header=True)

        im = cls(data, header)
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
            

    # If we index this object, assume we are after keywords in the header
    def __getitem__(self, key):
        if key in self.header.keys():
            return self.header[key]
        else:
            return self.pri_hdr[key]

    def __setitem__(self, key, value):
        self.header[key] = value





class DESImage(DESDataImage):

    mask_dtype = np.dtype(np.int16)
    weight_dtype = np.dtype(np.float32)

    def __init__(self):
        """Create a new DESImage

        Create an empty DESImage

        In mast cases, DESImage.create or DESImage.load will be safer 
        and more covenient.
        """
        self.data = None
        self.mask = None
        self.weight = None
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
        im.mask = localize_numpy_array(mask, cls.mask_dtype) \
                  if mask is not None \
                     else np.zeros_like(im.data, dtype=cls.mask_dtype)
        im.weight = localize_numpy_array(weight, cls.weight_dtype) \
                    if weight is not None \
                    else np.ones_like(im.data, dtype=cls.weight_dtype)
        return im

    @classmethod
    def load(cls, filename):
        """Load from a FITS file

        :Parameters:
            - `filename`: the name of the FITS file from which to load

        @returns: a new DESImage object with data from the FITS file
        """
        if image_hdu is None:
            data_hdu = 1 if filename.endswith('.fz') else 0
        else:
            data_hdu = image_hdu

        data_im = DESDataImage.load(filename)
        im = cls.create(data_im)

        found_mask, found_weight = False, False
        for ext in (mask_hdu, weight_hdu, 0, 1, 2):
            try:
                data, hdr = fitsio.read(filename, ext=ext+data_hdu, header=True)
            except ValueError:
                # we probably don't have all HDUs
                break

            def is_hdr_type(name1, name2):
                kv_pairs = (('DES_EXT', name1),
                            ('EXTNAME', name2))
                for key, value in kv_pairs:
                    if key in hdr.keys():
                        if hdr[key].rstrip()==value:
                            return True
                return False

            if is_hdr_type('MASK', 'MSK') and not found_mask:
                im.mask = localize_numpy_array(data, im.mask_dtype)
                im.mask_hdr = hdr
                found_mask = True
            elif is_hdr_type('WEIGHT', 'WGT') and not found_weight:
                im.weight = localize_numpy_array(data, im.weight_dtype)
                im.weight_hdr = hdr
                found_weight = True

        return im

    def save(self, filename, save_data=True, save_mask=True, save_weight=True):
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
                if hdu==mask_hdu:
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
                elif hdu==weight_hdu:
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

                self.mask_hdr['DES_EXT']='MASK'
                fits[mask_hdu].write(self.mask)
                fits[mask_hdu].write_keys(self.mask_hdr)

            if save_weight:
                hdr = fits[weight_hdu].read_header()
                if hdr['DES_EXT'].rstrip() != 'WEIGHT':
                    raise UnexpectedDesExt('weight not in expected HDU')

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
        print HeaderDifference(self, im)
        hdus = (('science', self.data, im.data),
                ('mask', self.mask, im.mask),
                ('weight', self.weight, im.weight))
        print

        diffs = DESImage()
        for ext, im1, im2 in hdus:
            print ext
            if not im1.shape==im2.shape:
                print "shape differs, im1: ", im1.shape, " im2: ", im2.shape
            if not im1.dtype==im2.dtype:
                print "dtype differs, im1: ", im1.dtype, " im2: ", im2.dtype

            delta = im1-im2
            if ext=='science':
                diffs.data = delta
            elif ext=='mask':
                diffs.mask = delta
            elif ext=='weight':
                diffs.weight = delta

            if (im1==im2).all():
                print "perfect match of values"
            else:
                print str(np.min(delta)) + " <= im1-im2 <= " + str(np.max(delta))

        return diffs

# internal functions & classes

def scan_fits_section(hdr, keyword):
    str_value = hdr[keyword]
    pattern = r"\[(\d+):(\d+),(\d+):(\d+)\]"
    m = re.match(pattern, str_value)
    if len(m.groups()) != 4:
        raise BadFITSSectionSpec("%s %s" % (keyword, str_value))

    values = [int(s) for s in m.groups()]
    return values


libdesimage = ctypes.CDLL('libdesimage.so')
set_desimage = libdesimage.set_desimage

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

    def __init__(self, im):
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
        set_desimage.restype = ctypes.c_int
        set_desimage.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float, ndim=2, shape=im_shape,
                                   flags = 'aligned, contiguous, writeable'),
            np.ctypeslib.ndpointer(ctypes.c_float, ndim=2, shape=im_shape,
                                   flags = 'aligned, contiguous, writeable'),
            np.ctypeslib.ndpointer(ctypes.c_short, ndim=2, shape=im_shape,
                                   flags = 'aligned, contiguous, writeable'),
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
