#!/usr/bin/env python
"""Class to manage all images in a focal plane
"""
from despyfits.DESImage import DESDataImage, DESImage, DESImageCStruct
from despyfits.DESImage import CCDNUM2, data_dtype
from despyfits.DESFITSInventory import DESFITSInventory
from fitsio import FITS
import numpy as np

FocalPlaneCStructArray = DESImageCStruct * CCDNUM2

class DESFocalPlaneImages(object):

    def __init__(self, 
                 init_data=False, 
                 shape=(4096, 2048)):
        if init_data:
            self.images = [DESDataImage(np.zeros(shape, dtype=data_dtype))
                           for i in range(CCDNUM2)]

    @classmethod
    def load(cls, fname):
        fits_inventory = DESFITSInventory(fname)
        hdus_present = sorted(fits_inventory.raws)

        images = cls()
        with FITS(fname) as fits:
            images.images = [DESDataImage.load_from_open(fits, ext)
                             for ext in hdus_present]
        return images


    def save(self, fname_template):
        for hdu, im in enumerate(self.images):
            fname = fname_template % hdu
            im.save(fname)

    @property
    def cstruct(self):
        num_dummy_structs = NUMCCD2-len(self.images)
        im_cstructs = [im.cstruct for im in self.images] \
                      + [DESImageCStruct() for i in range(num_dummy_structs)]
        fc_im_array = FocalPlaneCStructArray(*im_cstructs)
        return fc_im_array
