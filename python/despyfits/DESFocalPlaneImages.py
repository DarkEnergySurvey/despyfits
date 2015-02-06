#!/usr/bin/env python
"""Class to manage all images in a focal plane
"""
from despyfits.DESImage import DESDataImage, DESImage, DESImageCStruct, CCDNUM2
from fitsio import FITS

FocalPlaneCStructArray = DESImageCStruct * CCDNUM2

class DESFocalPlaneImages(object):

    def __init__(self, 
                 init_data=False, init_mask=False, init_weight=False,
                 shape=(4096, 2048)):
        if init_data or init_mask or init_weight:
            self.images = [DESDataImage(init_data=init_data,
                                        init_maks=init_mask,
                                        init_weight=init_weight,
                                        shape=shape)
                           for i in range(CCDNUM2)]

    @classmethod
    def load(cls, fname):
        fits = FITS(fname)
        hdus_present = len(fits)
        fits.close()

        images = cls()

        images.images = [DESDataImage.load(fname, ext)
                         for ext in range(hdus_present)]
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
