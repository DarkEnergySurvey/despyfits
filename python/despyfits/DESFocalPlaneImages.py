#!/usr/bin/env python
"""Class to manage all images in a focal plane
"""
from despyfits.DESImage import DESDataImage, DESImage, DESImageCStruct, CCDNUM2
from fitsio import FITS

FocalPlaneCStructArray = DESImageCStruct * CCDNUM2

class DESFocalPlaneImages(object):

    def __init__(self, fname):
        fits = FITS(fname)
        hdus_present = len(fits)
        fits.close()

        self.images = [DESDataImage.load(fname, ext)
                       for ext in range(hdus_present)]
            

    def save(self, fname_template):
        for hdu, im in enumerate(self.images):
            fname = fname_template % hdu
            self.images.save(fname)

    @property
    def cstruct(self):
        num_dummy_structs = NUMCCD2-len(self.images)
        im_cstructs = [im.cstruct for im in self.images] \
                      + [DESImageCStruct() for i in range(num_dummy_structs)]
        fc_im_array = FocalPlaneCStructArray(*im_cstructs)
        return fc_im_array
