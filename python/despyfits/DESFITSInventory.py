#!/usr/bin/env python
"""Utilities to find DES HDUs in FITS files
"""
from fitsio import FITS

class DESFITSInventory(object):
    """An invertory of the contents of a DES FITS file
    """

    def __init__(self, fname):
        self.hdr = []
        with FITS(fname) as fits:
            for i, hdu in enumerate(fits):
                self.hdr.append( hdu.read_header() )

    def hdr_keyword_matches(self, keyword, value=None, strip=True):
        def hdr_matches(hdr):
            if not keyword in hdr:
                return False
            if value is None:
                return True
            read_value = hdr[keyword]

            if strip:
                try:
                    read_value = read_value.strip()
                except AttributeError:
                    # Its not a string, so strip makes no sense
                    pass

            value_matches = read_value==value
            return value_matches

        matching_hdus = [i for i,h in enumerate(self.hdr) if hdr_matches(h)]
        return matching_hdus

    def ccd_hdus(self, ccdnum):
        matching_hdus = self.hdr_keyword_matches('CCDNUM', ccdnum)
        return matching_hdus


    @property
    def bpms(self):
        matching_hdus = self.hdr_keyword_matches('DES_EXT', 'MASK')
        return matching_hdus

    def type_matches(self, extname_value, des_ext_value):
        extname_matches = self.hdr_keyword_matches('EXTNAME', extname_value)
        des_ext_matches = self.hdr_keyword_matches('DES_EXT', des_ext_value)
        ## matching_hdus = sorted(set(extname_matches) & set(des_ext_matches))
        matching_hdus = sorted(set(extname_matches) | set(des_ext_matches))
        return matching_hdus

    @property
    def weights(self):
        return self.type_matches('WGT', 'WEIGHT')

    @property
    def masks(self):
        return self.type_matches('MSK', 'MASK')

    @property
    def scis(self):
        return self.type_matches('SCI', 'IMAGE')

    @property
    def raws(self):
        if not 'PROCTYPE' in self.hdr[0]:
            return []
        if not self.hdr[0]['PROCTYPE'].strip() == 'RAW':
            return []
        return self.hdr_keyword_matches('CCDNUM')
        
        
