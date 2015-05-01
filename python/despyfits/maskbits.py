#!/usr/bin/env python
"""Provied access to bit map canstants"""

import ctypes
import platform

lib_ext = {'Linux': 'so',
           'Darwin': 'dylib'}
try:
    libmaskbits = ctypes.CDLL(
        'libmaskbits.' + lib_ext[platform.system()])
except KeyError:
    raise RuntimeError, ("Unknown platform: " + platform.system())


const_names = ("BADPIX_BPM",
               "BADPIX_SATURATE",
               "BADPIX_INTERP",
               "BADPIX_THRESHOLD",
               "BADPIX_BADAMP",
               "BADPIX_CRAY",
               "BADPIX_STAR",
               "BADPIX_TRAIL",
               "BADPIX_EDGEBLEED",
               "BADPIX_SSXTALK",
               "BADPIX_EDGE",
               "BADPIX_STREAK",
               "BADPIX_SUSPECT",
               "BPMDEF_FLAT_MIN",
               "BPMDEF_FLAT_MAX",
               "BPMDEF_FLAT_MASK",
               "BPMDEF_BIAS_HOT",
               "BPMDEF_BIAS_WARM",
               "BPMDEF_BIAS_MASK",
               "BPMDEF_BIAS_COL",
               "BPMDEF_EDGE",
               "BPMDEF_CORR",
               "BPMDEF_SUSPECT",
               "BPMDEF_FUNKY_COL",
               "BPMDEF_WACKY_PIX",
               "BPMDEF_BADAMP")

for name in const_names:
    value = ctypes.c_int.in_dll(libmaskbits, name.lower()).value
    exec(name + " = %d" % value)

def parse_badpix_mask(input):
    """
    Utility routine that can take a string input that is either an integer, or a string
    of the form "EDGE,BPM,SATURATE" that this routine will convert into the sum of
    the appropriate BADPIX bits.  An exception will be raised if there are unknown values.
    """
    try:
        out = int(input)
    except ValueError:
        out = type(BADPIX_BPM)(0)
        for bit in input.split(','):
            out |= eval('BADPIX_'+bit.strip())
    return out
