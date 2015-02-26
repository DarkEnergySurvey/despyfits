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
               "BPMDEF_TAPE_BUMP",
               "BPMDEF_FUNKY_COL",
               "BPMDEF_WACKY_PIX",
               "BPMDEF_BADAMP")

for name in const_names:
    value = ctypes.c_int.in_dll(libmaskbits, name.lower()).value
    exec(name + " = %d" % value)
