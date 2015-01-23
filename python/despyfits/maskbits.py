#!/usr/bin/env python
"""Provied access to bit map canstants"""

import ctypes

libmaskbits = ctypes.CDLL('libmaskbits.so')

const_names = ("BADPIX_BPM",
               "BADPIX_SATURATE",
               "BADPIX_INTERP",
               "BADPIX_THRESHOLD",
               "BADPIX_LOW",
               "BADPIX_CRAY",
               "BADPIX_STAR",
               "BADPIX_TRAIL",
               "BADPIX_EDGEBLEED",
               "BADPIX_SSXTALK",
               "BADPIX_EDGE",
               "BADPIX_STREAK",
               "BADPIX_FIX",
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
               "BPMDEF_GENERIC")

for name in const_names:
    value = ctypes.c_int.in_dll(libmaskbits, name.lower()).value
    exec(name + " = %d" % value)
