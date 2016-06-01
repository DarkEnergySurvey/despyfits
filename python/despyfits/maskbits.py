#!/usr/bin/env python
"""
Provide access to the mask bits defined in mask_bits.h from
imsupport. Access comes through a shared library that is compiled by
despyfits and loaded with ctypes.

To access the mask bits:

>>> from despyfits import maskbits
>>> maskbits.BADPIX_BPM

$Id$
"""
import os, sys
import ctypes
import platform
import subprocess
from collections import OrderedDict

lib_ext = {'Linux': 'so',
           'Darwin': 'dylib'}

def get_bit_names(path):
    """
    Get the names of the mask bits from a shared library. Assumes that
    the only read-only symbols in the shared library are mask bits.
    """
    cmd = "nm %s --demangle | awk '$2~/^R$/ {print $3}'"%path
    out = subprocess.check_output(cmd,shell=True)
    bit_names = tuple([x for x in map(str.upper,out.strip().split())])
    return bit_names

def get_bit_dict(prefix=''):
    """
    Create a dictionary of mask bits with names starting with `prefix`.
    """
    bits = []
    for name in bit_names:
        if not name.startswith(prefix): continue
        value = ctypes.c_int.in_dll(libmaskbits, name.lower()).value
        bits.append((name,value))
    return dict(bits)

def get_bit_odict(prefix=''):
    """
    Create a value-ordered dictionary of bits.
    """
    return OrderedDict(sorted(get_bit_dict(prefix).items(),key=lambda x: x[1]))

def parse_badpix_mask(input):
    """
    Utility routine that takes an integer or string of the form
    "EDGE,BPM,SATURATE" and converts it into an OR of the appropriate
    BADPIX bits.  An exception will be raised if there are unknown
    values.
    """
    try:
        out = int(input)
    except ValueError:
        out = type(BADPIX_BPM)(0)
        for bit in input.split(','):
            out |= eval('BADPIX_'+bit.strip())
    return out

# Load the shared library
try:
    libdir = os.path.join(os.getenv('DESPYFITS_DIR'),'lib')
    libname = 'libmaskbits.' + lib_ext[platform.system()]
    libpath = os.path.join(libdir,libname)
    libmaskbits = ctypes.CDLL(libpath)
except KeyError:
    msg = "Unknown platform: " + platform.system()
    raise RuntimeError(msg)

# Get the mask bits
try:
    bit_names = get_bit_names(libpath)
except subprocess.CalledProcessError:
    msg = "Error accessing shared library: " + libmaskbits
    raise RuntimeError(msg)

# Create bit mask dictionaries
MASKBITS = get_bit_dict()
BADPIX = get_bit_odict('BADPIX')
BPMDEF = get_bit_odict('BPMDEF')

# Set global module variables
for name,value in MASKBITS.items():
    setattr(sys.modules[__name__],name,value)
