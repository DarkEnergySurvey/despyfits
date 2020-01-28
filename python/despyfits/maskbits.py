"""
Provide access to the mask bits defined in mask_bits.h from
imsupport. Access comes through a shared library that is compiled by
despyfits and loaded with ctypes.

To access the mask bits:

>>> from despyfits import maskbits
>>> maskbits.BADPIX_BPM

$Id: maskbits.py 42287 2016-06-06 22:22:15Z kadrlica $
"""
import os
import sys
import ctypes
import platform
import subprocess
import collections

__all__ = ['parse_badpix_mask']

lib_ext = {'Linux': 'so',
           'Darwin': 'dylib'}

def get_library_path():
    """
    Get the absolute path to the shared library.
    """
    try:
        libdir = os.path.join(os.getenv('DESPYFITS_DIR'), 'lib')
        libname = 'libmaskbits.' + lib_ext[platform.system()]
        libpath = os.path.join(libdir, libname)
    except KeyError:
        msg = "Unknown platform: " + platform.system()
        raise RuntimeError(msg)
    return libpath

def load_library(path=None):
    """
    Load the shared library.
    """
    if path is None:
        path = get_library_path()
    return ctypes.CDLL(path)

def get_bit_names(path=None):
    """
    Get the names of the mask bits from a shared library. Assumes that
    the only read-only symbols in the shared library are mask bits.
    """
    if path is None:
        path = get_library_path()

    if platform.system() == 'Linux':
        cmd = "nm %s --demangle | awk '$2~/^R$/ {print $NF}'" % path
    elif platform.system() == 'Darwin':
        cmd = "nm %s -U | awk '$2~/^S$/ {print $NF}'" % path
    else:
        msg = "Unrecognized platform: %s" % platform.system()
        raise RuntimeError(msg)
    try:
        out = subprocess.check_output(cmd, shell=True, text=True)
    except subprocess.CalledProcessError:
        msg = "Error accessing shared library: " + path
        raise RuntimeError(msg)

    bit_names = [x.lstrip('_') for x in map(str.upper, out.strip().split())]
    return tuple(bit_names)

def get_bit_dict(prefix=''):
    """
    Create a dictionary of mask bits with names starting with `prefix`.
    """
    libpath = get_library_path()
    libmaskbits = load_library(libpath)
    bit_names = get_bit_names(libpath)

    bits = []
    for _name in bit_names:
        if not _name.startswith(prefix):
            continue
        bits.append((_name, ctypes.c_int.in_dll(libmaskbits, _name.lower()).value))
    return dict(bits)

def get_bit_odict(prefix=''):
    """
    Create a value-ordered dictionary of bits.
    """
    d = get_bit_dict(prefix)
    return collections.OrderedDict(sorted(d.items(), key=lambda x: x[1]))

def parse_badpix_mask(inp):
    """
    Utility routine that takes an integer or string of the form
    "EDGE,BPM,SATURATE" and converts it into an OR of the appropriate
    BADPIX bits.  An exception will be raised if there are unknown
    values.
    """
    try:
        out = int(inp)
    except ValueError:
        out = type(BADPIX_BPM)(0)
        for bit in inp.split(','):
            #out |= eval('BADPIX_'+bit.strip())
            out |= getattr(sys.modules[__name__], 'BADPIX_' + bit.strip())
    return out

# Create bit mask dictionaries (not exported in __all__)
MASKBITS = get_bit_dict()
BADPIX = get_bit_odict('BADPIX')
BPMDEF = get_bit_odict('BPMDEF')

# Set global module variables and add to __all__
for name, value in MASKBITS.items():
    setattr(sys.modules[__name__], name, value)
    __all__.append(name)
