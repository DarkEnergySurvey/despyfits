"""Provied access to bit map canstants"""

import ctypes
import platform
import os

lib_ext = {'Linux': 'so',
           'Darwin': 'dylib'}
try:
    libcompressionhdu = ctypes.CDLL(
        os.path.join(
            os.environ['DESPYFITS_DIR'], 
            'libcompressionhdu.' + lib_ext[platform.system()],
        )
    )
except KeyError:
    raise RuntimeError("Unknown platform: " + platform.system())


const_int_names = ("IMG_FZQVALUE",
                   "WGT_FZQVALUE",
                   "MSK_FZQVALUE")

const_char_names = ("IMG_FZALGOR",
                    "WGT_FZALGOR",
                    "MSK_FZALGOR",
                    "IMG_FZQMETHD",
                    "WGT_FZQMETHD",
                    "MSK_FZQMETHD",
                    "IMG_FZDTHRSD",
                    "WGT_FZDTHRSD",
                    "MSK_FZDTHRSD")

for name in const_int_names:
    value = ctypes.c_int.in_dll(libcompressionhdu, name.lower()).value
    exec(name + " = {:d}".format(value))

for name in const_char_names:
    value = ctypes.c_char_p.in_dll(libcompressionhdu, name.lower()).value
    if isinstance(value, bytes):
        value = value.decode()
    exec(name + " = '{}'".format(value))

def get_FZALGOR(hdu_name):
    """ Get FZALGOR for hdu_name [IMG/WGT/MSK]"""
    if hdu_name == 'SCI':
        hdu_name = 'IMG'
    FZALGOR = eval(hdu_name + '_FZALGOR')
    return FZALGOR

def get_FZQMETHD(hdu_name):
    if hdu_name == 'SCI':
        hdu_name = 'IMG'
    FZQMETHD = eval(hdu_name + '_FZQMETHD')
    return FZQMETHD

def get_FZDTHRSD(hdu_name):
    if hdu_name == 'SCI':
        hdu_name = 'IMG'
    FZDTHRSD = eval(hdu_name + '_FZDTHRSD')
    return FZDTHRSD

def get_FZQVALUE(hdu_name):
    if hdu_name == 'SCI':
        hdu_name = 'IMG'
    FZQVALUE = eval(hdu_name + '_FZQVALUE')
    return FZQVALUE
