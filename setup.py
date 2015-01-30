import distutils
from distutils.core import setup
import glob

import shlib 
from shlib.build_shlib import SharedLibrary

bin_files = glob.glob("bin/*.py") + glob.glob("bin/*.txt")

libdesimage = SharedLibrary(
    'desimage',
    sources = ['src/libdesimage.c'],
    include_dirs = ['include'],
    extra_compile_args = ['-O3','-g','-Wall','-shared','-fPIC'])

libmaskbits = SharedLibrary(
    'maskbits', 
    sources = ['src/libmaskbits.c'],
    include_dirs = ['include'],
    extra_compile_args = ['-O3','-g','-Wall','-shared','-fPIC'])
 
# The main call
setup(name='despyfits',
      version ='0.1.1',
      license = "GPL",
      description = "A set of handy Python fitsfile-related utility functions for DESDM",
      author = "Felipe Menanteau",
      author_email = "felipe@illinois.edu",
      shlibs = [libdesimage, libmaskbits],
      packages = ['despyfits'],
      package_dir = {'': 'python'},
      data_files = [('ups', ['ups/despyfits.table'])],
      scripts = bin_files,
      data_files=[('ups',['ups/despyfits.table'])],
      )

