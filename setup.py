import os 
import distutils
from distutils.core import setup
import glob

import shlib 
from shlib.build_shlib import SharedLibrary

bin_files = glob.glob("bin/*") + glob.glob("bin/*.txt")
inc_files = glob.glob("include/*.h") 
doc_files = glob.glob("doc/*.*")
etc_files = glob.glob("etc/*.*")

libdesimage = SharedLibrary(
    'desimage',
    sources = ['src/libdesimage.c'],
    include_dirs = ['include'],
    extra_compile_args = ['-O3','-g','-Wall','-shared','-fPIC'])

libmaskbits = SharedLibrary(
    'maskbits', 
    sources = ['src/libmaskbits.c'],
    include_dirs = ['include', '%s/include' % os.environ['IMSUPPORT_DIR']],
    extra_compile_args = ['-O3','-g','-Wall','-shared','-fPIC'])

libcompressionhdu = SharedLibrary(
    'compressionhdu', 
    sources = ['src/libcompressionhdu.c'],
    include_dirs = ['include', '%s/include' % os.environ['IMSUPPORT_DIR']],
    extra_compile_args = ['-O3','-g','-Wall','-shared','-fPIC'])

# The main call
setup(name='despyfits',
      version ='0.3.0',
      license = "GPL",
      description = "A set of handy Python fitsfile-related utility functions for DESDM",
      author = "Felipe Menanteau, Eric Neilsen",
      author_email = "felipe@illinois.edu",
      shlibs = [libdesimage, libmaskbits,libcompressionhdu],
      packages = ['despyfits'],
      package_dir = {'': 'python'},
      scripts = bin_files,
      data_files=[('ups',['ups/despyfits.table']),
                  ('doc', doc_files),
                  ('etc', etc_files),
                  ('include', inc_files)],
      )

