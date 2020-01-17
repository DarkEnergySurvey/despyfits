import os
import distutils
from distutils.core import setup, Extension
import glob

bin_files = glob.glob("bin/*") + glob.glob("bin/*.txt")
inc_files = glob.glob("include/*.h")
doc_files = glob.glob("doc/*.*")
etc_files = glob.glob("etc/*.*")

libdesimage = Extension(
    'desimage',
    sources = ['src/libdesimage.c'],
    include_dirs = ['include'],
    extra_compile_args = ['-O3', '-g', '-Wall', '-shared', '-fPIC'])

libmaskbits = Extension(
    'maskbits',
    sources = ['src/libmaskbits.c'],
    include_dirs = ['include', f"{os.environ['IMSUPPORT_DIR']}/include"],
    extra_compile_args = ['-O3','-g','-Wall','-shared','-fPIC'])

libcompressionhdu = Extension(
    'compressionhdu',
    sources = ['src/libcompressionhdu.c'],
    include_dirs = ['include', f"{os.environ['IMSUPPORT_DIR']}/include"],
    extra_compile_args = ['-O3', '-g', '-Wall', '-shared', '-fPIC'])

# The main call
setup(name='despyfits',
      version ='devel',
      license = "GPL",
      description = "A set of handy Python fitsfile-related utility functions for DESDM",
      author = "Felipe Menanteau, Eric Neilsen",
      author_email = "felipe@illinois.edu",
      ext_modules = [libdesimage, libmaskbits, libcompressionhdu],
      packages = ['despyfits'],
      package_dir = {'': 'python'},
      scripts = bin_files,
      data_files=[('ups', ['ups/despyfits.table']),
                  ('doc', doc_files),
                  ('etc', etc_files),
                  ('include', inc_files)],
      )

