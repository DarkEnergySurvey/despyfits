import distutils
from distutils.core import setup
import glob

bin_files = glob.glob("bin/*.py") + glob.glob("bin/*.txt")

# The main call
setup(name='despyfits',
      version ='0.1.2',
      license = "GPL",
      description = "A set of handy Python fitsfile-related utility functions for DESDM",
      author = "Felipe Menanteau",
      author_email = "felipe@illinois.edu",
      packages = ['despyfits'],
      package_dir = {'': 'python'},
      scripts = bin_files,
      )

