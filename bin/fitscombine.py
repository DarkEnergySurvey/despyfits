#!/usr/bin/env python 

import despyfits
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a MEF fits file from a list of flat fits files")
    
    parser.add_argument("filenames", nargs='*',
                        help="List of inputs FITS files to use.")
    parser.add_argument("--outname",  
                        help="Name of output FITS file.")
    parser.add_argument("--extnames", nargs='*',
                        help="List of EXTNAME to use for each file.")
    parser.add_argument("--clobber", action='store_true', default=False,
                        help="Clobber output MEF fits file")
    args = parser.parse_args()
    kwargs = vars(args)
    despyfits.makeMEF(**kwargs)
