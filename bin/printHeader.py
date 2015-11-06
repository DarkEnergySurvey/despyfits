#!/usr/bin/env python

# $Id: printHeader.py 23378 2014-06-24 14:39:15Z mgower $
# $Rev:: 23378                            $:  # Revision of last commit.
# $LastChangedBy:: mgower                 $:  # Author of last commit.
# $LastChangedDate:: 2014-06-24 09:39:15 #$:  # Date of last commit.

"""
Print header values to either stdout or to a file
"""

import argparse
import os,sys
import fitsio

def print_header(fitsfile,ext=0,ofileh=sys.stdout):
    """ print header from fits file to either stdout or to a file """

    hdr = fitsio.read_header(fitsfile, ext=ext)
    ofileh.write("%s" % hdr)
    ofileh.write("\n")
    outfh.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prints fits headers')
    parser.add_argument('-o', '--outfile', action='store', type=str,help="Print header to given file", default=False)
    parser.add_argument('-x','--extension', action='store', default=0)
    parser.add_argument('fitsfile', action='store')
    args = parser.parse_args()
    
    if args.outfile:
        try:
            outfh = open(args.outfile, "w")
        except:
            sys.exit("ERROR: Cannot open %s" % args.outfile)
    else:
        outfh=sys.stdout

    # Convert extension to integers in not strings
    try:
        args.extension = int(args.extension)
    except:
        pass

    # Make the call
    print_header(args.fitsfile, ext=args.extension, ofileh=outfh)
