#!/usr/bin/env python

# $Id: printHeader.py 23378 2014-06-24 14:39:15Z mgower $
# $Rev:: 23378                            $:  # Revision of last commit.
# $LastChangedBy:: mgower                 $:  # Author of last commit.
# $LastChangedDate:: 2014-06-24 09:39:15 #$:  # Date of last commit.

"""
Print header values to either stdout or to a file
"""

import argparse
import os
import sys
from pyfits import getheader

def print_header(fitsfile,ofileh=sys.stdout):
    """ print header from fits file to either stdout or to a file """
    hdr = getheader(fitsfile)
    ofileh.write("%s" % hdr.ascard)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prints fits headers')
    parser.add_argument('-O', '--outfile', action='store', type=str, help="Print header to given file", default=sys.stdout)
    #parser.add_argument('-c', action='store', type=str, help="Text file containing which header values to print")
    parser.add_argument('fitsfile', action='store')
    args = parser.parse_args()
    
    outfh=sys.stdout
    if args.outfile is not None:
        outfh = open(args.outfile, "w")

    print_header(args.fitsfile, outfh)

    if args.outfile is not None:
        outfh.close()
