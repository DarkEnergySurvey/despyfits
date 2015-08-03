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
import pyfits
from pyfits import getheader

def print_header(fitsfile,ofileh=sys.stdout):
    """ print header from fits file to either stdout or to a file """

    # Get the Pyfits version as a float
    pyfitsVersion = float(".".join(pyfits.__version__.split(".")[0:2]))

    hdr = getheader(fitsfile)
    if pyfitsVersion < 3.1:  # Older method
        ofileh.write("%s" % hdr.ascardlist())
    else: # use the up-to-date method (Header.cards)
        for items in hdr.cards:
            ofileh.write("%s\n" % items)

    ofileh.write("\n")
    outfh.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prints fits headers')
    parser.add_argument('-O', '--outfile', action='store', type=str,help="Print header to given file", default=False)
    #parser.add_argument('-c', action='store', type=str, help="Text file containing which header values to print")
    parser.add_argument('fitsfile', action='store')
    args = parser.parse_args()
    
    if args.outfile:
        try:
            outfh = open(args.outfile, "w")
        except:
            sys.exit("ERROR: Cannot open %s" % args.outfile)
    else:
        outfh=sys.stdout

    # Make the call
    print_header(args.fitsfile, outfh)
