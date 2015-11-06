#!/usr/bin/env python
# $Id$
# $Rev::                                  $:  # Revision of last commit.
# $LastChangedBy::                        $:  # Author of last commit.
# $LastChangedDate::                      $:  # Date of last commit.

""" Combine cats into single file """

import argparse
import despyfits.fitsutils as fitsutils

def read_list(listname):
    """ Read input catalog names from list file """
    incats = []
    with open(listname, 'r') as listfh:
        incats = listfh.readlines()
    return incats


def main():
    """ Entry point """
    parser = argparse.ArgumentParser(description='Combine cats into single file')
    parser.add_argument('--outcat', action='store', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store')
    group.add_argument('--incats', action='store')

    args = vars(parser.parse_args())   # convert dict

    incats = args['incats']
    if args['list'] is not None:
        incats = ','.join(read_list(args['list']))

    print "Combining catalogs into %s" % (args['outcat'])
    fitsutils.combine_cats(incats, args['outcat'])


if __name__ == '__main__':
    main()
