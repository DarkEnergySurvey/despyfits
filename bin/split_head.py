#!/work/apps/RHEL6/dist/eups/packages/Linux64/python/2.7.6+2/bin/python
# $Id$
# $Rev::                                  $:  # Revision of last commit.
# $LastChangedBy::                        $:  # Author of last commit.
# $LastChangedDate::                      $:  # Date of last commit.

""" Split single head file into multiple files """

import argparse
import despyfits.fitsutils as fitsutils

def read_list(listname):
    """ Read output catalog names from list file """
    outfiles = []
    with open(listname, 'r') as listfh:
        outfiles = listfh.readlines()
    return outfiles


def main():
    """ Entry point """
    parser = argparse.ArgumentParser(description='Split single head file into multiple files')
    parser.add_argument('--in', action='store', help='head file to split')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--list', action='store', 
                       help='list file containing output filenames, order must match order in head file')
    group.add_argument('--out', action='store', 
                       help='output filenames, order must match order in head file')

    args = vars(parser.parse_args())   # convert dict

    inhead = args['in']

    outheads = args['out']
    if args['list'] is not None:
        outheads = ','.join(read_list(args['list']))

    print "Splitting %s into %s" % (inhead, outheads)
    fitsutils.splitScampHead(inhead, outheads)


if __name__ == '__main__':
    main()
