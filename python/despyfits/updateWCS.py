#!/usr/bin/env python3

# $Id: UpdateScampHead.py 40667 2015-11-04 20:25:56Z felipe $
# $Rev:: 40667                            $:  # Revision of last commit.
# $LastChangedBy:: felipe                 $:  # Author of last commit.
# $LastChangedDate:: 2015-11-04 14:25:56 #$:  # Date of last commit.

# This is the set of libraries used to update the WCS of finalcut images

# Standard Libraries
import unicodedata
import os
import fitsio

# DESDM libraries
from despyfits.fwhmFromFITS_LDAC import fwhmFromFITS_LDAC
from despyfits.DESImage import DESImage
from despymisc.xmlslurp import Xmlslurper

def get_record(header, keyword):
    """ Utility option to gets the record form a FITSHDR header """
    index = header._index_map[keyword]
    return header._record_list[index]

def update_headers(image, new_header, keywords_spec, verbose=False):
    """
    Update the header in image using the FITSHRD object new_record for keywords present in keywords_spec
      image:          despyfits.DESIma object
      new_record:     fitsio.FITSHDR object
      keywords_spec:  a dictionary with keywords and HDU to use
    """

    # Loop over keywords and HDUs
    for keyword in keywords_spec:

        new_record = get_record(new_header, keyword)
        if verbose:
            try:
                old_value = image.header[keyword]
            except:
                old_value = "Undefined"
            print(f"(updateWCS) Updating keyword: {keyword:8s} from {old_value} --> {new_record['value']} for {keywords_spec[keyword]}")

        # Loop over HDU's
        for _ in keywords_spec[keyword]:
            if 'SCI' in keywords_spec[keyword] or 0 in keywords_spec[keyword]:
                image.header.add_record(new_record)
            if 'MSK' in keywords_spec[keyword] or 1 in keywords_spec[keyword]:
                image.mask_hdr.add_record(new_record)
            if 'WGT' in keywords_spec[keyword] or 2 in keywords_spec[keyword]:
                image.weight_hdr.add_record(new_record)

    return image

def pyfitsrec2fitsiorecord(old_record):
    """ Translates old (pyfits?) record style to a fitsio compliant one """

    new_record = []
    for name in old_record:
        rec = {'name': name,
               'value': old_record[name][0],
               'comment': old_record[name][1],
               }
        new_record.append(rec)
    return new_record

def get_keywords_to_update(hdupcfg, verbose=False):
    """
    Read in the header update configuration file.
    Now all keywords that can be found/derived should be present are
    read from a configuration file
    """
    if not os.path.isfile(hdupcfg):
        exit(f"Header update config file ({hdupcfg}) not found! ")
    if verbose:
        print("(updateWCS): Reading the KEYWORD that we need want to UPDATE.")

    kywds_spec = {}
    for line in open(hdupcfg).readlines():
        if line[0] == "#":
            continue
        keyword = line.split()[0]
        values = line.split()[1]
        # Try converting str HDU to integer
        try:
            kywds_spec[keyword] = [int(v) for v in values.split(',')]
        except:
            kywds_spec[keyword] = values.split(',')

    return kywds_spec

def get_fwhm_from_catalog(fwhm, verbose=False, debug=False):

    """ Gets the FWHM and ELLIPTICITY """

    tmpdict = {}
    if fwhm:
        if verbose:
            print("(updateWCS): Will determine median FWHM & ELLIPTICITY and number of candidates")
        fwhm_med, ellp_med, count = fwhmFromFITS_LDAC(fwhm, debug=debug)
    else:
        if verbose:
            print("FWHM option keyword will not be populated\n")
        return tmpdict

    if verbose and not debug:
        print(f" FWHM    ={fwhm_med:.4f}")
        print(f" ELLIPTIC={ellp_med:.4f}")
        print(f" NFWHMCNT={count}")
    tmpdict['FWHM'] = [round(fwhm_med, 4), 'Median FWHM from SCAMP input catalog [pixels]']
    tmpdict['ELLIPTIC'] = [round(ellp_med, 4), 'Median Ellipticity from SCAMP input catalog']
    tmpdict['NFWHMCNT'] = [count, 'Number of objects used to find FWHM']
    return tmpdict

def slurp_XML(xml, tmpdict, verbose=False, debug=False, translate=True):

    """ Read in the XML File """

    key_pairs = {}
    if xml:
        if verbose:
            print(f"Reading SCAMP XML file: {xml:s}")
        try:
            tmp_xml_tbl = Xmlslurper(xml, ['FGroups']).gettables()
            # Possible that should redefine the keys using the same unicodedata.normalize method as for the values below
        except:
            print("Error: failed to slurp the data.")

        if 'FGroups' in tmp_xml_tbl:
            if debug:
                for key in tmp_xml_tbl['FGroups'][0]:
                    print(f"  {key} = {tmp_xml_tbl['FGroups'][0][key]}")

            # Here is where I have defined the current keywords that we can obtain from Scamp XML output.
            # Basiclly each line in key_pairs relates a fits KEYWORD to a field in the XML.
            key_pairs['SCAMPCHI'] = {'fld': 'astromchi2_reference_highsn',
                                     'type': 'float',
                                     'comment': 'Chi2 value from SCAMP'}
            key_pairs['SCAMPNUM'] = {'fld': 'astromndets_reference_highsn',
                                     'type': 'int',
                                     'comment': 'Number of matched stars from SCAMP'}
            # key_pairs['SCAMPFLG']={'fld':'astromndets_reference_highsn','type':'int','comment':'Number of matched stars from SCAMP'}
            key_pairs['SCAMPREF'] = {'fld': 'astref_catalog',
                                     'type': 'str',
                                     'comment': 'Astrometric Reference Catalog used by SCAMP'}

            # Check for presence of each key.  If exists then try to parse based on type expected
            for key in key_pairs:
                if key_pairs[key]['fld'] in tmp_xml_tbl['FGroups'][0]:
                    if key_pairs[key]['type'] == "str":
                        keyval = unicodedata.normalize('NFKD', tmp_xml_tbl['FGroups'][0][key_pairs[key]['fld']]).encode('ascii', 'ignore')
                        tmpdict[key] = [keyval, key_pairs[key]['comment']]
                    elif key_pairs[key]['type'] == 'float':
                        try:
                            keyval = float(tmp_xml_tbl['FGroups'][0][key_pairs[key]['fld']])
                            tmpdict[key] = [keyval, key_pairs[key]['comment']]
                        except:
                            try:
                                keyval = float(unicodedata.normalize('NFKD', tmp_xml_tbl['FGroups'][0][key_pairs[key]['fld']]).encode('ascii', 'ignore'))
                                tmpdict[key] = [keyval, key_pairs[key]['comment']]
                            except:
                                print(f"Failed to parse value for {key_pairs[key]['fld']} = {tmp_xml_tbl['FGroups'][0][key_pairs[key]['fld']]} as float (skipping)")

                    elif key_pairs[key]['type'] == 'int':
                        try:
                            keyval = int(tmp_xml_tbl['FGroups'][0][key_pairs[key]['fld']])
                            tmpdict[key] = [keyval, key_pairs[key]['comment']]
                        except:
                            try:
                                keyval = int(unicodedata.normalize('NFKD', tmp_xml_tbl['FGroups'][0][key_pairs[key]['fld']]).encode('ascii', 'ignore'))
                                tmpdict[key] = [keyval, key_pairs[key]['comment']]
                            except:
                                print(f"Failed to parse value for {key_pairs[key]['fld']} = {tmp_xml_tbl['FGroups'][0][key_pairs[key]['fld']]} as int (skipping)")

                    else:
                        print('Unspecified value type for XML parsing of keywords (skipping)')

    # Translate into fitsio record format
    if translate:
        new_record = pyfitsrec2fitsiorecord(tmpdict)
    else:
        new_record = tmpdict

    return new_record

def fix_PVs(header, verbose=False):
    keys = ['PV1_3', 'PV2_3']
    for name in keys:
        if name not in header:
            if verbose:
                print("(updateWCS) Updating {}".format(name))
            rec = {'name': name,
                   'value': 0.0,
                   'comment': 'Projection distortion parameter'}
            header.add_record(rec)
    return header

def cmdline():

    import argparse

    parser = argparse.ArgumentParser(description='Update FITS header based on new WCS solution from SCAMP')
    parser.add_argument('-i', '--input', action='store', type=str, default=None, required=True,
                        help='Input Image (to be updated)')
    parser.add_argument('-o', '--output', action='store', type=str, default=None, required=True,
                        help='Output Image (updated image)')
    parser.add_argument('--headfile', action='store', type=str, default=None, required=True,
                        help='Headfile (containing most update information)')
    parser.add_argument('--hdupcfg', action='store', type=str, default=None, required=True,
                        help='Configuration file for header update')
    parser.add_argument('-f', '--fwhm', action='store', type=str, default=None,
                        help='update FWHM (argument is a FITS_LDAC catalog to be used to determine the FWHM)')
    parser.add_argument('--xml', action='store', type=str, default=None,
                        help='obtain limited QA info from SCAMP XML output (optional)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Full debug information to stdout')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Flag to produce more verbose output')
    parser.add_argument('--desepoch', action='store', type=str, default=None,
                        help='Optional DESEPOCH value to add to the SCI header')


    args = parser.parse_args()
    if args.verbose:
        print(f"Running {os.path.basename(__file__)} ")
        print("##### Initial arguments #####")
        print("Args: ", args)
    return args


def run_update(input_image, headfile, hdupcfg, verbose=False, new_record=[]):

    """
    Function to perform only header update once all information has been read/load.
    It is performed on an 'input_image' which is DESimage object
    """

    # Read in the header update configuration file
    keywords_spec = get_keywords_to_update(hdupcfg, verbose=verbose)

    # Read in the SCAMP .head using fitsio
    scamp_header = fitsio.read_scamp_head(headfile)
    scamp_header = fix_PVs(scamp_header, verbose=verbose)

    # Merge scamp_header and new_record. Because scamp_header is a FITSHDR object we can append to it
    [scamp_header.add_record(rec) for rec in new_record]

    # Update the headers on the input image, with all the new information
    input_image = update_headers(input_image, scamp_header, keywords_spec, verbose=verbose)

    return input_image


def run_updateWCS(args):

    # Attempt to populate FWHM, ELLIPTIC, NFWHMCNT keywords
    if args.fwhm:
        new_record = get_fwhm_from_catalog(args.fwhm, verbose=args.verbose, debug=args.debug)
    else:
        new_record = {}

    # Populate the new record with the XML and transalte into fitsio records format
    if args.xml:
        new_record = slurp_XML(args.xml, new_record, verbose=args.verbose, debug=args.debug, translate=True)
    else:
        new_record = []

    # Read in the input fits file using despyfits.DESImage
    input_image = DESImage.load(args.input)

    # run the main header updater
    input_image = run_update(input_image, headfile=args.headfile, hdupcfg=args.hdupcfg,
                             verbose=args.verbose, new_record=new_record)

    # if desepoch option, we add a DESPOCH record only the SCI plane
    if args.desepoch:
        desepoch_rec = {'name': 'DESEPOCH',
                        'value': args.desepoch,
                        'comment': 'DES Observing epoch'}
        print(f"(updateWCS): Updating DESEPOCH={args.desepoch} to SCI header")
        input_image.header.add_record(desepoch_rec)

    # Saving the image as args.output, we compute the corners at write time
    print(f"(updateWCS): Closing/Saving image --> {args.output}")
    input_image.save(args.output)


if __name__ == "__main__":

    # Get all of the args
    _args = cmdline()
    run_updateWCS(_args)
