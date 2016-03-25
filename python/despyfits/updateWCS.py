#!/usr/bin/env python

# $Id: UpdateScampHead.py 40667 2015-11-04 20:25:56Z felipe $
# $Rev:: 40667                            $:  # Revision of last commit.
# $LastChangedBy:: felipe                 $:  # Author of last commit.
# $LastChangedDate:: 2015-11-04 14:25:56 #$:  # Date of last commit.

# This is the set of libraries used to update the WCS of finalcut images

# Standard Libraries
import unicodedata
import os,sys
import numpy
import fitsio

# DESDM libraries
from despyfits.fwhmFromFITS_LDAC import fwhmFromFITS_LDAC
from despymisc.xmlslurp import Xmlslurper
from despyfits.DESImage import DESImage

def get_record(header,keyword):

    """ Utility option to gets the record form a FITSHDR header"""
    index = header._index_map[keyword]
    return header._record_list[index]

def update_headers(image, new_header, keywords_spec, verb=False):

    """
    Update the header in image using the FITSHRD object new_record for keywords present in keywords_spec
      image:          despyfits.DESIma object
      new_record:     fitsio.FITSHDR object
      keywords_spec:  a dictionary with keywords and HDU to use
    """
    
    # Loop over keywords and HDUs
    for keyword in keywords_spec:

        new_record = get_record(new_header,keyword)
        if verb:
            try:
                old_value = image.header[keyword]
            except:
                old_value = "Undefined"
            print "Updating keyword: %-8s from %s --> %s for %s" % (keyword,old_value, new_record['value'], keywords_spec[keyword])

        # Loop over HDU's
        for hdu in keywords_spec[keyword]:
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
               'value':old_record[name][0],
               'comment':old_record[name][1],
               }
        new_record.append(rec)
    return new_record

def get_keywords_to_update(args):

    """
    Read in the header update configuration file.
    Now all keywords that can be found/derived should be present are
    read from a configuration file
    """
    if not os.path.isfile(args.hdupcfg):
        exit ('Header update config file (%s) not found! ',args.hdupcfg)
    if args.verbose: print "(updateWCS): Reading the KEYWORD that we need want to UPDATE."

    kywds_spec = {}
    for line in open(args.hdupcfg).readlines():
        if line[0] == "#":
            continue
        keyword = line.split()[0]
        values  = line.split()[1]
        # Try converting str HDU to integer 
        try:
            kywds_spec[keyword] = [int(v) for v in values.split(',')]
        except:
            kywds_spec[keyword] = values.split(',')
            
    return kywds_spec 

def get_fwhm_from_catalog(args):

    """ Gets the FWHM and ELLIPTICITY """

    tmpdict = {}
    if args.fwhm:
        if args.verbose: print "(updateWCS): Will determine median FWHM & ELLIPTICITY and number of candidates"
        fwhm_med,ellp_med,count =fwhmFromFITS_LDAC(args.fwhm,debug=args.debug)
    else:
        if args.verbose: print "FWHM option keyword will not be populated\n"
        return tmpdict
    
    if args.verbose and not args.debug:
        print " FWHM    =%.4f" % fwhm_med   
        print " ELLIPTIC=%.4f" % ellp_med   
        print " NFWHMCNT=%s" % count
    tmpdict['FWHM']     = [round(fwhm_med,4),'Median FWHM from SCAMP input catalog [pixels]']   
    tmpdict['ELLIPTIC'] = [round(ellp_med,4),'Median Ellipticity from SCAMP input catalog']   
    tmpdict['NFWHMCNT'] = [count,'Number of objects used to find FWHM']
    return tmpdict

def slurp_XML(args,tmpdict, translate=True):

    """ Read in the XML File """

    key_pairs={}
    if args.xml:
        if args.verbose: print "Reading SCAMP XML file: {:s}".format(args.xml)
        try:
            tmp_xml_tbl=Xmlslurper(args.xml,['FGroups']).gettables()
            # Possible that should redefine the keys using the same unicodedata.normalize method as for the values below
        except:
            print "Error: failed to slurp the data."
            pass 

        if ('FGroups' in tmp_xml_tbl):
            if args.debug: 
                for key in tmp_xml_tbl['FGroups']:
                    print "  %s = %s" % (key,tmp_xml_tbl['FGroups'][key])
                    
            # Here is where I have defined the current keywords that we can obtain from Scamp XML output.
            # Basiclly each line in key_pairs relates a fits KEYWORD to a field in the XML.
            key_pairs['SCAMPCHI']={'fld':'astromchi2_reference_highsn','type':'float','comment':'Chi2 value from SCAMP'}
            key_pairs['SCAMPNUM']={'fld':'astromndets_reference_highsn','type':'int','comment':'Number of matched stars from SCAMP'}
            # key_pairs['SCAMPFLG']={'fld':'astromndets_reference_highsn','type':'int','comment':'Number of matched stars from SCAMP'}
            key_pairs['SCAMPREF']={'fld':'astref_catalog','type':'str','comment':'Astrometric Reference Catalog used by SCAMP'}

            # Check for presence of each key.  If exists then try to parse based on type expected
            for key in key_pairs:
                if (key_pairs[key]['fld'] in tmp_xml_tbl['FGroups']):
                    if (key_pairs[key]['type'] == "str"):
                        keyval=unicodedata.normalize('NFKD',tmp_xml_tbl['FGroups'][key_pairs[key]['fld']]).encode('ascii','ignore')
                        tmpdict[key]=[keyval,key_pairs[key]['comment']]
                    elif (key_pairs[key]['type'] == 'float'):
                        try:
                            keyval=float(tmp_xml_tbl['FGroups'][key_pairs[key]['fld']])
                            tmpdict[key]=[keyval,key_pairs[key]['comment']]
                        except:
                            try:
                                keyval=float(unicodedata.normalize('NFKD',tmp_xml_tbl['FGroups'][key_pairs[key]['fld']]).encode('ascii','ignore'))
                                tmpdict[key]=[keyval,key_pairs[key]['comment']]
                            except:
                                print 'Failed to parse value for %s = %s as float (skipping)' % ( key_pairs[key]['fld'], mp_xml_tbl['FGroups'][key_pairs[key]['fld']])
                                pass
                    elif (key_pairs[key]['type'] == 'int'):
                        try:
                            keyval=int(tmp_xml_tbl['FGroups'][key_pairs[key]['fld']])
                            tmpdict[key]=[keyval,key_pairs[key]['comment']]
                        except:
                            try:
                                keyval=int(unicodedata.normalize('NFKD',tmp_xml_tbl['FGroups'][key_pairs[key]['fld']]).encode('ascii','ignore'))
                                tmpdict[key]=[keyval,key_pairs[key]['comment']]
                            except:
                                print 'Failed to parse value for %s = %s as int (skipping)' % (key_pairs[key]['fld'], mp_xml_tbl['FGroups'][key_pairs[key]['fld']])
                                pass
                    else:
                        print 'Unspecified value type for XML parsing of keywords (skipping)'

    # Translate into fitsio record format
    if translate:
        new_record = pyfitsrec2fitsiorecord(tmpdict)
    else:
        new_record = tmpdict

    return new_record

def fix_PVs(header,args):
    keys = ['PV1_3','PV2_3']
    for name in keys:
        if name not in header:
            if args.verbose: print "(updateWCS) Updating %s" % name
            rec = {'name': name, 'value':0.0, 'comment':'Projection distortion parameter'}
            header.add_record(rec)
    return header

def cmdline():

    import argparse

    svnid = "%s $Id:40667 2015-11-04 20:25:56Z felipe $" % os.path.basename(__file__)
    svnrev = svnid.split(" ")[2]
    
    parser = argparse.ArgumentParser(description='Update FITS header based on new WCS solution from SCAMP')
    parser.add_argument('-i', '--input',  action='store', type=str, default=None, required=True, help='Input Image (to be updated)')
    parser.add_argument('-o', '--output', action='store', type=str, default=None, required=True, help='Output Image (updated image)')
    parser.add_argument('--headfile',     action='store', type=str, default=None, required=True, help='Headfile (containing most update information)')
    parser.add_argument('--hdupcfg',      action='store', type=str, default=None, required=True, help='Configuration file for header update')
    parser.add_argument('-f', '--fwhm',   action='store', type=str, default=None, help='update FWHM (argument is a FITS_LDAC catalog to be used to determine the FWHM)')
    parser.add_argument('--xml',          action='store', type=str, default=None, help='obtain limited QA info from SCAMP XML output (optional)')
    parser.add_argument('--debug',        action='store_true', default=False, help='Full debug information to stdout')
    parser.add_argument('-v','--verbose', action='store_true', default=False, help='Flag to produce more verbose output')

    args = parser.parse_args()
    if (args.verbose):
        print "Running %s " % svnid
        print "##### Initial arguments #####"
        print "Args: ",args
    return args

def run_updateWCS(args):


    # Attempt to populate FWHM, ELLIPTIC, NFWHMCNT keywords 
    new_record = get_fwhm_from_catalog(args)

    # Populate the new record with the XML and transalte into fitsio records format
    new_record = slurp_XML(args,new_record,translate=True)

    # Read in the header update configuration file
    keywords_spec = get_keywords_to_update(args)

    # Read in the SCAMP .head using fitsio
    scamp_header = fitsio.read_scamp_head(args.headfile)
    scamp_header = fix_PVs(scamp_header,args)

    # Merge scamp_header and new_record. Because scamp_header is a FITSHDR object we can append to it
    [scamp_header.add_record(rec) for rec in new_record]
    
    # Read in the input fits file using despyfits.DESImage
    input_image = DESImage.load(args.input)
    
    # Update the headers on the input image, with all the new information
    input_image = update_headers(input_image, scamp_header, keywords_spec)

    # Saving the image as args.output, we compute the corners at write time
    print "(updateWCS): Closing/Saving image --> %s" % args.output
    input_image.save(args.output)


if __name__ == "__main__":

    # Get all of the args
    args = cmdline()
    run_updateWCS(args)
