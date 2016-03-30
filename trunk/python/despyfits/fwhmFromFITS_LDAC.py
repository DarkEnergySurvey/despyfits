
import fitsio
import numpy

CLASSLIM        = 0.75 # class threshold to define star
MAGERRLIMIT     = 0.1  # mag error threshold for stars
REQUIRED_COLS = ['FWHM_IMAGE','ELLIPTICITY','FLAGS','IMAFLAGS_ISO','MAGERR_AUTO','CLASS_STAR']
OPTIONAL_COLS = ['IMAFLAGS_ISO']

def fwhmFromFITS_LDAC(incat,debug=False):
    """
    Get the median FWHM and ELLIPTICITY from the scamp (FITS_LDAC) catalog (incat)
    """

    if debug:
        print "(fwhmFromFITS_LDAC) Opening scamp_cat ({:s}) to calculate median FWHM & ELLIPTICITY.".format(incat)

    # Read in catalog and LDAC_OBJECTS header using fitsio
    fits   = fitsio.FITS(incat,namemode='r')
    header = fits['LDAC_OBJECTS'].read_header()

    if debug:
        print "(fwhmFromFITS_LDAC) Checking to see that LDAC_OBJECTS in scamp_cat is a binary table."

    # Check for problems with XTENSION
    if 'XTENSION' not in header:
        exit("(fwhmFromFITS_LDAC) ERROR: XTENSION keyword not found")
    elif header['XTENSION'] != 'BINTABLE':
        exit("Error: (fwhmFromFITS_LDAC): this HDU is not a binary table")

    # Check for problems with NAXIS2
    if 'NAXIS2' not in header:
        exit("Error: (fwhmFromFITS_LDAC) NAXIS2 keyword not found")
    else:
        nrows = header['NAXIS2']
        print "(fwhmFromFITS_LDAC) Found %s rows in table" % nrows

    # Now we can read in the table and all of the columsn names
    table      = fits['LDAC_OBJECTS'].read()
    table_cols = fits['LDAC_OBJECTS'].get_colnames() 

    # Check that the columns REQUIRED are present
    for colname in REQUIRED_COLS:
        if colname not in table_cols:
            if colname in OPTIONAL_COLS:
                print "(fwhmFromFITS_LDAC): Optional column %s not present in binary table" % colname
            else:
                exit ("(fwhmFromFITS_LDAC): Required column %s not present in binary table" % colname)

    # Now select the object that we want according to cuts
    flags = table['FLAGS']
    cstar = table['CLASS_STAR']
    mgerr = table['MAGERR_AUTO']
    fwhm  = table['FWHM_IMAGE']
    ellp  = table['ELLIPTICITY']

    idx = numpy.where( (flags<1) & (cstar > CLASSLIM) & (mgerr< MAGERRLIMIT) & (fwhm>0.5) & (ellp >=0.0))
    count = len(idx[0])

    # allow the no-stars case count = 0 to proceed without crashing
    if count <= 0:
        fwhm_med = 4.0
        ellp_med = 0.0
    else:
        fwhm_med = numpy.median(fwhm[idx])
        ellp_med = numpy.median(ellp[idx])

    if debug:
        print "(fwhmFromFITS_LDAC):     FWHM=%.4f " % (fwhm_med)
        print "(fwhmFromFITS_LDAC): ELLIPTIC=%.4f " % (ellp_med)
        print "(fwhmFromFITS_LDAC): NFWHMCNT=%s " % (count)

    return fwhm_med,ellp_med,count

