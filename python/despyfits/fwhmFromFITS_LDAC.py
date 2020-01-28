import sys
import fitsio
import numpy

CLASSLIM = 0.75 # class threshold to define star
MAGERRLIMIT = 0.1  # mag error threshold for stars
REQUIRED_COLS = ['FWHM_IMAGE', 'ELLIPTICITY', 'FLAGS', 'IMAFLAGS_ISO', 'MAGERR_AUTO', 'CLASS_STAR']
OPTIONAL_COLS = ['IMAFLAGS_ISO']

def fwhmFromFITS_LDAC(incat, debug=False):
    """
    Get the median FWHM and ELLIPTICITY from the scamp (FITS_LDAC) catalog (incat)
    """

    if debug:
        print(f"(fwhmFromFITS_LDAC) Opening scamp_cat ({incat:s}) to calculate median FWHM & ELLIPTICITY.")

    # Read in catalog and LDAC_OBJECTS header using fitsio
    fits = fitsio.FITS(incat, namemode='r')
    header = fits['LDAC_OBJECTS'].read_header()

    if debug:
        print("(fwhmFromFITS_LDAC) Checking to see that LDAC_OBJECTS in scamp_cat is a binary table.")

    # Check for problems with XTENSION
    if 'XTENSION' not in header:
        print("(fwhmFromFITS_LDAC) ERROR: XTENSION keyword not found")
        sys.exit(1)
    elif header['XTENSION'] != 'BINTABLE':
        print("Error: (fwhmFromFITS_LDAC): this HDU is not a binary table")
        sys.exit(1)

    # Check for problems with NAXIS2
    if 'NAXIS2' not in header:
        print("Error: (fwhmFromFITS_LDAC) NAXIS2 keyword not found")
        sys.exit(1)
    else:
        nrows = header['NAXIS2']
        print(f"(fwhmFromFITS_LDAC) Found {nrows} rows in table")

    # Now we can read in the table and all of the columsn names
    table = fits['LDAC_OBJECTS'].read()
    table_cols = fits['LDAC_OBJECTS'].get_colnames()

    # Check that the columns REQUIRED are present
    for colname in REQUIRED_COLS:
        if colname not in table_cols:
            if colname in OPTIONAL_COLS:
                print(f"(fwhmFromFITS_LDAC): Optional column {colname} not present in binary table")
            else:
                print(f"(fwhmFromFITS_LDAC): Required column {colname} not present in binary table")
                sys.exit(1)

    # Now select the object that we want according to cuts
    flags = table['FLAGS']
    cstar = table['CLASS_STAR']
    mgerr = table['MAGERR_AUTO']
    fwhm = table['FWHM_IMAGE']
    ellp = table['ELLIPTICITY']

    idx = numpy.where((flags < 1) & (cstar > CLASSLIM) & (mgerr < MAGERRLIMIT) & (fwhm > 0.5) & (ellp >= 0.0))
    count = len(idx[0])

    # allow the no-stars case count = 0 to proceed without crashing
    if count <= 0:
        fwhm_med = 4.0
        ellp_med = 0.0
    else:
        fwhm_med = numpy.median(fwhm[idx])
        ellp_med = numpy.median(ellp[idx])

    if debug:
        print(f"(fwhmFromFITS_LDAC):     FWHM={fwhm_med:.4f} ")
        print(f"(fwhmFromFITS_LDAC): ELLIPTIC={ellp_med:.4f} ")
        print(f"(fwhmFromFITS_LDAC): NFWHMCNT={count} ")

    return fwhm_med, ellp_med, count
