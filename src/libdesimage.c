#include <stdlib.h>
#include "desimage.h"

extern const int ccdnum2 = CCDNUM2;

/* Use a proc to set these entries in the structure rather than
 * direct assigment by ctypes so we can use the np.ndpointer facitilty
 * which checks that the numpy arrays have the proper structure.
 */
int set_desimage(float *image, float *varim, short *mask, desimage *di) {
  di->image = image;
  di->varim = varim;
  di->mask = mask;
  return(0);
}

int set_data_desimage(float *image, desimage *di) {
  di->image = image;
  di->varim = NULL;
  di->mask = NULL;
  return(0);
}


int set_weightless_desimage(float *image, short *mask, desimage *di) {
  di->image = image;
  di->varim = NULL;
  di->mask = mask;
  return(0);
}

int set_bpm_desimage(short *bpm, desimage *di) {
  di->image = NULL;
  di->varim = NULL;
  di->mask = bpm;
  return(0);
}

