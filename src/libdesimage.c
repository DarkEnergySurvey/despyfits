#include <stdlib.h>
#include "desimage.h"

extern const int ccdnum2 = CCDNUM2;

/* Use a proc to set these entries in the structure rather than
 * direct assigment by ctypes so we can use the np.ndpointer facitilty
 * which checks that the numpy arrays have the proper structure.
 */
int set_desimage(float *image, float *variance, float *weight, short *mask, desimage *di) {
  di->image = image;
  di->variance = variance;
  di->weight = weight;
  di->mask = mask;
  return(0);
}
