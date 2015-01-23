#include "desimage.h"

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

