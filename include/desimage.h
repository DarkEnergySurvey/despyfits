
#define CCDNUM2 76

/* This struct must match DESImageStruct */

typedef struct {
  long npixels;
  long axes[7];
  float exptime;
  int ampsecan[4];
  int ampsecbn[4];
  float saturateA;
  float saturateB;
  float gainA;
  float gainB;
  float rdnoiseA;
  float rdnoiseB;
  float *image;
  float *variance;
  float *weight;
  short *mask;
} desimage;
