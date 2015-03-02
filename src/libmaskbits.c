#include "mask_bits.h"

/* define BADPIX bit mappings (for MASK HDU) */
extern const int badpix_bpm = BADPIX_BPM;
extern const int badpix_saturate = BADPIX_SATURATE;
extern const int badpix_interp = BADPIX_INTERP;
extern const float badpix_threshold = BADPIX_THRESHOLD;
/* extern const int badpix_low = BADPIX_LOW;   Should not be in use  */
extern const int badpix_cray = BADPIX_CRAY;
extern const int badpix_star = BADPIX_STAR;
extern const int badpix_trail = BADPIX_TRAIL;
extern const int badpix_edgebleed = BADPIX_EDGEBLEED;
extern const int badpix_ssxtalk = BADPIX_SSXTALK;
extern const int badpix_edge = BADPIX_EDGE;
extern const int badpix_streak = BADPIX_STREAK;
/* extern const int badpix_fix = BADPIX_FIX;  Should not be in use */
extern const int badpix_suspect = BADPIX_SUSPECT;
extern const int badpix_badamp = BADPIX_BADAMP;

/* define BPMDEF bit mappings (for BPM definition) */
extern const int bpmdef_flat_min = BPMDEF_FLAT_MIN;
extern const int bpmdef_flat_max = BPMDEF_FLAT_MAX;
extern const int bpmdef_flat_mask = BPMDEF_FLAT_MASK;
extern const int bpmdef_bias_hot = BPMDEF_BIAS_HOT;
extern const int bpmdef_bias_warm = BPMDEF_BIAS_WARM;
extern const int bpmdef_bias_mask = BPMDEF_BIAS_MASK;
extern const int bpmdef_bias_col = BPMDEF_BIAS_COL;
extern const int bpmdef_edge = BPMDEF_EDGE;
extern const int bpmdef_corr = BPMDEF_CORR;
extern const int bpmdef_tape_bump = BPMDEF_TAPE_BUMP;
extern const int bpmdef_funky_col = BPMDEF_FUNKY_COL;
extern const int bpmdef_wacky_pix = BPMDEF_WACKY_PIX;
extern const int bpmdef_badamp = BPMDEF_BADAMP;
/* extern const int bpmdef_generic = BPMDEF_GENERIC; Should not be in use*/
