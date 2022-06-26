#include "utils/gogui_helper.h"
#include "utils/format.h"

#include <cmath>
#include <algorithm>

/* Convert HSV colorspace to RGB
 * https://stackoverflow.com/a/6930407 */

void HsvToRgb(float h, float s, float v, int *r, int *g, int *b) {
    float hh, p, q, t, ff;
    int   i;

    if (s <= 0.0) {       // < is bogus, just shuts up warnings
        *r = v;  *g = v;  *b = v;  return;
    }

    hh = h;
    if (hh >= 360.0)  hh = 0.0;
    hh /= 60.0;
    i = (int)hh;
    ff = hh - i;
    p = v * (1.0 - s);
    q = v * (1.0 - (s * ff));
    t = v * (1.0 - (s * (1.0 - ff)));

    switch(i) {
        case 0:  *r = 255.0 * v;  *g = 255.0 * t;  *b = 255.0 * p;  break;
        case 1:  *r = 255.0 * q;  *g = 255.0 * v;  *b = 255.0 * p;  break;
        case 2:  *r = 255.0 * p;  *g = 255.0 * v;  *b = 255.0 * t;  break;
        case 3:  *r = 255.0 * p;  *g = 255.0 * q;  *b = 255.0 * v;  break;
        case 4:  *r = 255.0 * t;  *g = 255.0 * p;  *b = 255.0 * v;  break;
        case 5:
        default: *r = 255.0 * v;  *g = 255.0 * p;  *b = 255.0 * q;  break;
    }
}

void ValueToColor(float val, int *r, int *g, int *b) {
    /* Shrink cyan range, too bright:
     * val: [ 1.0                                        0.0 ]
     *   h: [  0                    145           215    242 ]
     *      [ red....................[.....cyan....]....blue ]  <- linear mapping
     *      [ .......................[. . . . . . .]....blue ]  <- we want this
     */
    int h1 = 145, h2 = 215;
    int w = h2 - h1;  /* orig cyan range, 70 */
    int w2 = 20;      /* new one */

    float h = (1.0 - val) * (242 - w + w2);
    float s = 1.0;
    float v = 1.0;

    /* Convert fake cyan range, and decrease lightness. */
    if (h1 <= h && h <= h1 + w2) {
        h = h1 + (h - h1) * w / w2;
        int m = w / 2;
        v -= (m - fabsf(h - (h1 + m))) * 0.2 / m;
    } else if (h >= h1 + w2) {
        h += w - w2;
    }

    /* Also decrease green range lightness. */
    int h0 = 100;  int m0 = (h2 - h0) / 2;
    if (h0 <= h && h <= h2)
        v -= (m0 - fabsf(h - (h0 + m0))) * 0.2 / m0;

    //fprintf(stderr, "h: %i\n", (int)h);
    HsvToRgb(h, s, v, r, g, b);
}

void ValueToGray(float val, int *r, int *g, int *b) {
    /* 
     * val: [ 1.0                                        0.0 ]
     *   v: [  0                    145           215    255 ]
     */

    float h = 0.f;
    float s = 0.f;
    float v = (1.f-val) * (255 - 0);

    HsvToRgb(h, s, v, r, g, b);
}

float FancyClamp(float val) {
    val = std::max(0.f, val);
    val = std::min(1.f, val);
    return val;
}

std::string GoguiColor(float val, std::string vtx) {
    val = FancyClamp(val);

    int rr, gg, bb;

    ValueToColor(val, &rr, &gg, &bb);

    return Format("COLOR #%02x%02x%02x %s", rr, gg, bb, vtx.c_str());
}

std::string GoguiGray(float val, std::string vtx, bool inves) {
    val = FancyClamp(val);

    if (inves) {
        val = 1.f - val;
    }
    int rr, gg, bb;

    ValueToGray(val, &rr, &gg, &bb);

    return Format("COLOR #%02x%02x%02x %s", rr, gg, bb, vtx.c_str());
}

std::string GoguiLable(float val, std::string vtx) {
    val = FancyClamp(val);

    return Format("LABEL %d %s", int(val * 100.f), vtx.c_str());
}
