#ifndef CP_PROJECT_HISTOGRAM_PAR_H
#define CP_PROJECT_HISTOGRAM_PAR_H

#include "wb.h"

namespace cp_par {
    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations = 1);
}

#endif //CP_PROJECT_HISTOGRAM_PAR_H