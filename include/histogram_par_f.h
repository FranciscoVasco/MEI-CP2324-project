#ifndef CP_PROJECT_HISTOGRAM_PAR_F_H
#define CP_PROJECT_HISTOGRAM_PAR_F_H

#include "wb.h"

namespace cp_par_f {
    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations = 1);
}

#endif //CP_PROJECT_HISTOGRAM_PAR_F_H