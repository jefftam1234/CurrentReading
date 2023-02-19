//
// Created by jeff on 1/13/23.
//

#ifndef JTQUANTCUDA_MATRIXDECOMPOSITIONTEST_CUH
#define JTQUANTCUDA_MATRIXDECOMPOSITIONTEST_CUH

#include <vector>

using namespace std;

namespace JTQuant::TestSuites {

    int vector_util_test();

    int householder_qr_cpu_test();

    int householder_qr_gpu_test();


}

#endif //JTQUANTCUDA_MATRIXDECOMPOSITIONTEST_CUH
