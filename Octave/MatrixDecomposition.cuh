//
// Created by jeff on 1/13/23.
//

#ifndef JTQUANTCUDA_MATRIXDECOMPOSITION_CUH
#define JTQUANTCUDA_MATRIXDECOMPOSITION_CUH

#include <vector>


namespace JTQuant::Math {

    std::vector<std::vector<std::vector<double>>> householderQR_CPU(std::vector<std::vector<double>> A);

    //void householderQR(double **A, int m, int n, double **Q, double **R);

    __global__ void
    householderQR_global(double *A, int m, int n, double *x, double *e, double *u, double *v, double *H, double *temp);

    __device__ void
    householderQR_device(double *A, int m, int n, double *x, double *e, double *u, double *v, double *H, double *temp);

}


#endif //JTQUANTCUDA_MATRIXDECOMPOSITION_CUH
