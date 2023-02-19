//
// Created by jeff on 2/16/23.
//

#ifndef JTQUANTCUDA_VECTORUTIL_CUH
#define JTQUANTCUDA_VECTORUTIL_CUH

#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

namespace JTQuant::Math {

    //sum of two vectors
    vector<double> sum(vector<double> a, vector<double> b);

    //minus of two vectors
    vector<double> minus(vector<double> a, vector<double> b);

    //inner product of two vectors
    double inner(vector<double> a, vector<double> b);

    //outer product of two vectors
    vector<vector<double> > outer(vector<double> a, vector<double> b);

    //transpose of a matrix
    vector<vector<double>> transpose(vector<vector<double>> A);

    //scalar product with a vector
    vector<double> product(double c, vector<double> b);

    //matrix product with a vector
    vector<double> product(vector<vector<double> > M, vector<double> b);

    //matrix product with a matrix
    vector<vector<double>> product(vector<vector<double>> M, vector<vector<double>> N);

    //identity matrix
    vector<vector<double>> identity(int n);

    void printMatrixCPU(const vector<vector<double>> matrix, const char *name);

    __device__ void printVectorGPU(double *vec, int size, const char *name);

    __device__ void printMatrixGPU(double *mat, int rows, int cols, const char *name);

} // JTQuant::Math

#endif //JTQUANTCUDA_VECTORUTIL_CUH
