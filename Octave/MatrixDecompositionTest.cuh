//
// Created by jeff on 1/13/23.
//

#ifndef JTQUANTCUDA_MATRIXDECOMPOSITIONTEST_CUH
#define JTQUANTCUDA_MATRIXDECOMPOSITIONTEST_CUH

#include <vector>

using namespace std;

namespace JTQuant::TestSuites {
    //sum of two vectors
    vector<double> sum(vector<double> a, vector<double> b);

    //minus of two vectors
    vector<double> minus(vector<double> a, vector<double> b);

    //inner product of two vectors
    double inner(vector<double> a, vector<double> b);

    //outer product of two vectors
    vector<vector<double> > outer(vector<double> a, vector<double> b);

    vector<vector<double>> transpose(vector<vector<double>> A);

    //scalar product with a vector
    vector<double> product(double c, vector<double> b);

    //matrix product with a vector
    vector<double> product(vector<vector<double> > M, vector<double> b);

    //matrix product with a matrix
    vector<vector<double>> product(vector<vector<double>> M, vector<vector<double>> N);

    vector<vector<double>> identity(int n);

    vector<vector<vector<double>>> householderQR2(vector<vector<double>> A);

    int vector_util_test();

    int householder_qr_test();


}

#endif //JTQUANTCUDA_MATRIXDECOMPOSITIONTEST_CUH
