//
// Created by jeff on 1/13/23.
//


#include <iostream>
#include <iomanip>
#include <cassert>
#include "MatrixDecompositionTest.cuh"
#include "VectorUtil.cuh"
#include "MatrixDecomposition.cuh"

#define EPSILON 1e-8

using namespace std;

int JTQuant::TestSuites::vector_util_test() {
    vector<double> a = {1, 2, 3};
    vector<double> b = {4, 5, 6};
    vector<double> c = Math::sum(a, b);
    assert(c[0] == 5 && c[1] == 7 && c[2] == 9);
    vector<double> d = Math::minus(a, b);
    assert(d[0] == -3 && d[1] == -3 && d[2] == -3);
    double e = Math::inner(a, b);
    assert(e == 32);
    vector<vector<double> > f = Math::outer(a, b);
    assert(f[0][0] == 4 && f[0][1] == 5 && f[0][2] == 6);
    assert(f[1][0] == 8 && f[1][1] == 10 && f[1][2] == 12);
    assert(f[2][0] == 12 && f[2][1] == 15 && f[2][2] == 18);
    vector<double> g = Math::product(f, c);
    assert(g[0] == 109 && g[1] == 218 && g[2] == 327);
    vector<vector<double> > A = {{1, 2, 3},
                                 {4, 5, 6}};
    vector<vector<double> > B = {{1, 2},
                                 {3, 4},
                                 {5, 6}};

    vector<vector<double> > C = Math::product(A, B);
    assert(C[0][0] == 22 && C[0][1] == 28);
    assert(C[1][0] == 49 && C[1][1] == 64);

    return 0;
}


int JTQuant::TestSuites::householder_qr_cpu_test() {
    vector<vector<double> > A = {
            {1.0, -1.0, 4.0},
            {1.0, 4.0,  -2.0},
            {1.0, 4.0,  2.0},
            {1.0, -1.0, 0.0}
    };

    vector<vector<vector<double>>> QR = JTQuant::Math::householderQR_CPU(A);
    vector<vector<double>> Q = QR[0];
    vector<vector<double>> R = QR[1];

//    cout << "Q: " << endl;
//    for (int i = 0; i < Q.size(); i++) {
//        for (int j = 0; j < Q[0].size(); j++) {
//            cout << setw(12) << setfill(' ') << fixed << setprecision(6) << Q[i][j];
//        }
//        cout << endl;
//    }
//
//    cout << "R:" << endl;
//    for (int i = 0; i < R.size(); i++) {
//        for (int j = 0; j < R[0].size(); j++)
//            cout << setw(12) << setfill(' ') << fixed << setprecision(6) << R[i][j];
//        cout << endl;
//    }

    vector<vector<double>> transpose_R = Math::transpose(R);
    vector<vector<double>> RTR = Math::product(transpose_R, R);

    vector<vector<double>> expected_R = {
            {-2.0, -3.0, -2.0},
            {0.0,  -5.0, 2.0},
            {0.0,  0.0,  -4.0}
    };
// Assert statement
    for (int i = 0; i < R.size(); i++) {
        for (int j = 0; j < R[0].size(); j++) assert(fabs(R[i][j] - expected_R[i][j]) < EPSILON);
    }

//    cout << "RTR:" << endl;
//    for (int i = 0; i < RTR.size(); i++) {
//        for (int j = 0; j < RTR[0].size(); j++)
//            cout << setw(12) << setfill(' ') << fixed << setprecision(6) << RTR[i][j];
//        cout << endl;
//    }

    return 0;
}


int JTQuant::TestSuites::householder_qr_gpu_test() {

    vector<vector<double>> A_h = {
            {1.0, -1.0, 4.0},
            {1.0, 4.0,  -2.0},
            {1.0, 4.0,  2.0},
            {1.0, -1.0, 0.0}
    };
    int m = A_h.size(); // number of rows in the matrix A
    int n = A_h[0].size(); // number of columns in the matrix A

    double *A_d; // device array for the matrix A
    double *x, *e, *u, *v, *H, *temp;
    cudaMallocManaged((void **) &A_d, m * n * sizeof(double));
    cudaMallocManaged((void **) &x, m * sizeof(double));
    cudaMallocManaged((void **) &e, m * sizeof(double));
    cudaMallocManaged((void **) &u, m * sizeof(double));
    cudaMallocManaged((void **) &v, m * sizeof(double));
    cudaMallocManaged((void **) &H, m * m * sizeof(double));
    cudaMallocManaged((void **) &temp, m * (n - 0) * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A_d[i * n + j] = A_h[i][j];
        }
    }
    dim3 grid(1, 1);
    dim3 block(1, 1);
    JTQuant::Math::householderQR_global<<<grid, block>>>(A_d, m, n, x, e, u, v, H, temp);
    cudaDeviceSynchronize();

//    cout << "R:" << endl;
//    for (int i = 0; i < m; i++) {
//        for (int j = 0; j < n; j++)
//            cout << setw(12) << setfill(' ') << fixed << setprecision(6) << A_d[i * n + j];
//        cout << endl;
//    }
    cudaFree(x);
    cudaFree(e);
    cudaFree(u);
    cudaFree(v);
    cudaFree(H);
    cudaFree(temp);
    // Parse matrix
    vector<vector<double>> expected_R = {
            {-2.0, -3.0, -2.0},
            {0.0,  -5.0, 2.0},
            {0.0,  0.0,  -4.0},
            {0.0,  0.0,  0.0}
    };
    // Assert statement
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double expected = expected_R[i][j];
            assert(fabs(A_d[i * n + j] - expected) < EPSILON);
        }
    }
    return 0;
}




