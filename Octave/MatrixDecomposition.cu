//
// Created by jeff on 1/13/23.
//

#include "MatrixDecomposition.cuh"
#include "CudaMarcos.cuh"
#include "CudaUtil.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include "VectorUtil.cuh"

using namespace std;


vector<vector<vector<double>>> JTQuant::Math::householderQR_CPU(vector<vector<double>> A) {
    int m = A.size();
    int n = A[0].size();
    vector<vector<double> > A_mod = A;
    vector<vector<double> > Q = Math::identity(m);

    for (int k = 0; k < n; k++) {
        //printf("k = %d\n", k);
        int e_size = m - k;
        //copy the vector x from the original matrix
        vector<double> x(e_size);
        for (int i = k; i < m; i++) x[i - k] = A_mod[i][k];
        vector<double> e(e_size, 0);
        e[0] = 1;
        double x_norm = sqrt(Math::inner(x, x));
        double sign_x = x[0] >= 0 ? 1 : -1;
        // create householder reflector vector u
        // (function) vector<double> u = sum(product(sign_x * x_norm, e), x);
        vector<double> u(e_size);
        for (int i = 0; i < e_size; i++) u[i] = sign_x * x_norm * e[i] + x[i];
        // v is the normalized vector u
        double v_norm = sqrt(Math::inner(u, u));
        vector<double> v(e_size, 0);
        for (int i = 0; i < e_size; i++) v[i] = u[i] / v_norm;

        vector<vector<double> > H(e_size, vector<double>(e_size, 0));
        for (int i = 0; i < e_size; i++) H[i][i] = 1;
        for (int i = 0; i < e_size; i++)
            for (int j = 0; j < e_size; j++)
                H[i][j] -= 2 * v[i] * v[j];

        vector<vector<double>> temp(e_size, vector<double>(n - k, 0));
        for (int i = 0; i < e_size; i++) {
            for (int j = 0; j < n - k; j++) {
                for (int l = 0; l < e_size; l++) {
                    //temp[i][j] += H[i][l] * A_mod[k + l][k + j];
                    double H_value = H[i][l];
                    double A_value = A_mod[k + l][k + j];
                    double res = H_value * A_value;
                    temp[i][j] += res;
//                    cout << "temp[" << i << "][" << j << "] += H[" << i << "][" << l << "] * A_mod[" << k + l << "]["
//                         << k + j
//                         << "] = " << H_value << " * " << A_value << " = " << res << endl;
                }
                //cout << "temp[" << i << "][" << j << "] = " << temp[i][j] << endl;
            }
        }


        for (int i = 0; i < e_size; i++)
            for (int j = 0; j < n - k; j++)
                A_mod[k + i][k + j] = temp[i][j];

        //printMatrixCPU(temp, "temp");

        // update Q with the current householder matrix
        vector<vector<double> > H_p(m, vector<double>(m, 0));
        for (int i = 0; i < m; i++) H_p[i][i] = 1;
        for (int i = 0; i < e_size; i++)
            for (int j = 0; j < e_size; j++)
                H_p[m - e_size + i][m - e_size + j] = H[i][j];

        vector<vector<double> > Q_temp(m, vector<double>(m, 0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                for (int l = 0; l < m; l++) {
                    Q_temp[i][j] += Q[i][l] * H_p[l][j];
                }
            }
        }
        Q = Q_temp;
    }

    vector<vector<double>> R(n, vector<double>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            R[i][j] = A_mod[i][j];
    return {Q, R};
}

__device__ void
JTQuant::Math::householderQR_device(double *A, int m, int n, double *x, double *e, double *u, double *v, double *H,
                                    double *temp) {
    for (int k = 0; k < n; k++) {
        //printf("k = %d\n", k);
        int e_size = m - k;
        for (int i = 0; i < e_size; i++) {
            if (i == 0) {
                e[i] = 1.0;
                continue;
            }
            e[i] = 0.0;
        }
        for (int i = k; i < m; i++) x[i - k] = A[i * n + k];
        double x_norm = 0;
        for (int i = 0; i < e_size; i++) x_norm += x[i] * x[i];
        x_norm = sqrt(x_norm);
        double sign_x = 0.0;

        if (x[0] >= 0) sign_x = 1;
        else sign_x = -1;

        for (int i = 0; i < e_size; i++) u[i] = sign_x * x_norm * e[i] + x[i];
        double u_norm = 0;
        for (int i = 0; i < e_size; i++) u_norm += u[i] * u[i];
        u_norm = sqrt(u_norm);
        for (int i = 0; i < e_size; i++) v[i] = u[i] / u_norm;


        //printVectorGPU(v, e_size, "v");

        for (int i = 0; i < e_size; i++) {
            for (int j = 0; j < e_size; j++) {
                H[i * e_size + j] = (i == j) ? 1 : 0;
                H[i * e_size + j] -= 2 * v[i] * v[j];
            }
        }

        //printMatrixGPU(H, e_size, e_size, "H");

        for (int i = 0; i < e_size; i++) {
            for (int j = 0; j < n - k; j++) {
                for (int l = 0; l < e_size; l++) {
                    //temp[i * (n - k) + j] += H[i * e_size + l] * A[(k + l) * n + (k + j)];   //problem here, step 2, H is right, A should be right (indicing is wrong?)
                    double H_value = H[i * e_size + l];
                    double A_value = A[(k + l) * n + (k + j)];
                    double res = H_value * A_value;
                    temp[i * (n - k) + j] += res;
                    //printf("temp[%d][%d] += H[%d][%d] * A[%d][%d] = %f * %f = %f\n", i, j, i, l, k + l, k + j, H_value, A_value, res);
                }
                //printf("temp[%d][%d] = %f\n", i, j, temp[i * (n - k) + j]);
            }
        }

        for (int i = 0; i < e_size; i++) {
            for (int j = 0; j < n - k; j++) {
                A[(k + i) * n + (k + j)] = temp[i * (n - k) + j];
                temp[i * (n - k) + j] = 0.0;   //initialized
            }
        }
        //printMatrixGPU(temp, e_size, n - k, "temp");
    }
}

__global__ void
JTQuant::Math::householderQR_global(double *A, int m, int n, double *x, double *e, double *u, double *v, double *H,
                                    double *temp) {
    householderQR_device(A, m, n, x, e, u, v, H, temp);
}

