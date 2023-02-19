//
// Created by jeff on 2/16/23.
//

#include "VectorUtil.cuh"

namespace JTQuant::Math {

    //sum of two vectors
    vector<double> sum(vector<double> a, vector<double> b) {
        int n = a.size();
        vector<double> c(n);
        for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
        return c;
    }

    //minus of two vectors
    vector<double> minus(vector<double> a, vector<double> b) {
        int n = a.size();
        vector<double> c(n);
        for (int i = 0; i < n; i++) c[i] = a[i] - b[i];
        return c;
    }

    //inner product of two vectors
    double inner(vector<double> a, vector<double> b) {
        int n = a.size();
        double res = 0;
        for (int i = 0; i < n; i++) res += a[i] * b[i];
        return res;
    }

    //outer product of two vectors
    vector<vector<double>> outer(vector<double> a, vector<double> b) {
        int n = a.size();
        int m = b.size();
        vector<vector<double>> res(n, vector<double>(m, 0));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                res[i][j] = a[i] * b[j];
        return res;
    }

    vector<vector<double>> transpose(vector<vector<double>> A) {
        int m = A.size();
        int n = A[0].size();
        vector<vector<double>> AT(n, vector<double>(m, 0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                AT[j][i] = A[i][j];
            }
        }
        return AT;
    }

    //scalar product with a vector
    vector<double> product(double c, vector<double> b) {
        int n = b.size();
        vector<double> res(n, 0);
        for (int i = 0; i < n; i++) res[i] = c * b[i];
        return res;
    }

    //matrix product with a vector
    vector<double> product(vector<vector<double>> M, vector<double> b) {
        int n = M.size();
        int m = b.size();
        vector<double> res(n, 0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) {
                res[i] += M[i][j] * b[j];
            }
        return res;
    }

    vector<vector<double>> product(vector<vector<double>> M, vector<vector<double>> N) {
        // function to return the product of two matrices
        int m = M.size();
        int n = N[0].size();
        int p = N.size();
        vector<vector<double>> C(m, vector<double>(n, 0));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < p; k++)
                    C[i][j] += M[i][k] * N[k][j];
        return C;
    }

    vector<vector<double>> identity(int n) {
        vector<vector<double>> I(n, vector<double>(n, 0));
        for (int i = 0; i < n; i++)
            I[i][i] = 1;
        return I;
    }

    void printMatrixCPU(const vector<vector<double>> matrix, const char *name) {
        printf("%s:\n", name);
        int rows = matrix.size();
        int cols = matrix[0].size();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("  matrix[%d][%d] = %f\n", i, j, matrix[i][j]);
            }
            std::cout << std::endl;
        }
    }


    __device__ void printVectorGPU(double *vec, int size, const char *name) {
        printf("%s:\n", name);
        for (int i = 0; i < size; i++) {
            printf("  vec[%d] = %f\n", i, vec[i]);
        }
    }


    __device__ void printMatrixGPU(double *mat, int rows, int cols, const char *name) {
        printf("%s:\n", name);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("  mat[%d][%d] = %f\n", i, j, mat[i * cols + j]);
            }
            printf("\n");
        }
    }
} // JTQuant::Math