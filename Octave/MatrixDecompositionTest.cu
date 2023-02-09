//
// Created by jeff on 1/13/23.
//


#include <iostream>
#include <iomanip>
#include "MatrixDecompositionTest.cuh"

//sum of two vectors
vector<double> JTQuant::TestSuites::sum(vector<double> a, vector<double> b) {
    int n = a.size();
    vector<double> c(n);
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
    return c;
}

//minus of two vectors
vector<double> JTQuant::TestSuites::minus(vector<double> a, vector<double> b) {
    int n = a.size();
    vector<double> c(n);
    for (int i = 0; i < n; i++) c[i] = a[i] - b[i];
    return c;
}

//inner product of two vectors
double JTQuant::TestSuites::inner(vector<double> a, vector<double> b) {
    int n = a.size();
    double res = 0;
    for (int i = 0; i < n; i++) res += a[i] * b[i];
    return res;
}

//outer product of two vectors
vector<vector<double>> JTQuant::TestSuites::outer(vector<double> a, vector<double> b) {
    int n = a.size();
    int m = b.size();
    vector<vector<double> > res(n, vector<double>(m, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res[i][j] = a[i] * b[j];
    return res;
}

vector<vector<double>> JTQuant::TestSuites::transpose(vector<vector<double> > A) {
    int m = A.size();
    int n = A[0].size();
    vector<vector<double> > AT(n, vector<double>(m, 0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            AT[j][i] = A[i][j];
        }
    }
    return AT;
}

//scalar product with a vector
vector<double> JTQuant::TestSuites::product(double c, vector<double> b) {
    int n = b.size();
    vector<double> res(n, 0);
    for (int i = 0; i < n; i++) res[i] = c * b[i];
    return res;
}

//matrix product with a vector
vector<double> JTQuant::TestSuites::product(vector<vector<double> > M, vector<double> b) {
    int n = M.size();
    int m = b.size();
    vector<double> res(n, 0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res[i] += M[i][j] * b[j];
    return res;
}

vector<vector<double>> JTQuant::TestSuites::product(vector<vector<double>> M, vector<vector<double>> N) {
    // function to return the product of two matrices
    int m = M.size();
    int n = N[0].size();
    int p = N.size();
    vector<vector<double> > C(m, vector<double>(n, 0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < p; k++)
                C[i][j] += M[i][k] * N[k][j];
    return C;
}


vector<vector<double>> JTQuant::TestSuites::identity(int n) {
    vector<vector<double>> I(n, vector<double>(n, 0));
    for (int i = 0; i < n; i++)
        I[i][i] = 1;
    return I;
}

int JTQuant::TestSuites::vector_util_test() {
    cout << "testing vector_util_test.\n";
    vector<double> a = {1, 2, 3};
    vector<double> b = {4, 5, 6};
    cout << "vector a: ";
    for (int i = 0; i < a.size(); i++) cout << a[i] << " ";
    cout << endl;
    cout << "vector b: ";
    for (int i = 0; i < b.size(); i++) cout << b[i] << " ";
    cout << endl;

    vector<double> c = sum(a, b);
    cout << "Sum of a and b: ";
    for (int i = 0; i < c.size(); i++) cout << c[i] << " ";
    cout << endl;

    vector<double> d = minus(a, b);
    cout << "Difference of a and b: ";
    for (int i = 0; i < d.size(); i++) cout << d[i] << " ";
    cout << endl;

    double e = inner(a, b);
    cout << "Inner product of a and b: " << e << endl;

    vector<vector<double> > f = outer(a, b);
    cout << "Outer product of a and b: " << endl;
    for (int i = 0; i < f.size(); i++) {
        for (int j = 0; j < f[0].size(); j++) cout << f[i][j] << " ";
        cout << endl;
    }

    vector<double> g = product(f, c);
    cout << "Product of f and c: ";
    for (int i = 0; i < g.size(); i++) cout << g[i] << " ";
    cout << endl;

    vector<vector<double> > A = {{1, 2, 3},
                                 {4, 5, 6}};
    vector<vector<double> > B = {{1, 2},
                                 {3, 4},
                                 {5, 6}};

    vector<vector<double> > C = product(A, B);
    cout << "Product of A and B:" << endl;
    for (int i = 0; i < C.size(); i++) {
        for (int j = 0; j < C[0].size(); j++) cout << C[i][j] << " ";
        cout << endl;
    }

    return 0;
}

vector<vector<vector<double>>> JTQuant::TestSuites::householderQR(vector<vector<double>> A) {
    int m = A.size();
    int n = A[0].size();
    vector<vector<double> > A_mod = A;
    vector<vector<double> > Q = identity(m);

    for (int k = 0; k < n; k++) {
        int e_size = m - k;
        //copy the vector x from the original matrix
        vector<double> x(e_size);
        for (int i = k; i < m; i++) x[i - k] = A_mod[i][k];
        vector<double> e(e_size, 0);
        e[0] = 1;
        double x_norm = sqrt(inner(x, x));
        double sign_x = x[0] >= 0 ? 1 : -1;
        // create householder reflector vector u
        // (function) vector<double> u = sum(product(sign_x * x_norm, e), x);
        vector<double> u(e_size);
        for (int i = 0; i < e_size; i++) u[i] = sign_x * x_norm * e[i] + x[i];
        // v is the normalized vector u
        double v_norm = sqrt(inner(u, u));
        vector<double> v(e_size, 0);
        for (int i = 0; i < e_size; i++) v[i] = u[i] / v_norm;

        vector<vector<double> > H(e_size, vector<double>(e_size, 0));
        for (int i = 0; i < e_size; i++) H[i][i] = 1;
        for (int i = 0; i < e_size; i++)
            for (int j = 0; j < e_size; j++)
                H[i][j] -= 2 * v[i] * v[j];

        vector<vector<double> > temp(e_size, vector<double>(n - k, 0));
        for (int i = 0; i < e_size; i++)
            for (int j = 0; j < n - k; j++)
                for (int l = 0; l < e_size; l++)
                    temp[i][j] += H[i][l] * A_mod[k + l][k + j];

        for (int i = 0; i < e_size; i++)
            for (int j = 0; j < n - k; j++)
                A_mod[k + i][k + j] = temp[i][j];

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

int JTQuant::TestSuites::householder_qr_cpu_test() {
    vector<vector<double> > A = {
            {1.0, -1.0, 4.0},
            {1.0, 4.0,  -2.0},
            {1.0, 4.0,  2.0},
            {1.0, -1.0, 0.0}
    };

    vector<vector<vector<double>>> QR = householderQR(A);
    vector<vector<double>> Q = QR[0];
    vector<vector<double>> R = QR[1];

    cout << "Q: " << endl;
    for (int i = 0; i < Q.size(); i++) {
        for (int j = 0; j < Q[0].size(); j++) {
            cout << setw(12) << setfill(' ') << fixed << setprecision(6) << Q[i][j];
        }
        cout << endl;
    }

    cout << "R:" << endl;
    for (int i = 0; i < R.size(); i++) {
        for (int j = 0; j < R[0].size(); j++)
            cout << setw(12) << setfill(' ') << fixed << setprecision(6) << R[i][j];
        cout << endl;
    }

    vector<vector<double>> transpose_R = transpose(R);

    vector<vector<double>> RTR = product(transpose_R, R);

    cout << "RTR:" << endl;
    for (int i = 0; i < RTR.size(); i++) {
        for (int j = 0; j < RTR[0].size(); j++)
            cout << setw(12) << setfill(' ') << fixed << setprecision(6) << RTR[i][j];
        cout << endl;
    }

    return 0;
}

//__device__ void householderQR_kernel(double *A, int m, int n) {
//    for (int k = 0; k < n; k++) {
//        int e_size = m - k;
//        double x[e_size];
//        double e[e_size];
//        double u[e_size];
//        double v[e_size];
//        double H[e_size][e_size];
//        double temp[e_size][n - k];
//        double x_norm, u_norm;
//        for (int i = k; i < m; i++) x[i - k] = A[i * n + k];
//        e[0] = 1;
//        for (int i = 0; i < e_size; i++) x_norm += x[i] * x[i];
//        x_norm = sqrt(x_norm);
//        double sign_x = x[0] >= 0 ? 1 : -1;
//        for (int i = 0; i < e_size; i++) u[i] = sign_x * x_norm * e[i] + x[i];
//        for (int i = 0; i < e_size; i++) u_norm += u[i] * u[i];
//        u_norm = sqrt(u_norm);
//        for (int i = 0; i < e_size; i++) v[i] = u[i] / u_norm;
//
//
//        for (int i = 0; i < e_size; i++) {
//            for (int j = 0; j < e_size; j++) {
//                H[i][j] = (i == j) ? 1 : 0;
//                H[i][j] -= 2 * v[i] * v[j];
//            }
//        }
//
//
//        for (int i = 0; i < e_size; i++) {
//            for (int j = 0; j < n - k; j++) {
//                for (int l = 0; l < e_size; l++) {
//                    temp[i][j] += H[i][l] * A[(k + l) * n + (k + j)];
//                }
//            }
//        }
//
//        for (int i = 0; i < e_size; i++) {
//            for (int j = 0; j < n - k; j++) {
//                A[(k + i) * n + (k + j)] = temp[i][j];
//            }
//        }
//    }
//}
__device__ void print_vector(double *vec, int size, const char *name) {
  printf("%s:\n", name);
  for (int i = 0; i < size; i++) {
    printf("  vec[%d] = %f\n", i, vec[i]);
  }
}

__device__ void print_matrix(double *mat, int rows, int cols, const char *name) {
  printf("%s:\n", name);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("  mat[%d][%d] = %f\n", i, j, mat[i * cols + j]);
    }
  }
}

__device__ void
householderQR_device(double *A, int m, int n, double *x, double *e, double *u, double *v, double *H, double *temp) {
    for (int k = 0; k < n; k++) {
        printf("k = %d\n", k);
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


        print_vector(v, e_size, "v");

        for (int i = 0; i < e_size; i++) {
            for (int j = 0; j < e_size; j++) {
                H[i * e_size + j] = (i == j) ? 1 : 0;
                H[i * e_size + j] -= 2 * v[i] * v[j];
            }
        }

        print_matrix(H, e_size, e_size, "H");

        for (int i = 0; i < e_size; i++) {
            for (int j = 0; j < n - k; j++) {
                for (int l = 0; l < e_size; l++) {
                    temp[i * (n - k) + j] += H[i * e_size + l] * A[(k + l) * n + (k + j)];   //problem here, step 2, H is right, A should be right (indicing is wrong?)
                }
            }
        }

        for (int i = 0; i < e_size; i++) {
            for (int j = 0; j < n - k; j++) {
                A[(k + i) * n + (k + j)] = temp[i * (n - k) + j];
            }
        }
        print_matrix(temp, e_size, n - k, "temp");
    }
}

__global__ void
householderQR_global(double *A, int m, int n, double *x, double *e, double *u, double *v, double *H, double *temp) {
    householderQR_device(A, m, n, x, e, u, v, H, temp);
}

int JTQuant::TestSuites::householder_qr_gpu_test() {
    int m = 4; // number of rows in the matrix A
    int n = 3; // number of columns in the matrix A
    vector<vector<double>> A_h = {
            {1.0, -1.0, 4.0},
            {1.0, 4.0,  -2.0},
            {1.0, 4.0,  2.0},
            {1.0, -1.0, 0.0}
    };

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
    householderQR_global<<<grid, block>>>(A_d, m, n, x, e, u, v, H, temp);
    cudaDeviceSynchronize();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << A_d[i * n + j] << " ";
        }
        cout << endl;
    }
    cudaFree(x);
    cudaFree(e);
    cudaFree(u);
    cudaFree(v);
    cudaFree(H);
    cudaFree(temp);
    return 0;
}




