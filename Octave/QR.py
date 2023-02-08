import numpy as np
import copy
from scipy.linalg import qr


def generate_array(m, n, range=10):
    np.random.seed(0)
    arr = 2 * range * np.random.rand(m, n) - range
    return arr


def print_matlab_format(arr):
    matlab_format = ";\n".join([",".join([str(val) for val in row]) for row in arr])
    print("Matlab format:")
    print(f"arr = [{matlab_format}];")
    print("[mQ, mR] = householderQR(arr)\n")
    print("[Q,R] = qr(arr)\n")


def print_cpp_format(arr):
    cpp_format = ",\n".join(["{{{}}}".format(",".join([str(val) for val in row])) for row in arr])
    print(f"std::vector<std::vector<double>> arr = {{\n{cpp_format}\n}};")
    print("\n")


def print_matrix(matrix):
    for row in matrix:
        print(" ".join(["{:.4f}".format(x) for x in row]))

def print_vector(vector):
    print(" ".join(["{:.4f}".format(x) for x in vector]))


def print_qr(q, r):
    if q.shape[0] > 8:
        print("Q matrix is too big to print.")
    else:
        print("Q = \n")
        print_matrix(q)
    print("\nR = \n")
    filtered_r = [r[i] for i in range(len(r)) if i < len(r[0])]
    print_matrix(filtered_r)


def householderQR(A, mode="default"):
    m, n = A.shape
    Q = np.eye(m)
    R = copy.deepcopy(A)

    for k in range(n):
        x = R[k:, k]
        e = np.zeros(m - k)
        e[0] = 1
        u = np.sign(x[0]) * np.linalg.norm(x) * e + x
        v = u / np.linalg.norm(u)
        H = np.eye(e.size)-2*np.outer(v,v)
        # R[k:, k:] = R[k:, k:] - 2 * np.outer(u, np.dot(u, R[k:, k:]))
        R[k:, k:] = np.dot(H, R[k:, k:])
        if mode.lower() != 'r':  #only need for Q
            H_p = np.eye(Q.shape[0])
            H_p[-H.shape[0]:, -H.shape[0]:] = H
            Q = np.dot(Q, H_p)

    if mode.lower() != 'r':
        return Q, R
    else:
        return R
def householderQR2(A):
    m, n = A.shape
    A_mod = copy.deepcopy(A)

    for k in range(n):
        x = A_mod[k:, k]                                #copy vector from k-th column of input matrix A
        e = np.zeros(m - k)
        e[0] = 1
        u = np.sign(x[0]) * np.linalg.norm(x) * e + x   # create householder reflector vector u
        v = u / np.linalg.norm(u)                       # normalize vector u
        H = np.eye(e.size)-2*np.outer(v,v)              # create householder matrix H by outer product
        A_mod[k:, k:] = np.dot(H, A_mod[k:, k:])        # reflect the matrix's so column k will have upper triangular element

    R = A_mod[:n, :n]
    return R

def householder_test():
    # m = 20
    # n = 20
    # arr = generate_array(m, n)
    # arr = np.array( [[2, -2, 18], [2,1,0], [1,2,0]], dtype=np.float64)
    arr = np.array([
        [1, -1, 4],
        [1, 4, -2],
        [1, 4, 2],
        [1, -1, 0],
    ], dtype=np.float64
    )

    print_matlab_format(arr)
    print_cpp_format(arr)
    print("\nCustomized QR result:\n")
    q, r = householderQR(arr)
    rtr = np.dot(r.T, r)
    print_qr(q, r)
    print("\nCustomized QR result 2:\n")
    r2 = householderQR2(arr)
    rtr2 = np.dot(r2.T, r2)
    print_matrix(r2)
    print("\nScipy QR result:\n")
    Q, R = np.linalg.qr(arr)
    RTR = np.dot(R.T, R)
    print_qr(Q, R)
    qr_custom = np.dot(q, r)
    qr_scipy = np.dot(Q, R)
    print("\n Reproduce of the original matrix")
    print("custom:")
    print_matrix(qr_custom)
    print("scipy:")
    print_matrix(qr_scipy)
    print("\ndifference of reproduced(custom) - original")
    print_matrix(qr_custom - arr)
    print("\ndifference of reproduced(scipy) - original")
    print_matrix(qr_scipy - arr)
    print("\ndifference of R'*R of custom vs scipy")
    print_matrix(rtr - RTR)
    print("\ndifference of R'*R of custom2 vs scipy")
    print_matrix(rtr2 - RTR)

def test_vector_util():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = a + b
    f = np.outer(a, b)
    g = np.inner(f, c)
    print("\nouter product:")
    print_matrix(f)
    print("\nmatrix-vector product:")
    print_vector(g)
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[1, 2], [3, 4], [5, 6]])
    C = np.dot(A, B)
    print("Product of A and B:")
    print_matrix(C)


if __name__ == "__main__":
    householder_test()
    #test_vector_util()
