clear
clc
##diary testQR
##diary on

pkg load symbolic
syms s1 s2 s3 s4 s5 s6 x1 x2 x3 x4 x5 x6

[Q,
assume(s1, 'real')
assume(s2, 'real')
assume(s3, 'real')
assume(s4, 'real')
assume(s5, 'real')
assume(s6, 'real')
assume(x1, 'real')
assume(x2, 'real')
assume(x3, 'real')
assume(x4, 'real')
assume(x5, 'real')
assume(x6, 'real')

A = [
1, s1, s1*s1, x1, x1*x1, x1*s1;
1, s2, s2*s2, x2, x2*x2, x2*s2;
1, s3, s3*s3, x3, x3*x3, x3*s3;
1, s4, s4*s4, x4, x4*x4, x4*s4;
1, s5, s5*s5, x5, x5*x5, x5*s5;
1, s6, s6*s6, x6, x6*x6, x6*s6;
]

##A11 = A(1:3, 1:3)
##A12 = A(1:3, 4:6)
##A21 = A(4:6, 1:3)
##A22 = A(4:6, 4:6)
##
##% recursively decompose submatrices
##[Q1, R1] = qr(A11);
##[Q2, R2] = qr(A22);
##
##% compute new submatrices
##B21 = Q1' * A21;
##B12 = A12 * Q2';
##
##% combine results to get final Q, R
##Q = [Q1, Q2];
##R = [R1, B12; B21, R2];
##
##R
##
##diary off