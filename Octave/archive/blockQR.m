function [Q, R] = blockQR(A)
    [n,m] = size(A);
    assert(n == m, 'Matrix must be square');
    mid = floor(n/2);
    % divide A into four submatrices
    A11 = A(1:mid, 1:mid);
    A12 = A(1:mid, mid+1:n);
    A21 = A(mid+1:n, 1:mid);
    A22 = A(mid+1:n, mid+1:n);

    % recursively decompose submatrices
    [Q1, R1] = qr(A11);
    [Q2, R2] = qr(A22);

    % compute new submatrices
    B21 = Q1' * A21;
    B12 = A12 * Q2';

    % combine results to get final Q, R
    Q = [Q1, Q2];
    R = [R1, B12; B21, R2];
end
