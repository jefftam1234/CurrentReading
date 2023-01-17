function [Q, R] = blockQR(A)
    [n,m] = size(A);
    if n <= m
        [Q, R] = qr(A);
        return;
    else
        % divide A into four submatrices
        A11 = A(1:m, 1:m);
        A12 = A(1:m, m+1:n);
        A21 = A(m+1:n, 1:m);
        A22 = A(m+1:n, m+1:n);

        % recursively decompose submatrices
        [Q1, R1] = blockQR(A11);
        [Q2, R2] = blockQR(A22);

        % compute new submatrices
        B21 = Q1' * A21;
        B12 = A12 * Q2';

        % combine results to get final Q, R
        Q = [Q1, Q2];
        R = [R1, B12; B21, R2];
    end
end