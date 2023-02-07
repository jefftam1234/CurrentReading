function [Q,R] = householderQR(A)

[m,n] = size(A);
Q = eye(m);
R = A;

for k = 1:n
    x = R(k:m,k);
    e = zeros(m-k+1,1);
    e(1) = 1;
    u = sign(x(1)) * norm(x) * e + x;
    u = u / norm(u);
    R(k:m,k:n) = R(k:m,k:n) - 2 * u * (u' * R(k:m,k:n));
    Q(:,k:m) = Q(:,k:m) - 2 * (Q(:,k:m) * u) * u';
end

end