A = rand(6,6); % generates a random 6x6 matrix
[blockQ, blockR] = blockQR(A);
[Q,R] = qr(A);

% Check that Q is orthogonal
assert(norm(blockQ'*blockQ - eye(6)) < 1e-10, 'blockQ is not orthogonal')

% Check that Q*R is approximately equal to A
assert(norm(blockQ*blockR - A) < 1e-10, 'blockQ*blockR is not equal to A')

% Check that Q*R is approximately equal to A
assert(norm(blockR - R) < 1e-10, 'blockR is not equal to R')

% Check that Q*R is approximately equal to A
assert(norm(blockQ - Q) < 1e-10, 'blockQ is not equal to Q')