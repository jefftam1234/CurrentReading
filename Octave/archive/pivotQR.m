clear;
clc;

% Define matrix A
A = [1 2 3 4; 4 5 6 7; 7 8 9 10; 16 28 19 10];

% Define block size
b = 2;

% Initialize Q and R matrices
Q = eye(size(A));
R = A;

% Loop through the columns of A in blocks of size b
for j = 1:b:size(A, 2)
    % Get the current block of A
    A_block = A(:, j:j+b-1);
    
    % Find the pivot column in the current block
    [~, pivot_col] = max(abs(A_block(j:end, :)));
    pivot_col = pivot_col + j - 1;
    
    % Swap the pivot column with the first column of the block
    A_block([j pivot_col], :) = A_block([pivot_col j], :);
    R([j pivot_col], :) = R([pivot_col j], :);
    Q(:, [j pivot_col]) = Q(:, [pivot_col j]);
    
    % Perform QR decomposition on the current block
    [Q_block, R_block] = qr(A_block);
    
    % Update Q and R matrices
    Q(:, j:j+b-1) = Q(:, j:j+b-1) * Q_block;
    R(j:j+b-1, j:j+b-1) = R_block;
    
    % Update A
    A(:, j:j+b-1) = Q_block' * A(:, j:j+b-1);
end

% Display Q, R, and A matrices
disp('Q:');
disp(Q);
disp('R:');
disp(R);
disp('A:');
disp(Q*R);
