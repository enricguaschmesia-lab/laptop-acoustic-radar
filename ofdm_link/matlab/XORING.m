function [x_out, xor_seq] = XORING(x_in, xor_seq, seed)
%{
XORING  Bit-wise XOR whitening / de-whitening helper.

[x_out, xor_seq] = XORING(x_in, xor_seq, seed) takes a binary input vector x_in (0/1),
and XORs it with a binary sequence xor_seq. If xor_seq is not provided (or is empty),
the function generates a random binary sequence of the same length; if a seed is given,
the generated sequence is reproducible.

The same function is used for whitening and de-whitening: applying XORING twice with the
same xor_seq returns the original bits.

Inputs:
  - x_in    : binary vector (0/1), any shape (will be vectorized)
  - xor_seq : (optional) binary vector used for XOR; if length differs, it is repeated/truncated
  - seed    : (optional) RNG seed for reproducible sequence generation (used only when xor_seq empty)

Outputs:
  - x_out   : XORed output bits as a double vector (0/1)
  - xor_seq : the XOR sequence actually used (generated or resized)
%}

    x = x_in(:);

    if nargin < 2 || isempty(xor_seq)
        if nargin >= 3 && ~isempty(seed)
            old = rng;                 % preserve RNG state
            rng(seed, 'twister');
            xor_seq = randi([0 1], numel(x), 1);
            rng(old);
        else
            xor_seq = randi([0 1], numel(x), 1);
        end
    else
        % Make sure xor_seq matches length(x): repeat/truncate as needed
        xor_seq = xor_seq(:);
        if numel(xor_seq) ~= numel(x)
            xor_seq = repmat(xor_seq, ceil(numel(x)/numel(xor_seq)), 1);
            xor_seq = xor_seq(1:numel(x));
        end
    end

    x_out = double(xor(logical(x), logical(xor_seq)));
end

