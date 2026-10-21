function pn_seq = lfsr_framesync(Nbits)
% LFSR_FRAMESYNC  Generate PN sequence for preamble & training
%
%   pn_seq = lfsr_framesync(Nbits)
%
%   Uses an order-11 LFSR with primitive polynomial:
%       p(x) = x^11 + x^2 + 1
%   → period = 2^11 - 1 = 2047
%
%   Output:
%     pn_seq : column vector of 0/1 bits (length Nbits)

    if Nbits <= 0
        pn_seq = [];
        return;
    end

    % ---------------------------------------------------------------------
    % Order-11 LFSR: state = [s1 s2 ... s11]^T, s1 = newest, s11 = oldest
    % Feedback taps at positions 11 and 2 (x^11 + x^2 + 1)
    % new_bit = s11 XOR s2
    % ---------------------------------------------------------------------
    n = 11;

    % Initial non-zero state (any non-zero works for an m-sequence)
    state = ones(n,1);   % [1 1 1 1 1 1 1 1 1 1 1]^T

    pn_seq = zeros(Nbits,1);

    for k = 1:Nbits
        % Output bit – choose one position and keep it consistent
        % Here we output the oldest bit s11 (state(end))
        pn_seq(k) = state(end);

        % Compute feedback bit (mod-2 sum of taps 11 and 2)
        feedback = xor(state(end), state(2));

        % Shift register: insert new bit at the "front"
        state = [feedback; state(1:end-1)];
    end
end
