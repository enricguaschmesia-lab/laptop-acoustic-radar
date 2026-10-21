%{
SCRIPT: Compare OFDM baseband PAPR for an image bitstream (baseline vs interleaving vs whitening).

This script loads “orig.png”, converts its pixels into a serialized bitstream, and computes the
PAPR (in dB) of the corresponding QPSK-OFDM baseband waveform (before carrier upconversion) under
three payload processing strategies:
  1) baseline mapping (no processing),
  2) subcarrier interleaving (one fixed random permutation of subcarriers),
  3) bit whitening via reproducible XOR scrambling.

It prints the three PAPR values to the console. Helper functions implement the QPSK mapping to the
frequency-domain grid, OFDM modulation (IFFT + CP + resampling + normalization), and PAPR
computation.

Input:
  - orig.png

Outputs:
  - Printed PAPR values (dB): baseline, interleaved, whitened
%}

clear
close all
clc

img = imread("orig.png");
s = size(img);
bitstream = de2bi(img(:), 'left-msb');
bitstream = bitstream(:);

%OFDM
conf.ofdm.bandwidth = 2000; %f_s/bw should be an integer for resampling
conf.ofdm.ncarrier  = 512;
conf.ofdm.cplen     = 256;
conf.modulation_order = 2; % 2 for QPSK
conf.constellation = 1/sqrt(2) * [(-1-1j) (-1+1j) ( 1-1j) ( 1+1j)];
% Fix audio settings 
conf.f_s = 48000; % audio sampling rate, fixed by audiocard
conf.bitsps = 16;   % bits per audio sample
% all calculations that you only have to do once
conf.ofdm.spacing  = conf.ofdm.bandwidth/conf.ofdm.ncarrier;
conf.ofdm.os_factor = conf.f_s/(conf.ofdm.ncarrier*conf.ofdm.spacing);

[papr_base_dB, papr_int_dB, papr_wht_dB] = ...
    compute_papr_image_sequence(bitstream, conf);

fprintf('Baseline   PAPR: %.4f dB\n', papr_base_dB);
fprintf('Interleaving    PAPR: %.4f dB\n', papr_int_dB);
fprintf('Whitening    PAPR: %.4f dB\n', papr_wht_dB);


function [PAPR_dB, ofdm_os] = papr_from_subcarriers(A, conf)
% PAPR_FROM_SUBCARRIERS
%   - IFFT across subcarriers
%   - CP insertion
%   - ofdm_tx_resampling(...)
%   - normalisation to unit average power
%   - PAPR on ofdm_os (baseband, before upconversion)

    Ncar      = conf.ofdm.ncarrier;
    Ncp       = conf.ofdm.cplen;
    Nofdm_sym = size(A, 2);

    ofdm_td = ifft(A, Ncar, 1); 

    sym_len       = Ncar + Ncp;
    ofdm_baseband = zeros(sym_len * Nofdm_sym, 1);

    for n = 1:Nofdm_sym
        sym_n    = ofdm_td(:,n);               % 512 x 1
        cp_n     = sym_n(end-Ncp+1:end);       % 256 x 1
        sym_cp_n = [cp_n; sym_n];              % 768 x 1

        idx_start = (n-1)*sym_len + 1;
        idx_end   = n*sym_len;
        ofdm_baseband(idx_start:idx_end) = sym_cp_n;
    end

    ofdm_os = ofdm_tx_resampling(ofdm_baseband, conf); 

    ofdm_os = ofdm_os / sqrt(mean(abs(ofdm_os).^2));
    power   = abs(ofdm_os).^2;
    P_peak  = max(power);
    P_avg   = mean(power);
    PAPR_dB = 10*log10(P_peak / P_avg);
end

function [papr_base_dB, papr_int_dB, papr_wht_dB] = ...
         compute_papr_image_sequence(bitstream, conf)
% COMPUTE_PAPR_IMAGE_SEQUENCE
%   Outputs PAPR (in dB) for:
%     - baseline (no processing),
%     - interleaving across subcarriers (single permutation),
%     - whitening (bit scrambling).


    Ncar = conf.ofdm.ncarrier;
    const = conf.constellation(:);  

    bits = bitstream(:);
    bits_per_ofdm = 2 * Ncar;      

    Nsym = ceil(numel(bits) / bits_per_ofdm);
    Ntot = Nsym * bits_per_ofdm;
    if Ntot > numel(bits)
        bits = [bits; zeros(Ntot - numel(bits), 1)];
    end

    bits_mat = reshape(bits, bits_per_ofdm, Nsym);

    % 1) BASELINE
    A_base = bits_to_subcarrier_grid(bits_mat, const, Ncar);
    [papr_base_dB, ~] = papr_from_subcarriers(A_base, conf);

    % 2) INTERLEAVING: permute subcarrier indices once
    perm = randperm(Ncar);
    A_int = A_base(perm, :);  % permute rows
    [papr_int_dB, ~] = papr_from_subcarriers(A_int, conf);

    % 3) WHITENING: scramble bits, then same mapping
    [bits_wh, ~] = XORING(bits, [], 0);          % seed = 0
    bits_wh_mat = reshape(bits_wh, bits_per_ofdm, Nsym);
    A_wh = bits_to_subcarrier_grid(bits_wh_mat, const, Ncar);
    [papr_wht_dB, ~] = papr_from_subcarriers(A_wh, conf);

end

function A = bits_to_subcarrier_grid(bits_mat, const, Ncar)
% Map bits (2 per symbol) to QPSK using SAME Gray mapping as in txofdm,
% then arrange into [Ncar x Nsym] frequency-domain grid.

    [bits_per_ofdm, Nsym] = size(bits_mat);
    if bits_per_ofdm ~= 2*Ncar
        error('bits_to_subcarrier_grid: bits_per_ofdm must be 2*Ncar.');
    end
    ofdm_bits_reshaped = reshape(bits_mat, 2, []).';  
    idx = bi2de(ofdm_bits_reshaped, 'left-msb') + 1;
    mappedGray = const(idx).';          
    A = reshape(mappedGray, Ncar, Nsym);
end

