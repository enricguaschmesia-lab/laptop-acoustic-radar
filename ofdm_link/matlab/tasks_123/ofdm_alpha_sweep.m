%% ofdm_alpha_dd_sweep.m
% Sweep alpha_dd in [0,1] for DD channel tracking (Joelle scheme)
% and measure BER for a chosen emulator channel.

clear; close all; clc;

%% 1) CONFIGURATION (copied/adapted from audiotrans.m)

conf.audiosystem = 'emulator';

% ---- SELECT CHANNEL AND SNR HERE ----
conf.emulator_idx = 2;      % choose 1..5 (e.g. 2 = oscillator mismatch)
conf.emulator_snr = 100;     % dB (change as desired)
% ------------------------------------

% General parameters 
conf.nbits   = 512*2*50;
conf.f_c     = 8000;

% Preamble
conf.sc.f_sym = 1000;  % symbol rate
conf.sc.nsyms = 500;

% OFDM
conf.ofdm.bandwidth   = 2000; 
conf.ofdm.ncarrier    = 512;
conf.ofdm.cplen       = 256;
conf.modulation_order = 2; % QPSK
conf.constellation    = [1+1j -1+1j 1-1j -1-1j] * 1/sqrt(2);
conf.ofdm.nsym        = conf.nbits / (conf.ofdm.ncarrier * conf.modulation_order);

% Phase tracking configuration: DD tracking (Joelle)
conf.phase.tracking   = true;   % not used in avg==0 path but kept for clarity
conf.phase.avg        = 1;      % 0 → per-subcarrier DD tracking
conf.phase.alpha_dd   = 0.95;   % will be overwritten in the loop
conf.phase.alpha_cpe  = 0.98;
conf.phase.lpf_factor = 0.98;

% Audio settings 
conf.f_s    = 48000;
conf.bitsps = 16; 

% Pre-calculations
conf.ofdm.spacing   = conf.ofdm.bandwidth/conf.ofdm.ncarrier;
conf.sc.os_factor   = conf.f_s/conf.sc.f_sym;
conf.ofdm.os_factor = conf.f_s/(conf.ofdm.ncarrier*conf.ofdm.spacing);

% Pre-generate Preamble Pulse
conf.sc.txpulse_length = 20*conf.sc.os_factor;
conf.sc.txpulse        = rrc(conf.sc.os_factor, 0.22, conf.sc.txpulse_length);

%% 2) Define alpha sweep

alpha_vec = 0:0.04:1;   % sweep α from 0 to 1 in steps of 0.01
Nalpha    = numel(alpha_vec);

ber_vec   = zeros(Nalpha, 1);

rng(1);   % reproducible bits

%% 3) Loop over alpha values

for ia = 1:Nalpha
    conf.phase.alpha_cpe = 1- alpha_vec(ia);

    fprintf('Testing alpha_dd = %.2f\n', conf.phase.alpha_cpe);

    % --- Generate data bits (fresh frame for each alpha) ---
    txbits = randi([0 1], conf.nbits, 1);

    % --- Transmit (as in audiotrans.m) ---
    [txsignal, conf] = txofdm(txbits, conf);

    rawtxsignal = [ zeros(conf.f_s,1) ; txsignal ; zeros(conf.f_s,1) ];
    rawtxsignal = [ rawtxsignal zeros(size(rawtxsignal)) ];

    % --- Channel (emulator) ---
    rxsignal = channel_emulator(rawtxsignal(:,1), conf);

    % --- Receive with DD tracking using current alpha_dd ---
    [rxbits, conf, ~] = rxofdm(rxsignal, conf);

    % --- BER ---
    biterrors    = sum(rxbits ~= txbits);
    ber_vec(ia)  = biterrors / length(rxbits);
end

%% 4) Plot alpha vs BER

figure;
plot(alpha_vec, ber_vec, 'LineWidth', 1.5);
grid on;
xlabel('\alpha_{CPE}');
ylabel('Bit Error Rate (BER)');
title(sprintf('CPE tracking: BER vs \\alpha_{CPE} (Chan ID %d, SNR = %d dB)', ...
      conf.emulator_idx, conf.emulator_snr));

hold off;
