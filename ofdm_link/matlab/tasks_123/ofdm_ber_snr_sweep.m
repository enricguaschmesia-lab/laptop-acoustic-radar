%% OFDM BER vs SNR Sweep (Channel ID 1)

clear; close all; clc;

%% 1) COPY THE CONFIGURATION BLOCK FROM audiotrans.m

conf.audiosystem = 'emulator';

% Emulator configuration
conf.emulator_idx = 1;        % simple AWGN-like channel
conf.emulator_snr = 100;      % will be overwritten in the loop

% General parameters 
conf.nbits   = 512*2*50;
conf.f_c     = 8000;

% Preamble
conf.sc.f_sym = 1000;  % symbol rate
conf.sc.nsyms = 500;

% OFDM
conf.ofdm.bandwidth = 2000; 
conf.ofdm.ncarrier  = 512;
conf.ofdm.cplen     = 256;
conf.modulation_order = 2; % QPSK
conf.constellation = [1+1j -1+1j 1-1j -1-1j] * 1/sqrt(2);
conf.ofdm.nsym = conf.nbits / (conf.ofdm.ncarrier * conf.modulation_order);

% Phase tracking configuration (default, as in audiotrans.m)
conf.phase.tracking   = false;
conf.phase.avg        = 0;      
conf.phase.alpha_dd   = 0.95;   
conf.phase.alpha_cpe  = 0.98;   
conf.phase.lpf_factor = 0.98;   

% Audio settings 
conf.f_s = 48000;
conf.bitsps = 16; 

% Pre-calculations
conf.ofdm.spacing   = conf.ofdm.bandwidth/conf.ofdm.ncarrier;
conf.sc.os_factor   = conf.f_s/conf.sc.f_sym;
conf.ofdm.os_factor = conf.f_s/(conf.ofdm.ncarrier*conf.ofdm.spacing);

% Pre-generate preamble pulse
conf.sc.txpulse_length = 20*conf.sc.os_factor;
conf.sc.txpulse        = rrc(conf.sc.os_factor, 0.22, conf.sc.txpulse_length);

%% 2) Define SNR range and number of Monte-Carlo trials

snr_vec   = -10:2:20;      % in dB, adjust as you like
Nsnr      = numel(snr_vec);
Ntrials   = 10;          % frames per SNR for averaging

ber_mat   = zeros(Nsnr, Ntrials);

rng(1);   % reproducible random bits

%% 3) Loop over SNR and trials

for iS = 1:Nsnr
    conf.emulator_snr = snr_vec(iS);

    fprintf('SNR = %2d dB\n', snr_vec(iS));

    for it = 1:Ntrials

        % --- Generate data bits
        txbits = randi([0 1], conf.nbits, 1);

        % --- Transmit (exactly as in audiotrans.m)
        [txsignal, conf] = txofdm(txbits, conf);

        rawtxsignal = [ zeros(conf.f_s,1) ; txsignal ; zeros(conf.f_s,1) ];
        rawtxsignal = [ rawtxsignal zeros(size(rawtxsignal)) ];

        % Emulated channel
        rxsignal = channel_emulator(rawtxsignal(:,1), conf);

        % --- Receive
        [rxbits, conf, ~] = rxofdm(rxsignal, conf);

        % --- BER for this realization
        biterrors = sum(rxbits ~= txbits);
        ber_mat(iS, it) = biterrors / length(rxbits);
    end
end

%% 4) Average BER for each SNR and plot

ber_mean = mean(ber_mat, 2);

figure;
semilogy(snr_vec, ber_mean, '-o','LineWidth',1.3);
grid on;
xlabel('SNR [dB]');
ylabel('Bit Error Rate (BER)');
title('OFDM system: BER vs SNR, Channel ID 1');

% If you want to know the SNR threshold for BER < 1%:
idx_ok = find(ber_mean < 0.01, 1, 'first');
if ~isempty(idx_ok)
    fprintf('First SNR with BER < 1%%: %.1f dB\n', snr_vec(idx_ok));
else
    fprintf('No SNR in the tested range achieved BER < 1%%.\n');
end
