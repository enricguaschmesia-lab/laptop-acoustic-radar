%{
SCRIPT: Sweep pilot-training interval and evaluate BER vs spectral efficiency (audio OFDM link).

This script studies the trade-off between channel tracking overhead and performance in the real
loudspeaker–microphone channel. It sweeps the pilot interval K (one training OFDM symbol inserted
every K data symbols), runs multiple trials per K, and measures the mean BER for each setting
(using the periodic-training TX/RX pair txofdm1/rxofdm1).

After collecting BER results, it computes the corresponding spectral efficiency η (bits/s/Hz)
accounting for: (i) training-symbol overhead, (ii) cyclic prefix overhead, and (iii) QPSK bits per
subcarrier. Finally, it plots BER versus η on a log-scale and saves the figure.

Outputs:
  - Printed per-trial BER and mean BER for each pilot interval K
  - ber_vs_eta.png / ber_vs_eta.fig : BER vs spectral efficiency plot
%}


clear 
close all
clc

% 0) Configuration Values
conf.audiosystem = 'audio'; 

% General parameters 
conf.nbits   = 512*2*20;  % number of bits 
conf.f_c     = 8000;

% Channel parameter
conf.channel_tracking = 1;  % 1 if we want tracking

% Channel/phase tracking: (only works if conf.channel_tracking is 1)
%   0 → channel tracking (amplitude and phase)
%   1 → CPE-only Viterbi–Viterbi (global)
%   2 → per-subcarrier Viterbi–Viterbi
conf.phase.avg        = 0; 

% Filter / smoothing factors (to be tuned per channel)
conf.phase.alpha_dd   = 0.95;   % smoothing for channel tracking 
conf.phase.alpha_cpe  = 0.98;   % smoothing for global CPE tracker
conf.phase.lpf_factor = 0.98;   % smoothing for per-subcarrier VV (avg == 2)

% Preamble (single-carrier)
conf.sc.f_sym = 1000;     % symbol rate
conf.sc.nsyms = 500;      % number of preamble symbols

% OFDM parameters
conf.ofdm.bandwidth = 2000;   % Hz (OFDM baseband sampling rate)
conf.ofdm.ncarrier  = 512;
conf.ofdm.cplen     = 256;    % fixed CP for this experiment
conf.modulation_order = 2;    % 2 for QPSK
conf.constellation = 1/sqrt(2) * [(-1-1j) (-1+1j) ( 1-1j) ( 1+1j)];

% Audio settings 
conf.f_s    = 48000;      % audio sampling rate, fixed by sound card
conf.bitsps = 16;         % bits per audio sample

% Derived parameters
conf.ofdm.spacing  = conf.ofdm.bandwidth/conf.ofdm.ncarrier;
conf.sc.os_factor  = conf.f_s/conf.sc.f_sym;

if mod(conf.sc.os_factor,1) ~= 0
   disp('WARNING: Sampling rate must be a multiple of the symbol rate for single-carrier system'); 
end

conf.ofdm.os_factor = conf.f_s/(conf.ofdm.ncarrier*conf.ofdm.spacing);

% Pregenerate useful data
conf.sc.txpulse_length = 20*conf.sc.os_factor;
conf.sc.txpulse        = rrc(conf.sc.os_factor,0.22,conf.sc.txpulse_length);

% 1) Pilot-interval sweep configuration
% pilot_interval_list = K values, e.g. one pilot every K data symbols
pilot_interval_list = [1 2 4 8 16]; 
n_pi = numel(pilot_interval_list);

Ntrials = 3;                        
ber_pi  = zeros(n_pi,1);

% 2) Sweep over pilot intervals
for ipi = 1:n_pi
    conf.ofdm.pilot_interval = pilot_interval_list(ipi);
    fprintf('\n=== Testing pilot interval K = %d ===\n', conf.ofdm.pilot_interval);

    ber_trials = zeros(Ntrials,1);

    for it = 1:Ntrials
        fprintf('  Trial %d / %d\n', it, Ntrials);

        %  TX 
        txbits = randi([0 1], conf.nbits, 1);
        [txsignal, conf] = txofdm1(txbits, conf);

        rawtxsignal = [ zeros(conf.f_s,1) ; txsignal ; zeros(conf.f_s,1) ];
        rawtxsignal = [ rawtxsignal  zeros(size(rawtxsignal)) ];

           
                % Play & record through sound card
                txdur = length(rawtxsignal)/conf.f_s; 
                audiowrite('out.wav', rawtxsignal, conf.f_s);

                playobj = audioplayer(rawtxsignal, conf.f_s, conf.bitsps);
                recobj  = audiorecorder(conf.f_s, conf.bitsps, 1);

                record(recobj);
                pause(2);
                disp('Recording...');
                playblocking(playobj);
                pause(2);
                stop(recobj);
                disp('Recording ended');

                rawrxsignal = getaudiodata(recobj,'int16');
                rawrxsignal = double(rawrxsignal) / double(intmax('int16'));
                rxsignal    = rawrxsignal;


        %  RX 
        [rxbits, conf, debug] = rxofdm1(rxsignal, conf); 

        biterrors      = sum(rxbits ~= txbits);
        ber_trials(it) = biterrors / length(rxbits);
        fprintf('    BER (trial %d): %.4f\n', it, ber_trials(it));
    end

    ber_pi(ipi) = mean(ber_trials);
    fprintf('=> Mean BER for pilot interval K = %d : %.4f\n', ...
             conf.ofdm.pilot_interval, ber_pi(ipi));
end

% 3) Compute  efficiency for each pilot interval K
M   = conf.modulation_order;
N   = conf.ofdm.ncarrier;
Ncp = conf.ofdm.cplen;

bits_per_data_sym = N * M;

Ns = ceil(conf.nbits / bits_per_data_sym);

eta_pi = zeros(n_pi,1);

for ipi = 1:n_pi
    K = pilot_interval_list(ipi);
    Nts_periodic = ceil(Ns / K);
    Nts_total    = 1 + Nts_periodic;
    eta_pi(ipi) = (Ns / (Ns + Nts_total)) * (N / (N + Ncp)) * M;
end

% 4) Plot BER vs spectral efficiency
figure;
semilogy(eta_pi, ber_pi, '-o', 'LineWidth', 1.5, 'MarkerSize', 8);
grid on;
xlabel('\eta  [bits/s/Hz]');
ylabel('BER');
title('BER vs efficiency');
saveas(gcf, 'ber_vs_eta.fig');               
print(gcf, 'ber_vs_eta.png', '-dpng', '-r300'); 
