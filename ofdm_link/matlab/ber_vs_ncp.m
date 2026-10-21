%{
SCRIPT: Sweep cyclic-prefix (CP) length and measure BER over the audio OFDM link.

This script evaluates how the OFDM cyclic prefix length affects performance in the real
loudspeaker–microphone channel. It sweeps a list of CP lengths (cp_list), and for each CP runs
multiple independent trials (Ntrials) where a new random bit payload is transmitted and received.
For each CP value, it computes the mean BER across trials, then plots BER vs. CP length on a
log-scale and saves the figure.

Outputs:
  - Printed per-trial BER and mean BER for each CP length
  - ber_vs_cp.png / ber_vs_cp.fig : BER vs CP-length plot
%}

clear 
close all
clc;

% 0) Configuration Values

conf.audiosystem = 'audio'; 

% General parameters 
conf.nbits   = 512*2*10;  % number of bits 
conf.f_c     = 8000;

% Channel parameter
conf.channel_tracking = 1; % 1 if we want tracking
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
conf.ofdm.cplen     = 256;    % will be overwritten in sweep
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

% 1) CP-length sweep configuration
cp_list = [64 128 192 256 320];  
n_cp    = numel(cp_list);

Ntrials = 3;                      
ber_cp  = zeros(n_cp,1);

% 2) Sweep over CP lengths
for icp = 1:n_cp
    conf.ofdm.cplen = cp_list(icp);
    fprintf('\n=== Testing CP = %d samples ===\n', conf.ofdm.cplen);

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

    ber_cp(icp) = mean(ber_trials);
    fprintf('=> Mean BER for CP = %d : %.4f\n', conf.ofdm.cplen, ber_cp(icp));
end

% 3) Plot BER vs CP length
figure;
semilogy(cp_list, ber_cp, '-o', 'LineWidth', 1.5, 'MarkerSize', 8);
grid on;
xlabel('CP length N_{CP} [samples]');
ylabel('BER');
title('BER vs CP length for audio channel');

saveas(gcf, 'ber_vs_cp.fig');               
print(gcf, 'ber_vs_cp.png', '-dpng', '-r300'); 
