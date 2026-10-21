%{
SCRIPT: Single-run audio OFDM transmission with channel/phase tracking and saved diagnostics.

This script generates a random payload, transmits it once over the audio OFDM link
(loudspeaker → microphone), decodes the received bits using rxofdm2 with the selected tracking
mode, and reports the BER. It also saves a standard set of receiver/channel diagnostic plots
(channel estimate, impulse response/PDP, channel evolution heatmaps, constellation/EVM, etc.)
to a specified results folder via save_audio_ofdm_plots().

Outputs:
  - out.wav (optional inspection/replay of the transmitted waveform)
  - Diagnostic figures saved under results/cool_audio_plots_X
  - BER printed in the console
%}

clear 
close all
clc

conf.audiosystem = 'audio'; 
conf.nbits   = 512*2*20;  % number of bits 
conf.f_c     = 8000;

%chanel parameter
conf.channel_tracking = 1 ; %1 if we want phase tracking,

% Channel/phase tracking: (only works if conf.channel_tracking is set to 1)
%   0 → channel tracking (amplitude and phase)
%   1 → CPE-only Viterbi–Viterbi (global)
%   2 → per-subcarrier Viterbi–Viterbi
conf.phase.avg        = 0; 

% Filter / smoothing factors (to be tuned per channel)
conf.phase.alpha_dd   = 0.95;   % smoothing for channel tracking 
conf.phase.alpha_cpe  = 0.98;   % smoothing for global CPE tracker
conf.phase.lpf_factor = 0.98;   % smoothing for per-subcarrier VV (avg == 2)

% Preamble
conf.sc.f_sym = 1000;  % symbol rate
conf.sc.nsyms = 500;    %number of preamble symbols

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
conf.sc.os_factor  = conf.f_s/conf.sc.f_sym;

if mod(conf.sc.os_factor,1) ~= 0
   disp('WARNING: Sampling rate must be a multiple of the symbol rate for single carrier system'); 
end

conf.ofdm.os_factor = conf.f_s/(conf.ofdm.ncarrier*conf.ofdm.spacing);

% Pregenerate useful data
conf.sc.txpulse_length = 20*conf.sc.os_factor;
conf.sc.txpulse    = rrc(conf.sc.os_factor,0.22,conf.sc.txpulse_length);


% Generate random data
txbits = randi([0 1],conf.nbits,1);

% Transmit Function
[txsignal, conf] = txofdm(txbits,conf);

rawtxsignal = [ zeros(conf.f_s,1) ; txsignal ; zeros(conf.f_s,1) ];
rawtxsignal = [  rawtxsignal  zeros(size(rawtxsignal)) ];

   

txdur = length(rawtxsignal)/conf.f_s; % calculate length of transmitted signal
audiowrite('out.wav',rawtxsignal,conf.f_s)
disp('MATLAB generic');
playobj = audioplayer(rawtxsignal,conf.f_s,conf.bitsps);
recobj  = audiorecorder(conf.f_s,conf.bitsps,1);
record(recobj);
pause(2);
disp('Recording...');
playblocking(playobj)
pause(2);
stop(recobj);
disp('Recording ended')
rawrxsignal  = getaudiodata(recobj,'int16');
rawrxsignal     = double(rawrxsignal(1:end))/double(intmax('int16')) ;
rxsignal = rawrxsignal; 

% Receive Function
[rxbits, conf, debug] = rxofdm2(rxsignal,conf);
out_dir = fullfile("results","cool_audio_plots_near");
save_audio_ofdm_plots(debug, conf, out_dir, "near");

 res.biterrors    = sum(rxbits ~= txbits);
 ber = res.biterrors/length(rxbits);
 fprintf('BER: %.4f\n', ber);

