%{
SCRIPT: Transmit and reconstruct “orig.png” over an audio OFDM link (with bit whitening).

This script reads the image “orig.png”, serializes its pixel values into a bitstream, applies
reproducible XOR-based whitening, and transmits the resulting bits over the audio OFDM framework
(loudspeaker → microphone). After reception, it decodes the bitstream, de-whitens it using the
same XOR sequence, computes the BER with respect to the transmitted bits, and reconstructs/saves
the received image as “whitening.png”.

Input:
  - orig.png

Outputs:
  - out.wav        : transmitted waveform (for inspection/replay)
  - whitening.png  : reconstructed received image
  - BER printed in the console
%}



clear
close all
clc;

img = imread("orig.png");
s = size(img);
figure;
imshow(img);
bitstream = de2bi(img(:), 'left-msb');
bitstream = bitstream(:);


conf.audiosystem = 'audio'; 
conf.f_c     = 8000;

%chanel parameter
conf.channel_tracking = 1 ; % if we want tracking
                         

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




txbits = bitstream;
[txbits_wh, xor_seq] = XORING(txbits, [], 0);   % seed=0 (reproducible)
conf.nbits = length(txbits_wh);
[txsignal, conf] = txofdm(txbits_wh, conf);
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

[rxbits, conf, debug] = rxofdm(rxsignal, conf);

% de-whiten (same XOR sequence)
rxbits_dewh = XORING(rxbits(1:conf.nbits), xor_seq);

% reconstruct from rxbits_dewh
bitstream_rec = bi2de(reshape(rxbits_dewh(1:length(bitstream)), [], 8), 'left-msb');


res.biterrors    = sum(rxbits_dewh ~= txbits);
ber = res.biterrors/length(rxbits_dewh);
fprintf('BER: %.4f\n', ber);

% image reconstruction

image_rec = uint8(reshape(bitstream_rec, s));
imwrite(image_rec, 'whitening.png');  
figure;
imshow(image_rec);
