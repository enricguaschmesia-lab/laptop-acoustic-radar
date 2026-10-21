%{
SCRIPT: Transmit and reconstruct a grayscale image over an audio OFDM link.

This script reads “orig.png”, converts it to grayscale, serializes its pixel values into a
bitstream, and transmits the bits over the audio OFDM framework (loudspeaker → microphone) using
QPSK-OFDM. The receiver decodes the payload with enabling channel/phase tracking,
computes the BER against the transmitted bits, and reconstructs the
received image, saving it as “photo_rec_phase.png” and displaying it.

Input:
  - orig.png

Outputs:
  - out.wav              : transmitted waveform (for inspection/replay)
  - photo_rec_phase.png  : reconstructed received image
  - BER printed in the console
%}


clear 
close all
clc

img = imread("orig.png");
img = rgb2gray(img);

s = size(img);
figure;
imshow(img);
bitstream = de2bi(img(:), 'left-msb');
bitstream = bitstream(:);



% Options for transmission are : 
% emulator: use a channel emulator with 5 different configurations.
% audio: use the loudspeaker and microphone for the data transmission
conf.audiosystem = 'audio'; 



conf.f_c     = 8000;

%chanel parameter
conf.channel_tracking = 1; %1 if we want  tracking

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
conf.nbits = length(txbits);

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
[rxbits, conf, debug] = rxofdm(rxsignal,conf);

res.biterrors    = sum(rxbits ~= txbits);
ber = res.biterrors/length(rxbits);
fprintf('BER: %.4f\n', ber);

l = img(:);
bitstream_rec = bi2de(reshape(rxbits(1:length(bitstream)), [], 8), 'left-msb');

image_rec = uint8(reshape(bitstream_rec, s));
imwrite(image_rec, 'photo_rec_phase.png');

figure;
imshow(image_rec);




