%{
SCRIPT: Transmit and reconstruct the “Viterbi” image over an audio OFDM link.

This script reads the image “viterbi.jpg”, converts it to grayscale, serializes the pixel values
into a bitstream, and pads the bits to fit an integer number of QPSK-OFDM symbols (512 subcarriers).
It then generates an audio waveform, plays it through the loudspeaker while recording with the
microphone, decodes the received bits, computes the BER on the original (unpadded) image payload,
and reconstructs/saves the received image as “photo_rec_viterbi.png”.

Input:
  - viterbi.jpg

Outputs:
  - out.wav
  - photo_rec_viterbi.png
  - BER printed in the console
%}


clear 
close all
clc

img = imread("viterbi.jpg");
img = rgb2gray(img);
s = size(img);
figure;
imshow(img);
bitstream = de2bi(img(:), 'left-msb');
bitstream = bitstream(:);


conf.audiosystem = 'audio'; 
conf.f_c     = 8000;

conf.channel_tracking = 1 ; % for channel tracking

% Channel/phase tracking: (only works if conf.channel_tracking is set to 1)
%   0 → channel tracking (amplitude and phase)
%   1 → CPE-only Viterbi–Viterbi (global)
%   2 → per-subcarrier Viterbi–Viterbi
conf.phase.avg        = 2; 

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

% 1) RAW bits from image
txbits_raw = bitstream(:);
Lraw = length(txbits_raw);

% 2) PAD to full OFDM symbols
%    (QPSK: 2 bits/subcarrier)

bits_per_ofdm_symbol = conf.modulation_order * conf.ofdm.ncarrier;   % 2*512 = 1024
pad_len = mod(-Lraw, bits_per_ofdm_symbol);                          % 0..1023
txbits_pad = [txbits_raw; zeros(pad_len,1)];
fprintf('Payload bits: %d (raw) -> %d (padded), pad=%d\n', Lraw, length(txbits_pad), pad_len);
txbits_wh = txbits_pad;
conf.nbits = length(txbits_wh);


% 4) Transmit
[txsignal, conf] = txofdm(txbits_wh, conf);

rawtxsignal = [ zeros(conf.f_s,1) ; txsignal ; zeros(conf.f_s,1) ];
rawtxsignal = [ rawtxsignal  zeros(size(rawtxsignal)) ];



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

% 4) Receive
[rxbits, conf, debug] = rxofdm(rxsignal, conf);
rxbits_pad = rxbits;

% BER only on the original raw payload (exclude padding)
biterrors = sum(rxbits_pad(1:Lraw) ~= txbits_raw);
ber = biterrors / Lraw;
fprintf('BER (raw image bits): %.4f\n', ber);

% 5) Reconstruct image from RAW bits only
bitstream_rec = bi2de(reshape(rxbits_dewh_pad(1:Lraw), [], 8), 'left-msb');
image_rec = uint8(reshape(bitstream_rec, s));

imwrite(image_rec, 'photo_rec_viterbi.png');
figure; imshow(image_rec);



