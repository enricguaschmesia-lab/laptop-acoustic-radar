function [txsignal, conf] = txofdm(txbits,conf)
%{
TXOFDM  Generate an audio-band OFDM transmit waveform with a single training symbol.

[txsignal, conf] = TXOFDM(txbits, conf) maps an input payload bitstream to QPSK-OFDM
(Ncar subcarriers, CP length Ncp), prepends one known OFDM training symbol (derived from an
LFSR PN sequence), and prepends a BPSK single-carrier preamble for synchronization (also PN-based).

The complex baseband frame is resampled to the audio sampling rate conf.f_s, normalized to unit
average power, and digitally upconverted to a real passband signal centered at conf.f_c. The final
waveform is scaled to stay within [-1, 1] (with headroom) to avoid audio clipping.

Inputs:
  - txbits : column vector of payload bits (QPSK → length should be 2*Ncar*Ndatasym)
  - conf   : configuration struct (OFDM, preamble, carrier/audio parameters)

Outputs:
  - txsignal : real-valued passband transmit waveform at conf.f_s
  - conf     : (passed through) configuration struct used for generation
%}


% locakl variables for convenience
Ncar      = conf.ofdm.ncarrier;          % 512
Ncp       = conf.ofdm.cplen;             % 256
Ndatasym  = length(txbits)/(2*Ncar);     % should be 50

if Ndatasym ~= 50
    warning('Expected 50 OFDM data symbols, got %d',Ndatasym);
end

% 1) Generate PN sequence for preamble + known OFDM symbol
% Need: 500 bits (preamble) + 1024 bits (known OFDM symbol) = 1524 bits
pn_len_needed = conf.sc.nsyms + 2*conf.ofdm.ncarrier; % 500 + 1024
pn_seq = lfsr_framesync(pn_len_needed);               % 0/1 bits

% Split PN bits:
preamble_bits    = pn_seq(1 : conf.sc.nsyms);                         % 500 bits
known_ofdm_bits  = pn_seq(conf.sc.nsyms+1 : conf.sc.nsyms+2*Ncar);    % 1024 bits

% 2) Build OFDM bitstream: [known OFDM bits] + [data bits]
ofdm_bits = [known_ofdm_bits; txbits];        % length = 2*512*(1 + 50) = 52224 bits
Nofdm_sym = length(ofdm_bits)/(2*Ncar);       % should be 51 (1 training + 50 data)

if mod(length(ofdm_bits), 2*Ncar) ~= 0
    error('OFDM bits length is not a multiple of 2*Ncarriers.');
end

% 3) QPSK mapping using GRAY MAPPING
% ofdm_bits_reshaped: each row = [b0 b1] for one QPSK symbol
ofdm_bits_reshaped = reshape(ofdm_bits, 2, []).';  % [Nsym*Ncar x 2]

% Gray mapping (Symbols, normalized)
GrayMap = conf.constellation;

% Map input bitstream using Gray mapping
mappedGray = GrayMap(bi2de(ofdm_bits_reshaped, 'left-msb') + 1).';   % column vector

% Serial-to-Parallel: Ncar subcarriers per OFDM symbol
A = reshape(mappedGray, Ncar, Nofdm_sym);  % size: [512 x 51]
% Column 1 = known training OFDM symbol

% 4) IFFT and CP insertion and P/S (baseband OFDM at fs_ofdm = Ncar * spacing)
% IFFT along subcarrier dimension
ofdm_td = ifft(A, Ncar, 1);   % [Ncar x Nofdm_sym], complex

% CP insertion: each symbol -> [CP(256); symbol(512)]
sym_len   = Ncar + Ncp;
ofdm_baseband = zeros(sym_len * Nofdm_sym, 1); %will hold all CP+OFDM symbols
                                            % after Parallel to Serial, in
                                            % baseband, at sampling rate =
                                            % OFDM Bandwidth(conf.ofdm.ncarrier*conf.ofdm.spacing)

for n = 1:Nofdm_sym
    sym_n      = ofdm_td(:,n);                    % 512x1
    cp_n       = sym_n(end-Ncp+1:end);           % 256x1
    sym_cp_n   = [cp_n; sym_n];                  % 768x1
    idx_start  = (n-1)*sym_len + 1;
    idx_end    = n*sym_len;
    ofdm_baseband(idx_start:idx_end) = sym_cp_n;
end


% 5) Resample OFDM baseband to conf.f_s (audio sampling rate)
ofdm_os = ofdm_tx_resampling(ofdm_baseband, conf);  % complex, @ conf.f_s

%now we normalize our OFDM data to unit average power
ofdm_os = ofdm_os / sqrt(mean(abs(ofdm_os).^2));
power   = abs(ofdm_os).^2;
P_peak  = max(power);
P_avg   = mean(power);
PAPR_dB = 10*log10(P_peak / P_avg);
fprintf('OFDM PAPR (baseband, before upconversion) = %.2f dB\n', PAPR_dB);

% 6) Generate preamble: BPSK, upsample to conf.f_s, RRC pulse shaping
% map preamble using BPSK
preamble_bpsk = -2*(preamble_bits) + 1;    % 0 -> +1, 1 -> -1 


% Upsample by oversampling factor
os_factor = conf.sc.os_factor;             % = conf.f_s / conf.sc.f_sym, should be 48
preamble_ups = upsample(preamble_bpsk, os_factor);  % insert zeros

% RRC pulse (already in conf.sc.txpulse)
tx_filter_length = conf.sc.txpulse_length; 
% Convolution: FULL
preamble_pulse = conv(preamble_ups, conf.sc.txpulse, 'full');

%The convolution above pads the result with 0's at the beginning and end.
%So we trim our signal to the non-zero window:
preamble_tx = preamble_pulse(1+tx_filter_length:end - tx_filter_length);   % real, pulse-shaped


%we normalize our preamble waveform to unit average power so that it matches the
%OFDM waveform average power, and none of them is weaker/stronger in power
%relative to the other:
preamble_tx = preamble_tx / sqrt(mean(abs(preamble_tx).^2));


% 7) Concatenate preamble and OFDM (both at conf.f_s)
% We make sure OFDM baseband is column vector
ofdm_os = ofdm_os(:);

% Frame baseband: [preamble ; OFDM (1 known symbol, 50 random symbols)] at fs'= 48kHz
frame_baseband = [preamble_tx; ofdm_os];  % contains real preamble, complex normalized QPSK symbols

% 8) Digital upconversion to carrier f_c (real passband for audio)
Nsamples = length(frame_baseband);
n = (0:Nsamples-1).';

wc = 2*pi*conf.f_c / conf.f_s;

carrier_cos = cos(wc * n);
carrier_sin = sin(wc * n);

I_bb = real(frame_baseband);
Q_bb = imag(frame_baseband);

% I(t)cos(2πfct) - Q(t)sin(2πfct)
txsignal = I_bb .* carrier_cos - Q_bb .* carrier_sin;

%LINEAR SCALE TO THE RANGE [-1,1] TO AVOID CLIPPING
maxabs = max(abs(txsignal));
if maxabs > 0
    txsignal = txsignal / maxabs * 0.95;  % 5% headroom
end

% Ensure column vector
txsignal = txsignal(:);

end
