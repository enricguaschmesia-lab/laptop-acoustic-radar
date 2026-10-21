function [txsignal, conf] = txofdm1(txbits,conf)
%{
TXOFDM1  Generate an audio-band OFDM transmit waveform with preamble and periodic training symbols.

[txsignal, conf] = TXOFDM1(txbits, conf) maps an input payload bitstream to QPSK-OFDM (Ncar
subcarriers, CP length Ncp), inserts known training OFDM symbols periodically (controlled by
conf.ofdm.pilot_interval, otherwise only one training at the start), and prepends a BPSK single-
carrier preamble generated from an LFSR PN sequence.

The resulting complex baseband frame is resampled to the audio sampling rate conf.f_s, normalized,
and digitally upconverted to a real passband signal centered at conf.f_c. The output txsignal is
scaled to stay within [-1, 1] (with headroom) to avoid audio clipping.

Inputs:
  - txbits : column vector of payload bits (length must be a multiple of 2*Ncar for QPSK)
  - conf   : configuration struct (OFDM, preamble, carrier/audio parameters)

Outputs:
  - txsignal : real-valued passband transmit waveform at conf.f_s
  - conf     : (passed through) configuration struct used for generation
%}

% for convenience local variables
Ncar      = conf.ofdm.ncarrier;          
Ncp       = conf.ofdm.cplen;            
Ndatasym  = length(txbits)/(2*Ncar);

if mod(Ndatasym,1) ~= 0
    error('txofdm:bitLength','txbits length is not a multiple of 2*Ncar.');
end

% Pilot / training OFDM spacing (data symbols between trainings)
if isfield(conf.ofdm,'pilot_interval')
    pilot_int = conf.ofdm.pilot_interval;
else
    % Fallback: only one training at start (legacy behavior)
    pilot_int = Ndatasym + 1;
end

% 1) Generate PN sequence for preamble + known OFDM symbol
pn_len_needed = conf.sc.nsyms + 2*conf.ofdm.ncarrier; % 500 + 1024
pn_seq = lfsr_framesync(pn_len_needed);               % 0/1 bits

% Split PN bits:
preamble_bits    = pn_seq(1 : conf.sc.nsyms);                         % 500 bits
known_ofdm_bits  = pn_seq(conf.sc.nsyms+1 : conf.sc.nsyms+2*Ncar);    % 1024 bits
known_ofdm_bits  = known_ofdm_bits(:);                                % column

% 2) Build OFDM bitstream pattern with periodic training symbols
% Data bits grouped per OFDM symbol
data_bits_mat = reshape(txbits, 2*Ncar, Ndatasym);    % [2*Ncar x Ndatasym]

% Compute how many training symbols we will send
Ntrain_sym = ceil(Ndatasym / pilot_int);
max_sym    = Ndatasym + Ntrain_sym;

sym_is_training = false(1, max_sym);  % true if this OFDM symbol is training
sym_data_idx    = zeros(1, max_sym);  % which data index is mapped to this symbol

data_idx = 1;
sym_idx  = 0;

while data_idx <= Ndatasym
    % Insert one training symbol
    sym_idx = sym_idx + 1;
    sym_is_training(sym_idx) = true;
    sym_data_idx(sym_idx)    = 0;

    % Insert up to pilot_int data symbols after it
    for k = 1:pilot_int
        if data_idx > Ndatasym
            break;
        end
        sym_idx = sym_idx + 1;
        sym_is_training(sym_idx) = false;
        sym_data_idx(sym_idx)    = data_idx;
        data_idx = data_idx + 1;
    end
end

Nofdm_sym = sym_idx;
sym_is_training = sym_is_training(1:Nofdm_sym);
sym_data_idx    = sym_data_idx(1:Nofdm_sym);

% Build OFDM bits matrix: each column = 2*Ncar bits for one OFDM symbol
ofdm_bits_mat = zeros(2*Ncar, Nofdm_sym);

for s = 1:Nofdm_sym
    if sym_is_training(s)
        ofdm_bits_mat(:,s) = known_ofdm_bits;
    else
        d_idx = sym_data_idx(s);
        ofdm_bits_mat(:,s) = data_bits_mat(:,d_idx);
    end
end

% Serialize into a single bitstream
ofdm_bits = ofdm_bits_mat(:);


% 3) QPSK mapping using GRAY MAPPING
% ofdm_bits_reshaped: each row = [b0 b1] for one QPSK symbol
ofdm_bits_reshaped = reshape(ofdm_bits, 2, []).';  % [Nofdm_sym*Ncar x 2]

% Gray mapping (Symbols, normalized)
GrayMap = conf.constellation;

% Map input bitstream using Gray mapping
mappedGray = GrayMap(bi2de(ofdm_bits_reshaped, 'left-msb') + 1).';   % column vector

% Serial-to-Parallel: Ncar subcarriers per OFDM symbol
A = reshape(mappedGray, Ncar, Nofdm_sym);  % size: [Ncar x Nofdm_sym]

% 4) IFFT and CP insertion and P/S (baseband OFDM at fs_ofdm)
% IFFT along subcarrier dimension
ofdm_td = ifft(A, Ncar, 1);   % [Ncar x Nofdm_sym], complex

% CP insertion: each symbol -> [CP(Ncp); symbol(Ncar)]
sym_len       = Ncar + Ncp;
ofdm_baseband = zeros(sym_len * Nofdm_sym, 1);

for n = 1:Nofdm_sym
    sym_n    = ofdm_td(:,n);
    cp_n     = sym_n(end-Ncp+1:end);
    sym_cp_n = [cp_n; sym_n];
    idx_start = (n-1)*sym_len + 1;
    idx_end   = n*sym_len;
    ofdm_baseband(idx_start:idx_end) = sym_cp_n;
end

% 5) Resample OFDM baseband to conf.f_s (audio sampling rate)
ofdm_os = ofdm_tx_resampling(ofdm_baseband, conf);  % complex, @ conf.f_s

% Normalize OFDM part to unit average power
ofdm_os = ofdm_os / sqrt(mean(abs(ofdm_os).^2));

% PAPR (baseband, before upconversion)
power   = abs(ofdm_os).^2;
P_peak  = max(power);
P_avg   = mean(power);
PAPR_dB = 10*log10(P_peak / P_avg);
fprintf('OFDM PAPR (baseband, before upconversion) = %.2f dB\n', PAPR_dB);

% 6) Generate preamble: BPSK, upsample to conf.f_s, RRC pulse shaping
% map preamble using BPSK
preamble_bpsk = -2*(preamble_bits) + 1;    % 0 -> +1, 1 -> -1 

% Upsample by oversampling factor
os_factor = conf.sc.os_factor;             % = conf.f_s / conf.sc.f_sym
preamble_ups = upsample(preamble_bpsk, os_factor);  % insert zeros

% RRC pulse (already in conf.sc.txpulse)
tx_filter_length = conf.sc.txpulse_length; 
% Convolution: FULL
preamble_pulse = conv(preamble_ups, conf.sc.txpulse, 'full');

% Trim convolution edges to the non-zero window:
preamble_tx = preamble_pulse(1+tx_filter_length : end-tx_filter_length);   % real

% Normalize preamble waveform to unit average power
preamble_tx = preamble_tx / sqrt(mean(abs(preamble_tx).^2));

% 7) Concatenate preamble and OFDM (both at conf.f_s)
ofdm_os = ofdm_os(:);
frame_baseband = [preamble_tx; ofdm_os];

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

% LINEAR SCALE TO THE RANGE [-1,1] TO AVOID CLIPPING
maxabs = max(abs(txsignal));
if maxabs > 0
    txsignal = txsignal / maxabs * 0.95;  % 5% headroom might be 
end

txsignal = txsignal(:);   % ensure column vector

end
