function [rxbits, conf, debug] = rxofdm2(rxsignal, conf)
%{
RXOFDM2  OFDM receiver for the audio link with frame sync, channel estimation, and optional tracking.

[rxbits, conf, debug] = RXOFDM2(rxsignal, conf) processes a real-valued received audio waveform,
downconverts it to complex baseband, low-pass filters it, performs frame synchronization using the
single-carrier preamble, and resamples to the OFDM baseband rate. It then extracts OFDM symbols,
removes the cyclic prefix, FFTs to the frequency domain, estimates the channel from the known
training OFDM symbol, and equalizes/demodulates QPSK data symbols to recover the transmitted bits.

Depending on the configuration, the receiver can operate with:
  - static one-tap equalization (no tracking),
  - decision-directed per-subcarrier channel tracking (amplitude+phase), or
  - Viterbi–Viterbi phase tracking (CPE-only or per-subcarrier).

The function returns the recovered bitstream and a debug struct containing intermediate signals,
channel estimates, and constellation points for plotting/analysis.

Inputs:
  - rxsignal : real-valued received audio samples (column or row vector)
  - conf     : configuration struct (OFDM params, carrier/audio params, tracking settings)

Outputs:
  - rxbits : recovered payload bits (length = conf.nbits, padded/truncated if needed)
  - conf   : (passed through) configuration struct used by the receiver
  - debug  : struct with intermediate outputs (sync info, FFT symbols, H/H_hat, theta_hat, etc.)
%}

% 0) Init / parameters
rxsignal = rxsignal(:);
Ns       = length(rxsignal);

Ncar     = conf.ofdm.ncarrier;     
Ncp      = conf.ofdm.cplen;         
Nfft     = Ncar;
Nsym_len = Nfft + Ncp;             

rxbits = zeros(conf.nbits, 1);
debug  = struct();

% 1) Downconversion to complex baseband (at conf.f_s)
n  = (0:Ns-1).';
wc = 2*pi*conf.f_c/conf.f_s;
rx_bb = rxsignal .* exp(-1j*wc.*n);

% 2) Low-pass filtering (OFDM front-end)
rx_bb_filt = ofdmlowpass(rx_bb, conf, (1.5*conf.ofdm.bandwidth/2));


% 3) Matched filter + Frame synchronization (single-carrier preamble)
h_mf = conj(flipud(conf.sc.txpulse(:)));
rx_preamble_mf = conv(rx_bb, h_mf, 'same');

[beginning_of_data, fs_debug] = frame_sync(rx_preamble_mf, conf);
debug.fs = fs_debug;
debug.rx_preamble_mf = rx_preamble_mf;

ofdm_48k = rx_bb_filt(beginning_of_data:end);

% 4) Resample to OFDM baseband sampling rate
ofdm_fs = ofdm_rx_resampling(ofdm_48k, conf);
ofdm_fs = ofdm_fs(:);

% 5) Serial-to-Parallel: segment into OFDM symbols + remove CP
Ns_ofdm = length(ofdm_fs);
max_sym = floor(Ns_ofdm / Nsym_len);

% Expected symbols: 1 known + Ndata
Ndata_sym_exp = conf.nbits / (conf.modulation_order * Ncar);
Ntot_sym_exp  = floor(Ndata_sym_exp) + 1;

Nsym = min(max_sym, Ntot_sym_exp);
debug.Nsym = Nsym;
debug.Nfft = Nfft;
debug.beginning_of_data = beginning_of_data;

% Take only first Nsym symbols
ofdm_fs = ofdm_fs(1 : Nsym * Nsym_len);

ofdm_blocks_cp = reshape(ofdm_fs, Nsym_len, Nsym);
ofdm_blocks    = ofdm_blocks_cp(Ncp+1:end, :);  % remove CP

debug.ofdm_blocks_cp = ofdm_blocks_cp;
debug.ofdm_blocks    = ofdm_blocks;

% 6) FFT -> frequency-domain subcarriers
Z = fft(ofdm_blocks, Nfft, 1);    % [Nfft x Nsym]
debug.Z = Z;

if Nsym >= 2
    debug.rx_data_raw = Z(:,2:end);
else
    debug.rx_data_raw = Z(:);
end

% 7) Regenerate known OFDM symbol (TX-consistent)
pn_len = conf.sc.nsyms + 2*Ncar;
pn_seq = lfsr_framesync(pn_len);

known_bits = pn_seq(conf.sc.nsyms + 1 : conf.sc.nsyms + 2*Ncar);
known_bits = reshape(known_bits, 2, []).';         % [Ncar x 2]

GrayMap = conf.constellation(:).';
A_known = GrayMap(bi2de(known_bits, 'left-msb') + 1).';  % [Ncar x 1]

% 8) LS channel estimate (quasi - not really LS) + channel-length truncation
Z_known = Z(:,1);
H_long  = Z_known ./ A_known;        % [Nfft x 1]
h_long  = ifft(H_long, Nfft);

thr = 0.01 * max(abs(h_long));
idx_last = find(abs(h_long) > thr, 1, 'last');
if isempty(idx_last)
    Lh = 1;
else
    Lh = idx_last;
end

h_trunc = h_long(1:Lh);
h_pad   = [h_trunc; zeros(Nfft-Lh,1)];
H_hat_freq = fft(h_pad, Nfft);

debug.H   = H_hat_freq;
debug.H0  = H_hat_freq;
debug.h0  = ifft(H_hat_freq, Nfft);
debug.Lh  = Lh;

% 9) Equalization + tracking
Ndata_sym_eff = Nsym - 1;                  % data symbols actually present
rx_data = Z(:,2:Nsym);                     % [Ncar x Ndata_sym_eff]
eq_syms = zeros(Nfft, Ndata_sym_eff);

if conf.channel_tracking == 0
    % Static one-tap equalization
    eq_syms = rx_data ./ H_hat_freq;

    % For plotting convenience: constant channel across time
    debug.H_hat = repmat(H_hat_freq, 1, Nsym);
    debug.theta_hat = [];

elseif conf.channel_tracking == 1
    H0 = H_hat_freq;

    if conf.phase.avg == 0
        % Decision-directed per-tone channel tracking (amp+phase)
        alpha = conf.phase.alpha_dd;

        H_hat = zeros(Nfft, Nsym);
        H_hat(:,1) = H0;

        for s = 2:Nsym
            Z_s   = Z(:,s);
            H_prev = H_hat(:,s-1);

            tmp_eq = Z_s ./ H_prev;

            Lsym  = length(tmp_eq);
            cand  = repmat(tmp_eq, 1, 4);
            grayM = repmat(GrayMap, Lsym, 1);
            D     = abs(cand - grayM).^2;
            [~,ind] = min(D, [], 2);
            a_hat = GrayMap(ind).';

            H_new_est = Z_s ./ a_hat;
            H_hat(:,s) = alpha*H_prev + (1-alpha)*H_new_est;

            eq_syms(:,s-1) = Z_s ./ H_hat(:,s);
        end

        debug.H_hat = H_hat;
        debug.theta_hat = angle(H_hat(:,2:end) ./ H_hat(:,1:end-1)); % Ncar x (Nsym-1)
        debug.H_mag = abs(H_hat);

    elseif conf.phase.avg == 1
        % Viterbi–Viterbi CPE-only (one phase per OFDM symbol)
        theta_hat  = 0;
        alpha_cpe  = conf.phase.alpha_cpe;
        theta_hist = zeros(1, Ndata_sym_eff);

        for k = 1:Ndata_sym_eff
            sym_static = rx_data(:,k) ./ H0;
            sym_corr   = sym_static .* exp(-1j * theta_hat);

            vv_sum   = sum(-(sym_corr).^4);
            delta_th = 0.25 * angle(vv_sum);

            theta_hat = theta_hat + alpha_cpe * delta_th;
            theta_hat = mod(theta_hat + pi, 2*pi) - pi;

            theta_hist(k) = theta_hat;
            eq_syms(:,k)  = sym_static .* exp(-1j * theta_hat);
        end

        debug.theta_hat = theta_hist;  % 1 x Ndata
        debug.H_hat = [H0, H0 .* exp(1j*theta_hist)]; % Ncar x (1+Ndata)

    elseif conf.phase.avg == 2
        % Per-subcarrier Viterbi–Viterbi + LPF (one phase per tone)
        theta_hat  = zeros(Ncar, 1);
        theta_hist = zeros(Ncar, Ndata_sym_eff);
        alpha      = conf.phase.lpf_factor;

        for k = 1:Ndata_sym_eff
            sym_static = rx_data(:,k) ./ H0;
            sym_corr   = sym_static .* exp(-1j * theta_hat);

            vv_vec   = -(sym_corr.^4);
            delta_th = 0.25 * angle(vv_vec);

            theta_hat = theta_hat + alpha * delta_th;
            theta_hat = mod(theta_hat + pi, 2*pi) - pi;

            theta_hist(:,k) = theta_hat;
            eq_syms(:,k) = sym_static .* exp(-1j * theta_hat);
        end

        debug.theta_hat = theta_hist;                 % Ncar x Ndata
        debug.H_hat = [H0, H0 .* exp(1j*theta_hist)]; % Ncar x (1+Ndata)

    else
        error('rxofdm:badPhaseMode', 'Unknown conf.phase.avg = %d', conf.phase.avg);
    end
else
    error('rxofdm:badTrackingFlag', 'Unknown conf.channel_tracking = %d', conf.channel_tracking);
end

% Save equalized constellation for plotting
debug.rx_constellation = eq_syms(:);
debug.tx_constellation = GrayMap(:);
debug.rx_baseband      = rx_bb;
debug.rx_filtered      = rx_bb_filt;

% 10) QPSK Gray demapping -> bits
eq_vec = eq_syms(:);
if isempty(eq_vec)
    warning('rxofdm:noEqSyms', 'No equalized symbols available, returning zeros.');
    return;
end

Lsym = length(eq_vec);
cand2  = repmat(eq_vec, 1, 4);
grayM2 = repmat(GrayMap, Lsym, 1);
D2     = abs(cand2 - grayM2).^2;
[~,ind2] = min(D2, [], 2);

demappedGray = de2bi(ind2-1, 2, 'left-msb');  % [Lsym x 2]

rxbit_vec  = demappedGray.';      % [2 x Lsym]
rxbits_all = rxbit_vec(:);        % [2*Lsym x 1]

% Trim/pad to conf.nbits
if length(rxbits_all) < conf.nbits
    warning('rxofdm:bitLength', ...
        'Received fewer bits (%d) than conf.nbits (%d). Padding zeros.', ...
        length(rxbits_all), conf.nbits);
    rxbits = [rxbits_all; zeros(conf.nbits - length(rxbits_all), 1)];
else
    rxbits = rxbits_all(1:conf.nbits);
end

end
