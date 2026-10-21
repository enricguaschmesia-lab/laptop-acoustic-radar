function [rxbits, conf, debug] = rxofdm1(rxsignal,conf)
%{
RXOFDM1  OFDM receiver with periodic training symbols and optional channel/phase tracking.

[rxbits, conf, debug] = RXOFDM1(rxsignal, conf) recovers a QPSK-OFDM payload from a received
audio waveform. The function downconverts and low-pass filters the signal, performs frame
synchronization using the single-carrier preamble, resamples to the OFDM baseband rate, extracts
OFDM symbols (CP removal + FFT), and regenerates the known training symbol used at the transmitter.

It supports periodic training symbols (spacing given by conf.ofdm.pilot_interval) and uses the
latest training symbol to (re-)estimate the channel. Data symbols are then equalized using either:
  - no tracking (static per-block channel from training),
  - decision-directed per-subcarrier channel tracking (amplitude+phase),
  - Viterbi–Viterbi common phase error tracking (one phase per symbol), or
  - per-subcarrier Viterbi–Viterbi phase tracking.

The equalized QPSK symbols are finally Gray-demapped to produce the recovered payload bitstream.
A debug struct is returned with intermediate variables (sync info, raw/equalized constellations,
channel snapshots, phase histories, etc.) for visualization.

Inputs:
  - rxsignal : real-valued received audio samples
  - conf     : configuration struct (OFDM/preamble parameters and tracking settings)

Outputs:
  - rxbits : recovered payload bits (length = conf.nbits, padded/truncated if needed)
  - conf   : (passed through) configuration struct used by the receiver
  - debug  : struct with intermediate results for plotting/analysis
%}

% 0) Init / parameters
rxsignal = rxsignal(:);                 % ensure column vector
Ns       = length(rxsignal);

Ncar     = conf.ofdm.ncarrier;         % e.g. 512
Ncp      = conf.ofdm.cplen;            % e.g. 256
Nfft     = Ncar;
Nsym_len = Nfft + Ncp;                 % samples per OFDM symbol (CP+data)

M        = 2^conf.modulation_order;    % 4 for QPSK

% Number of DATA symbols (from bit length and QPSK)
Ndata_sym = conf.nbits / (conf.modulation_order * Ncar);
if mod(Ndata_sym,1) ~= 0
    error('rxofdm:bitLength',...
          'conf.nbits is not a multiple of modulation_order*Ncar.');
end

% Pilot / training spacing (must match TX)
if isfield(conf.ofdm,'pilot_interval')
    pilot_int = conf.ofdm.pilot_interval;
else
    % Fallback: legacy behavior = 1 training symbol only
    pilot_int = Ndata_sym + 1;
end

% How many training symbols were sent (expected)
Ntrain_sym = ceil(Ndata_sym / pilot_int);
Ntot_sym   = Ndata_sym + Ntrain_sym;    % total OFDM symbols (training+data)

% Default output and debug struct
rxbits = zeros(conf.nbits,1);
debug  = struct();

% 1) Downconversion to complex baseband (48 kHz)
n  = (0:Ns-1).';
wc = 2*pi*conf.f_c/conf.f_s;           % digital carrier rad/samp

rx_bb = rxsignal .* exp(-1j*wc.*n);    % complex baseband @ conf.f_s

% 2) Low-pass filtering (OFDM front-end)
% ofdmlowpass uses conf.ofdm.bandwidth internally, f argument is ignored
rx_bb_filt = ofdmlowpass(rx_bb, conf, (1.5*conf.ofdm.bandwidth / 2));

% 3) Matched filter and Frame synchronization (single-carrier preamble)
% Matched filter for the RRC pulse used at TX:
h_mf = conj(flipud(conf.sc.txpulse(:)));

% Output has same length and time grid as rx_bb
rx_preamble_mf = conv(rx_bb, h_mf, 'same');

% Run frame synchronization on the matched-filtered signal
[beginning_of_data, fs_debug] = frame_sync(rx_preamble_mf, conf);
debug.fs = fs_debug;                 % store in debug struct

% Store matched-filtered preamble output for debugging
debug.rx_preamble_mf = rx_preamble_mf;

% Extract OFDM portion at conf.f_s
ofdm_48k = rx_bb_filt(beginning_of_data:end);

% 4) Resample to OFDM baseband sampling rate fs_ofdm ~ ofdm.bandwidth
ofdm_fs = ofdm_rx_resampling(ofdm_48k, conf);   % complex vector at fs_ofdm
ofdm_fs = ofdm_fs(:);                           % ensure column

% 5) Serial-to-Parallel: segment into OFDM symbols + remove CP
Ns_ofdm = length(ofdm_fs);
max_sym = floor(Ns_ofdm / Nsym_len);

% We expect at most Ntot_sym symbols
Nsym = min(max_sym, Ntot_sym);

% Take only the first Nsym symbols
ofdm_fs = ofdm_fs(1 : Nsym * Nsym_len);

% Reshape into [Nsym_len x Nsym] blocks: each column = one OFDM symbol (with CP)
ofdm_blocks_cp = reshape(ofdm_fs, Nsym_len, Nsym);

% Remove CP: keep last Nfft samples → [Nfft x Nsym]
ofdm_blocks = ofdm_blocks_cp(Ncp+1:end, :);   % [Nfft x Nsym]

% 6) FFT → frequency-domain symbols (subcarriers)
Z = fft(ofdm_blocks, Nfft, 1);   % size [Nfft x Nsym]

% 7) Regenerate PN bits for preamble + known OFDM (TX-consistent)
pn_len = conf.sc.nsyms + 2*Ncar;
pn_seq = lfsr_framesync(pn_len);

% Known OFDM part
known_bits = pn_seq(conf.sc.nsyms + 1 : conf.sc.nsyms + 2*Ncar);
known_bits = reshape(known_bits, 2, []).';         % [Ncar x 2]

GrayMap = conf.constellation;

A_known = GrayMap(bi2de(known_bits, 'left-msb') + 1).';  % [Ncar x 1]

% 8) Rebuild symbol pattern (training vs data) to match TX
max_sym_pattern = Ntot_sym;
sym_is_training = false(1, max_sym_pattern);
sym_data_idx    = zeros(1, max_sym_pattern);

data_idx = 1;
sym_idx  = 0;

while data_idx <= Ndata_sym
    % One training symbol
    sym_idx = sym_idx + 1;
    sym_is_training(sym_idx) = true;
    sym_data_idx(sym_idx)    = 0;

    % Up to pilot_int data symbols
    for k = 1:pilot_int
        if data_idx > Ndata_sym
            break;
        end
        sym_idx = sym_idx + 1;
        sym_is_training(sym_idx) = false;
        sym_data_idx(sym_idx)    = data_idx;
        data_idx = data_idx + 1;
    end
end

Nsym_pattern      = sym_idx;
sym_is_training   = sym_is_training(1:Nsym_pattern);
sym_data_idx      = sym_data_idx(1:Nsym_pattern);

% we restirvt pattern to actually received symbols
sym_is_training_rx = sym_is_training(1:Nsym);
sym_data_idx_rx    = sym_data_idx(1:Nsym);

% how many data symbols did we effectively receive 
Ndata_sym_eff = sum(~sym_is_training_rx);
Ndata_sym_eff = min(Ndata_sym_eff, Ndata_sym);

% Preallocate equalized symbols for DATA only
eq_syms = zeros(Nfft, Ndata_sym_eff);
data_counter = 0;

% For debug: store raw unequalized data symbols only
data_cols = find(~sym_is_training_rx);
data_cols = data_cols(1:min(length(data_cols), Ndata_sym_eff));
if ~isempty(data_cols)
    debug.rx_data_raw = Z(:, data_cols);
else
    debug.rx_data_raw = [];
end

% 9) Channel estimation and equalization (using latest training)
% We will process OFDM symbols in time order. For each training symbol
% we compute an LS + time-truncated channel estimate H_static. For each
% data symbol, we use the most recent H_static (and optionally tracking).

H_static   = [];   % last LS+trunc estimate from a training symbol
H_prev     = [];   % for decision-directed tracking (avg=0)
H0         = [];   % static amplitude+phase for CPE tracking
theta_hat  = 0;    % scalar CPE
theta_hist = [];   % for debug
theta_hat_vec = [];    % per-subcarrier CPE

if conf.channel_tracking == 0
    %  Static LS, block-wise 
    for s = 1:Nsym
        Z_s = Z(:,s);

        if sym_is_training_rx(s)
            % Compute LS+trunc channel estimate from this training symbol
            H_long = Z_s ./ A_known;
            h_long = ifft(H_long, Nfft);

            thr = 0.01 * max(abs(h_long));
            idx_last = find(abs(h_long) > thr, 1, 'last');
            if isempty(idx_last)
                Lh = 1;
            else
                Lh = idx_last;
            end

            h_trunc = h_long(1:Lh);
            h_pad   = [h_trunc; zeros(Nfft-Lh,1)];
            H_static = fft(h_pad, Nfft);

            debug.H = H_static;  % store last training-based H

        else
            if isempty(H_static)
                % No training seen yet, cannot equalize reliably
                continue;
            end
            data_counter = data_counter + 1;
            if data_counter > Ndata_sym_eff
                break;
            end
            eq_syms(:,data_counter) = Z_s ./ H_static;
        end
    end

elseif conf.channel_tracking == 1 && conf.phase.avg == 0
    %  Decision-directed channel tracking (amp+phase) 
    alpha = conf.phase.alpha_dd;   % smoothing factor

    H_hist = zeros(Nfft, Ndata_sym_eff);   % store per-data-step H for debug

    for s = 1:Nsym
        Z_s = Z(:,s);

        if sym_is_training_rx(s)
            % New training symbol → re-anchor H_prev with LS+trunc
            H_long = Z_s ./ A_known;
            h_long = ifft(H_long, Nfft);

            thr = 0.01 * max(abs(h_long));
            idx_last = find(abs(h_long) > thr, 1, 'last');
            if isempty(idx_last)
                Lh = 1;
            else
                Lh = idx_last;
            end

            h_trunc = h_long(1:Lh);
            h_pad   = [h_trunc; zeros(Nfft-Lh,1)];
            H_static = fft(h_pad, Nfft);

            H_prev = H_static;
            debug.H = H_static;

        else
            if isempty(H_prev)
                continue;
            end
            data_counter = data_counter + 1;
            if data_counter > Ndata_sym_eff
                break;
            end

            % One-tap equalization with previous estimate
            tmp_eq = Z_s ./ H_prev;

            % Nearest-neighbor decision to QPSK constellation
            Lsym  = length(tmp_eq);
            cand  = repmat(tmp_eq, 1, 4);
            grayM = repmat(GrayMap, Lsym, 1);
            D     = abs(cand - grayM).^2;
            [~,ind] = min(D, [], 2);
            a_hat = GrayMap(ind).';              % decided symbols

            % Instantaneous channel estimate
            H_new_est = Z_s ./ a_hat;

            % Smoothed update
            H_new = alpha * H_prev + (1-alpha) * H_new_est;

            % Final equalization using updated channel
            eq_syms(:,data_counter) = Z_s ./ H_new;

            H_prev = H_new;
            H_hist(:,data_counter) = H_new;
        end
    end

    % Debug: phase and magnitude evolution over data symbols  DO NOT USE IN
    % THE END WE HAVE BETTER PLOTS IN OTHER SCRPIT AS THIS IS VS SUBCARRIER
    % INDEX NOT VS FREQUENCY!!!!
    if data_counter >= 2
        debug.theta_hat = angle(H_hist(:,2:data_counter) ./ H_hist(:,1:data_counter-1));
        debug.H_mag     = abs(H_hist(:,1:data_counter));
    else
        debug.theta_hat = [];
        if ~isempty(H_prev)
            debug.H_mag = abs(H_prev);
        else
            debug.H_mag = [];
        end
    end

elseif conf.channel_tracking == 1 && conf.phase.avg == 1
    %  Viterbi-Viterbi CPE tracking (global scalar) 
    alpha_cpe = conf.phase.alpha_cpe;
    theta_hat = 0;
    theta_hist = zeros(1, Ndata_sym_eff);

    for s = 1:Nsym
        Z_s = Z(:,s);

        if sym_is_training_rx(s)
            % Recompute static H0 from training
            H_long = Z_s ./ A_known;
            h_long = ifft(H_long, Nfft);

            thr = 0.01 * max(abs(h_long));
            idx_last = find(abs(h_long) > thr, 1, 'last');
            if isempty(idx_last)
                Lh = 1;
            else
                Lh = idx_last;
            end

            h_trunc = h_long(1:Lh);
            h_pad   = [h_trunc; zeros(Nfft-Lh,1)];
            H_static = fft(h_pad, Nfft);

            H0 = H_static;
            theta_hat = 0;      % restart CPE from scratch at this training
            debug.H = H_static;

        else
            if isempty(H0)
                continue;
            end
            data_counter = data_counter + 1;
            if data_counter > Ndata_sym_eff
                break;
            end

            % 1) Equalize with static H0
            sym_static = Z_s ./ H0;

            % 2) Apply current CPE estimate
            sym_corr = sym_static .* exp(-1j * theta_hat);

            % 3) Viterbi-Viterbi residual phase
            vv_sum   = sum(-(sym_corr).^4);
            delta_th = 0.25 * angle(vv_sum);

            % 4) LPF update
            theta_hat = theta_hat + alpha_cpe * delta_th;
            theta_hat = mod(theta_hat + pi, 2*pi) - pi;

            theta_hist(data_counter) = theta_hat;

            % 5) Final correction
            eq_syms(:,data_counter) = sym_static .* exp(-1j * theta_hat);
        end
    end

    debug.theta_hat = theta_hist(1:data_counter);

elseif conf.channel_tracking == 1 && conf.phase.avg == 2
    %  Per-subcarrier Viterbi-Viterbi phase tracking
    alpha_lpf = conf.phase.lpf_factor;

    H = [];                        % static LS estimate from last training
    theta_hat_vec = [];            % per-subcarrier CPE
    theta_hist = zeros(Nfft, Ndata_sym_eff);

    for s = 1:Nsym
        Z_s = Z(:,s);

        if sym_is_training_rx(s)
            % LS+trunc to get H from this training
            H_long = Z_s ./ A_known;
            h_long = ifft(H_long, Nfft);

            thr = 0.01 * max(abs(h_long));
            idx_last = find(abs(h_long) > thr, 1, 'last');
            if isempty(idx_last)
                Lh = 1;
            else
                Lh = idx_last;
            end

            h_trunc = h_long(1:Lh);
            h_pad   = [h_trunc; zeros(Nfft-Lh,1)];
            H = fft(h_pad, Nfft);

            theta_hat_vec = zeros(Nfft,1);   % reset phase
            debug.H = H;

        else
            if isempty(H)
                continue;
            end
            data_counter = data_counter + 1;
            if data_counter > Ndata_sym_eff
                break;
            end

            % Equalize using H (amp + initial phase)
            sym_static = Z_s ./ H;

            % Apply previous per-subcarrier phase estimate
            sym_corr = sym_static .* exp(-1j * theta_hat_vec);

            % VV residual phase per subcarrier
            vv_vec   = -(sym_corr.^4);
            delta_th = 0.25 * angle(vv_vec);

            % LPF update
            theta_hat_vec = theta_hat_vec + alpha_lpf * delta_th;
            theta_hat_vec = mod(theta_hat_vec + pi, 2*pi) - pi;

            theta_hist(:,data_counter) = theta_hat_vec;

            % Final correction
            eq_syms(:,data_counter) = sym_static .* exp(-1j * theta_hat_vec);
        end
    end

    debug.theta_hat = theta_hist(:,1:data_counter);
else
    error('rxofdm:channel_tracking','Unsupported channel_tracking / phase.avg combination.');
end

% If for some reason we filled fewer data symbols
if data_counter < Ndata_sym_eff
    eq_syms = eq_syms(:, 1:data_counter);
    Ndata_sym_eff = data_counter;
end

% Save equalized constellation for plotting
debug.rx_constellation = eq_syms(:);
debug.tx_constellation = GrayMap(:);
debug.rx_baseband      = rx_bb;
debug.rx_filtered      = rx_bb_filt;

% 10) QPSK Gray demapping → DATA bits only
eq_vec = eq_syms(:);              % complex vector, all DATA symbols

Lsym = length(eq_vec);

if Lsym == 0
    warning('rxofdm:noData','No data symbols equalized, returning zeros.');
    rxbits = zeros(conf.nbits,1);
    return;
end

% Distances to GrayMap for demapper
cand2  = repmat(eq_vec, 1, 4);    % [Lsym x 4]
grayM2 = repmat(GrayMap, Lsym, 1);
D2     = abs(cand2 - grayM2).^2;
[~,ind2] = min(D2, [], 2);

demappedGray = de2bi(ind2-1, 2, 'left-msb');  % [Lsym x 2]

% Serialize bits (only data symbols)
rxbit_vec  = demappedGray.';        % [2 x Lsym]
rxbits_all = rxbit_vec(:);          % column vector

% Trim or pad to conf.nbits
if length(rxbits_all) < conf.nbits
    warning('rxofdm:bitLength',...
        'Received fewer bits (%d) than conf.nbits (%d).', length(rxbits_all), conf.nbits);
    rxbits = [rxbits_all; zeros(conf.nbits - length(rxbits_all),1)];
else
    rxbits = rxbits_all(1:conf.nbits);
end

end
