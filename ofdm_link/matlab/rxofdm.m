function [rxbits, conf, debug] = rxofdm(rxsignal,conf)
%{
RXOFDM  Baseline OFDM receiver with a single training symbol and optional tracking.

[rxbits, conf, debug] = RXOFDM(rxsignal, conf) recovers a QPSK-OFDM payload from a received
audio waveform for the “single-training-symbol” frame structure (1 known OFDM symbol followed by
Ndata OFDM data symbols). The function downconverts and low-pass filters the signal, performs
frame synchronization using the preamble, resamples to the OFDM baseband rate, extracts OFDM
symbols (CP removal + FFT), estimates the channel from the known training symbol, and equalizes
the remaining data symbols.

Depending on conf.channel_tracking / conf.phase.avg, it can run with:
  - no tracking (static one-tap equalization using the initial channel estimate),
  - decision-directed per-subcarrier channel tracking (amplitude+phase),
  - Viterbi–Viterbi common phase error tracking (one phase per OFDM symbol), or
  - per-subcarrier Viterbi–Viterbi phase tracking.

The recovered bits are obtained by Gray-demapping the equalized QPSK symbols. The debug struct
stores intermediate variables (sync info, raw/equalized constellations, channel estimates, phase
history, etc.) for plotting and analysis.

Inputs:
  - rxsignal : real-valued received audio samples
  - conf     : configuration struct (OFDM/preamble parameters and tracking settings)

Outputs:
  - rxbits : recovered payload bits (length = conf.nbits, padded/truncated if needed)
  - conf   : (passed through) configuration struct used by the receiver
  - debug  : struct with intermediate results for visualization/verification
%}

% 0) Init / parameters
rxsignal = rxsignal(:);                 % ensure column vector
Ns       = length(rxsignal);

Ncar     = conf.ofdm.ncarrier;         % 512
Ncp      = conf.ofdm.cplen;            % 256
Nfft     = Ncar;
Nsym_len = Nfft + Ncp;                 % 768 samples (OFDM symbol+CP)

M        = 2^conf.modulation_order;    % 4 for QPSK
Ndata_sym = conf.nbits / (conf.modulation_order*Ncar); % should be 50
Ntot_sym  = Ndata_sym + 1;             % 1 known + 50 data = 51

% Default output and 
rxbits = zeros(conf.nbits,1);
debug  = struct();

% 1) Downconversion to complex baseband (48 kHz)
n  = (0:Ns-1).';
wc = 2*pi*conf.f_c/conf.f_s;           % digital carrier rad/samp

rx_bb = rxsignal .* exp(-1j*wc.*n);    % complex baseband @ 48 kHz

% 2) Low-pass filtering (OFDM front-end)
% ofdmlowpass uses conf.ofdm.bandwidth internally, f argument is ignored
rx_bb_filt = ofdmlowpass(rx_bb, conf, (1.5*conf.ofdm.bandwidth / 2));

% 3) Matched filter and Frame synchronization (single-carrier preamble)

% Matched filter for the RRC pulse used at TX:
h_mf = conj(flipud(conf.sc.txpulse(:)));

% Output has same length and time grid as rx_bb
rx_preamble_mf = conv(rx_bb, h_mf, 'same');

% Run frame synchronization on 
% the matched-filtered signal
[beginning_of_data, fs_debug] = frame_sync(rx_preamble_mf, conf);
debug.fs = fs_debug;                 % store in debug struct

% Store matched-filtered preamble output for debugging
debug.rx_preamble_mf = rx_preamble_mf;

% beginning_of_data is the index (in the original non-match filtered rx_bb_filt)
% of the first OFDM sample right after the preamble sequence used by frame_sync.

% Extract OFDM portion at 48 kHz
ofdm_48k = rx_bb_filt(beginning_of_data:end);

% 4) Resample to OFDM baseband sampling rate fs ~2 kHz
ofdm_fs = ofdm_rx_resampling(ofdm_48k, conf);   % complex vector at fs_ofdm

ofdm_fs = ofdm_fs(:);                           % ensure column

% 5) Serial-to-Parallel: segment into OFDM symbols + remove CP
Ns_ofdm = length(ofdm_fs);
max_sym = floor(Ns_ofdm / Nsym_len);

% We expect at least Ntot_sym symbols:
Nsym = min(max_sym, Ntot_sym);

% Take only the first Nsym symbols
ofdm_fs = ofdm_fs(1 : Nsym * Nsym_len);

% Reshape into [Nsym_len x Nsym] blocks: each column = one OFDM symbol (with CP)
ofdm_blocks_cp = reshape(ofdm_fs, Nsym_len, Nsym);

% Remove CP: keep last Nfft samples → [Nfft x Nsym]
ofdm_blocks = ofdm_blocks_cp(Ncp+1:end, :);

% 6) FFT → frequency-domain symbols (subcarriers)
Z = fft(ofdm_blocks, Nfft, 1);   % size [Nfft x Nsym]
debug.Z = Z;                         % [Ncar x Nsym]
debug.Nsym = Nsym;
debug.Nfft = Nfft;
debug.beginning_of_data = beginning_of_data;

debug.ofdm_blocks = ofdm_blocks;     % [Nfft x Nsym] CP removed
debug.ofdm_blocks_cp = ofdm_blocks_cp;


% Save raw unequalized data (all data symbols)
if Nsym >= 2
    debug.rx_data_raw = Z(:,2:end);
else
    debug.rx_data_raw = Z(:);
end

% 7) Regenerate PN bits for preamble + known OFDM (TX-consistent)
% Total PN bits used at TX:
pn_len = conf.sc.nsyms + 2*Ncar;
pn_seq = lfsr_framesync(pn_len);

% We only need the known OFDM part here
known_bits = pn_seq(conf.sc.nsyms + 1 : conf.sc.nsyms + 2*Ncar);

% Map known bits to QPSK using same Gray mapping as TX
known_bits = reshape(known_bits, 2, []).';         % [512 x 2]

GrayMap = conf.constellation;

A_known = GrayMap(bi2de(known_bits, 'left-msb') + 1).';  % [512 x 1]


% 8) LS channel estimate with automatic channel-length detection
Z_known = Z(:,1);                        % column 1 = training OFDM symbol

% Per-tone channel estimate in frequency domain (Nfft tones)
H_long = Z_known ./ A_known;            % [Nfft x 1]

% Time-domain impulse response h_long via IFFT
h_long = ifft(H_long, Nfft);           % [Nfft x 1]

% Detect effective channel length Lh from significant taps
% we use a small threshold relative to max tap
thr = 0.01 * max(abs(h_long));         % 1% of max magnitude
%we find the last index of magnitude greater than the threshold
idx_last = find(abs(h_long) > thr, 1, 'last');

if isempty(idx_last)
    Lh = 1;                             % degenerate fallback
else
    Lh = idx_last;
end

% Truncate h_long to first Lh taps and zero-pad back to Nfft
h_trunc = h_long(1:Lh);                 % [Lh x 1]
h_pad   = [h_trunc; zeros(Nfft-Lh,1)];  % [Nfft x 1]

% Recompute frequency-domain channel from truncated chanel
H_hat_freq = fft(h_pad, Nfft);      % [Nfft x 1]  
debug.H0 = H_hat_freq;       % [Ncar x 1] initial LS/truncated estimate
debug.h0 = ifft(H_hat_freq); % time-domain impulse response (length Ncar)
debug.Lh = Lh;
  

% Store for debugging
debug.H = H_hat_freq;

if conf.channel_tracking == 0
    % 9) Static one-tap equalization (no phase tracking)
    Ndata_sym_eff = Nsym - 1;                % how many data OFDM symbols we actually have
    eq_syms = zeros(Nfft, Ndata_sym_eff);    % equalized QPSK symbols
    
    for s = 2:Nsym   % s index in [2..Nsym], data symbols
        Z_s = Z(:,s);                        % received subcarriers for symbol s
        eq_syms(:,s-1) = Z_s ./ H_hat_freq;  % same LS channel for all data symbols
    end
    

elseif conf.channel_tracking == 1
    % Initial per-subcarrier channel estimate from known OFDM symbol
    H_hat = zeros(Nfft, Nsym);               % store channel per subcarrier & symbol
    H0 = H_hat_freq;
    H_hat(:,1) = H_hat_freq;         % initial estimate H_hat(:,1)
    
    % 9) channel/phase tracking
    Ndata_sym_eff = Nsym - 1;                % how many data OFDM symbols we actually have
    
    eq_syms = zeros(Nfft, Ndata_sym_eff);    % equalized QPSK symbols
    
    rx_data = Z(:, 2:Nsym);            % [Ncar x Ndata_sym_eff]
    

    if conf.phase.avg == 0 %channel tracking (both amplitude and phase)
        alpha = conf.phase.alpha_dd;   % smoothing factor for decision-directed channel tracking
    
        for s = 2:Nsym   % s index in [2..Nsym], data symbols
            Z_s   = Z(:,s);              % received subcarriers for symbol s
            H_prev = H_hat(:,s-1);       % previous channel estimate (per subcarrier)
        
            % One-tap equalization with previous estimate
            tmp_eq = Z_s ./ H_prev;      % rough equalized symbols (per subcarrier)
        
            % Compute distances to GrayMap points
            Lsym  = length(tmp_eq);
            cand  = repmat(tmp_eq, 1, 4);        % [Lsym x 4]
            grayM = repmat(GrayMap, Lsym, 1);    % [Lsym x 4]
            D     = abs(cand - grayM).^2;        % squared distances
            [~,ind] = min(D, [], 2);             % closest constellation index
            a_hat = GrayMap(ind).';              % decided QPSK symbols [512 x 1]
        
            % Decision-directed channel update per subcarrier
            H_new_est = Z_s ./ a_hat;            % instantaneous channel estimate
            H_hat(:,s) = alpha*H_prev + (1-alpha)*H_new_est;   % smooth update
			debug.H_hat = H_hat;          % [Ncar x Nsym] channel estimate per symbol (including training)

        
            % Final equalization using updated channel for this symbol
            eq_syms(:,s-1) = Z_s ./ H_hat(:,s);
        end
       
        % For consistency in debug, we store a per-subcarrier phase
        % difference and magnitude evolution
        debug.theta_hat = angle(H_hat(:,2:end) ./ H_hat(:,1:end-1)); 
        debug.H_mag = abs(H_hat);

    elseif conf.phase.avg == 1 %Viterbi-Viterbi Constant Phase error (CPE) tracking (per OFDM symbol)
        theta_hat  = 0;                        % scalar CPE estimate
        alpha_cpe  = conf.phase.alpha_cpe;    
        theta_hist = zeros(1, Ndata_sym_eff);

        for k = 1:Ndata_sym_eff
            % Equalize using static H0 (amplitude + initial phase)
            sym_static = rx_data(:,k) ./ H0;   % [Ncar x 1]
    
            % Remove current CPE estimate (feedback)
            sym_corr = sym_static .* exp(-1j * theta_hat);
    
            % Viterbi–Viterbi estimate of residual phase
            %    NOTE: minus sign compensates for Gray QPSK at ±π/4
            vv_sum   = sum(-(sym_corr).^4);    % multiply by -1 AFTER ^4
            delta_th = 0.25 * angle(vv_sum);   % small residual increment
    
            % LPF update
            theta_hat = theta_hat + alpha_cpe * delta_th;
            theta_hat = mod(theta_hat + pi, 2*pi) - pi;
    
            theta_hist(k) = theta_hat;
			debug.theta_hat = theta_hist;         % [1 x Ndata_sym_eff]
			H_eff = zeros(Nfft, Nsym);
			H_eff(:,1) = H0;
			for k = 1:(Nsym-1)
				H_eff(:,k+1) = H0 .* exp(1j*theta_hist(k));
			end
			debug.H_hat = H_eff;

            % 5) Final correction
            eq_syms(:,k) = sym_static .* exp(-1j * theta_hat);
        end
    
        debug.theta_hat = theta_hist;

    elseif conf.phase.avg == 2
        % Per-subcarrier Viterbi–Viterbi + LPF tracking
        %     (vector theta_hat, one per subcarrier)
        H = H0;                                 % static LS estimate per subcarrier
        eq_syms   = zeros(Ncar, Ndata_sym_eff);
        theta_hat = zeros(Ncar, 1);             % current phase per subcarrier
        theta_hist = zeros(Ncar, Ndata_sym_eff);
    
        alpha = conf.phase.lpf_factor;     

        for k = 1:Ndata_sym_eff
            % Equalize using static H (Amplitude + Initial Phase)
            sym_static = rx_data(:, k) ./ H;     % [Ncar x 1]

            % Correct using PREVIOUS estimate
            sym_corr = sym_static .* exp(-1j * theta_hat);   % [Ncar x 1]

            % Viterbi–Viterbi residual phase per subcarrier
            vv_vec   = -(sym_corr.^4);                       % [Ncar x 1]
            delta_th = 0.25 * angle(vv_vec);              % [Ncar x 1]

            % LPF update
            theta_hat = theta_hat + alpha * delta_th;
            theta_hat = mod(theta_hat + pi, 2*pi) - pi;   % wrap per subcarrier

            % Store for plotting
            theta_hist(:, k) = theta_hat;

            % Apply final correction
            eq_syms(:, k) = sym_static .* exp(-1j * theta_hat);
        end
    
        debug.theta_hat = theta_hist;   % [Ncar x Ndata_sym_eff]
		H_eff = zeros(Ncar, Nsym);
		H_eff(:,1) = H0;
		for k = 1:(Nsym-1)
			H_eff(:,k+1) = H0 .* exp(1j*theta_hist(:,k));
		end
		debug.H_hat = H_eff;

    end

end

% Save equalized constellation for plotting
debug.rx_constellation = eq_syms(:);
debug.tx_constellation = GrayMap(:);
debug.rx_baseband      = rx_bb;
debug.rx_filtered      = rx_bb_filt;

% 10) QPSK Gray demapping → bits
% Flatten equalized symbols column-wise
eq_vec = eq_syms(:);              % complex vector, length Nfft*Ndata_sym_eff

Lbitsym = length(eq_vec);

% Distances to GrayMap for demapper
cand2  = repmat(eq_vec, 1, 4);    % [Lbitsym x 4]
grayM2 = repmat(GrayMap, Lbitsym, 1);
D2     = abs(cand2 - grayM2).^2;
[~,ind2] = min(D2, [], 2);

demappedGray = de2bi(ind2-1, 2, 'left-msb');  % [Lbitsym x 2]

% Serialize bits
rxbit_vec = demappedGray.';                  % [2 x Lbitsym]
rxbits_all = rxbit_vec(:);        % column vector

% Trim or pad to conf.nbits just in case
if length(rxbits_all) < conf.nbits
    warning('rxofdm:bitLength', ...
        'Received fewer bits (%d) than conf.nbits (%d).', length(rxbits_all), conf.nbits);
    rxbits = [rxbits_all; zeros(conf.nbits - length(rxbits_all),1)];
else
    rxbits = rxbits_all(1:conf.nbits);
end

end
