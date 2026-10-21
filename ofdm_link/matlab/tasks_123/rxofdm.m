function [rxbits, conf, debug] = rxofdm(rxsignal, conf)
% RXOFDM  Digital OFDM Receiver for the acoustic project
%
%   [rxbits, conf, debug] = rxofdm(rxsignal, conf)
%
%   Implements a complete receiver in digital domain using:
%     - single-carrier BPSK preamble (frame_sync)
%     - OFDM known training symbol for initial channel estimation
%     - OFDM data symbols with either:
%         * per-subcarrier decision-directed tracking (default), or
%         * common phase error (CPE) Viterbi–Viterbi tracking (conf.phase.avg == 1)
%
%   Inputs:
%     rxsignal : received real-valued signal (emulator or microphone path)
%     conf     : configuration structure
%
%   Outputs:
%     rxbits   : recovered bit stream (column vector, length >= conf.nbits)
%     conf     : (unchanged, kept for API compatibility)
%     debug    : struct with internal signals (for plotting / debugging)

% -------------------------------------------------------------------------
% 0) Init / parameters
% -------------------------------------------------------------------------
rxsignal = rxsignal(:);                 % ensure column vector
Ns       = length(rxsignal);

Ncar     = conf.ofdm.ncarrier;         % e.g. 512
Ncp      = conf.ofdm.cplen;            % e.g. 256
Nfft     = Ncar;
Nsym_len = Nfft + Ncp;                 % samples per OFDM symbol at fs_ofdm

M        = 2^conf.modulation_order;    % 4 for QPSK
Ndata_sym = conf.nbits / (conf.modulation_order * Ncar);  % expected # data OFDM symbols (e.g. 50)
Ntot_sym  = Ndata_sym + 1;             % 1 known + Ndata_sym data

% Default output and debug
rxbits = zeros(conf.nbits,1);
debug  = struct();

% Ensure phase configuration exists (for backward compatibility)
if ~isfield(conf, 'phase')
    conf.phase.tracking   = false;
    conf.phase.lpf_factor = 1;
    conf.phase.alpha_cpe  = 1;
    conf.phase.avg        = 0;   % 0 = per-subcarrier DD tracking (Joelle), 1 = CPE VV tracking
    conf.phase.soft_dd    = false;   
end

% -------------------------------------------------------------------------
% 1) Downconversion to complex baseband (fs = conf.f_s)
% -------------------------------------------------------------------------
n  = (0:Ns-1).';
wc = 2*pi*conf.f_c / conf.f_s;             % digital carrier rad/sample

% Complex baseband @ 48 kHz
rx_bb = rxsignal .* exp(-1j * wc .* n);    
rx_bb = rx_bb(:);                          % ensure column vector

% -------------------------------------------------------------------------
% 2) Single-carrier preamble chain: RRC matched filter + frame sync
% -------------------------------------------------------------------------
% Matched filter for the RRC pulse used at TX:
% TX shaping: conv(upsample(bpsk, L), conf.sc.txpulse, 'same')
% RX matched filter: h_MF[n] = conj(flip(conf.sc.txpulse))
h_mf = conj(flipud(conf.sc.txpulse(:)));

% Output has same length and time grid as rx_bb
rx_preamble_mf = conv(rx_bb, h_mf, 'same');

% Run frame synchronization on the matched-filtered signal
[beginning_of_data, fs_debug] = frame_sync(rx_preamble_mf, conf);
debug.fs = fs_debug;                 % store in debug struct

% Sanity check on returned index
if beginning_of_data < 1 || beginning_of_data > length(rx_bb)
    warning('rxofdm:frameSync', ...
        'Invalid beginning_of_data index (%d). Returning zeros.', beginning_of_data);
    rxbits = zeros(size(txbits)); % or handle as you prefer
    return;
end

% (Optional) Store matched-filtered preamble output for debugging
debug.rx_preamble_mf = rx_preamble_mf;
debug.beginning_of_data = beginning_of_data;


% -------------------------------------------------------------------------
% 3) OFDM front-end: low-pass (baseband) and extraction at 48 kHz
% -------------------------------------------------------------------------
% Now we run the OFDM low-pass filter on the baseband signal itself
% and cut from the detected beginning_of_data index onwards.
rx_bb_filt = ofdmlowpass(rx_bb, conf, (1.5 * conf.ofdm.bandwidth / 2));

% Extract OFDM portion (still at fs = conf.f_s = 48 kHz)
ofdm_48k = rx_bb_filt(beginning_of_data:end);


% -------------------------------------------------------------------------
% 4) Resample to OFDM baseband sampling rate fs_ofdm ≈ 2 kHz
% -------------------------------------------------------------------------
ofdm_fs = ofdm_rx_resampling(ofdm_48k, conf);   % complex vector at fs_ofdm
ofdm_fs = ofdm_fs(:);                           % ensure column

% -------------------------------------------------------------------------
% 5) Serial-to-Parallel: segment into OFDM symbols + remove CP
% -------------------------------------------------------------------------
Ns_ofdm = length(ofdm_fs);
max_sym = floor(Ns_ofdm / Nsym_len);

% We expect at least Ntot_sym symbols:
Nsym = min(max_sym, Ntot_sym);

if Nsym < Ntot_sym
    warning('rxofdm:symbolCount', 'Only %d OFDM symbols detected, expected %d.', ...
        Nsym, Ntot_sym);
end

% Keep only the first Nsym symbols
ofdm_fs = ofdm_fs(1 : Nsym * Nsym_len);

% -------------------------------------------------------------------------
% 5a) Drop very low-energy last symbols (noise tail / partial capture)
% -------------------------------------------------------------------------
if Nsym > 1
    tmp_blocks_cp = reshape(ofdm_fs, Nsym_len, Nsym);         % [Nsym_len x Nsym]

    % Energy per symbol *including* CP
    sym_energy = sum(abs(tmp_blocks_cp).^2, 1);               % [1 x Nsym]

    % Use the median energy (excluding the very first training symbol if you want)
    % as a robust reference
    ref_energy = median(sym_energy);

    % Consider symbols below, say, 20% of median as "invalid"
    energy_thr = 0.2 * ref_energy;

    valid_syms = find(sym_energy >= energy_thr);

    if isempty(valid_syms)
        warning('rxofdm:energyPrune', ...
            'All OFDM symbols have very low energy. Keeping first symbol only.');
        last_valid = 1;
    else
        last_valid = max(valid_syms);
    end

    if last_valid < Nsym
        fprintf('Pruning %d low-energy OFDM symbols at the end (kept %d).\n', ...
                Nsym - last_valid, last_valid);
        Nsym    = last_valid;
        ofdm_fs = ofdm_fs(1 : Nsym * Nsym_len);
    end
end


% Reshape into [Nsym_len x Nsym] blocks: each column = one OFDM symbol (with CP)
ofdm_blocks_cp = reshape(ofdm_fs, Nsym_len, Nsym);

% Remove CP: keep last Nfft samples → [Nfft x Nsym]
ofdm_blocks = ofdm_blocks_cp(Ncp+1:end, :);

% -------------------------------------------------------------------------
% 6) FFT → frequency-domain symbols (subcarriers)
% -------------------------------------------------------------------------
Z = fft(ofdm_blocks, Nfft, 1);   % size [Nfft x Nsym]
% Columns: 1 = known OFDM, 2..Nsym = data symbols

% Save raw unequalized data (all data symbols)
if Nsym >= 2
    debug.rx_data_raw = Z(:,2:end);
else
    debug.rx_data_raw = Z(:);
end

% -------------------------------------------------------------------------
% 7) Regenerate PN bits for preamble + known OFDM (TX-consistent)
% -------------------------------------------------------------------------
% Total PN bits used at TX:
% - first conf.sc.nsyms      bits → preamble
% - next 2*Ncar              bits → known OFDM symbol
pn_len = conf.sc.nsyms + 2 * Ncar;
pn_seq = lfsr_framesync(pn_len);   % 0/1 bits, same LFSR as TX

% We only need the known OFDM part here
known_bits = pn_seq(conf.sc.nsyms + 1 : conf.sc.nsyms + 2*Ncar);

% Map known bits to QPSK using same Gray mapping as TX
known_bits = reshape(known_bits, 2, []).';   % [Ncar x 2]

GrayMap = conf.constellation;   % QPSK Gray map

A_known = GrayMap(bi2de(known_bits, 'left-msb') + 1).';    % [Ncar x 1]

debug.A_known = A_known;   % store training symbol for later


% -------------------------------------------------------------------------
% 8) Initial per-subcarrier channel estimate from known OFDM symbol
% -------------------------------------------------------------------------
Z_known = Z(:,1);                  % column 1 = training OFDM symbol

H0 = Z_known ./ A_known;           % initial static channel estimate (per subcarrier)
debug.H = H0;

% -------------------------------------------------------------------------
% 9) Equalization + Channel / Phase tracking
% -------------------------------------------------------------------------
Ndata_sym_eff = Nsym - 1;          % how many data OFDM symbols we actually have

if Ndata_sym_eff <= 0
    warning('rxofdm:noData', 'No data OFDM symbols detected. Returning zeros.');
    return;
end

rx_data = Z(:, 2:Nsym);            % [Ncar x Ndata_sym_eff]

% Allocate equalized symbols
eq_syms = zeros(Ncar, Ndata_sym_eff);

% Phase/Tracking mode:
%   conf.phase.avg == 0 : Joelle-style per-subcarrier decision-directed tracking (default)
%   conf.phase.avg == 1 : CPE Viterbi–Viterbi tracking (your CPE idea)
if conf.phase.avg == 1
    % -------------------------------------------------------------
    % (A) Common Phase Error (CPE) Viterbi–Viterbi tracking
    % -------------------------------------------------------------
    theta_hat  = 0;                        % scalar CPE estimate
    alpha_cpe  = conf.phase.alpha_cpe;     % e.g. 0.95–0.999
    theta_hist = zeros(1, Ndata_sym_eff);

    for k = 1:Ndata_sym_eff
        % 1) Equalize using static H0 (amplitude + initial phase)
        sym_static = rx_data(:,k) ./ H0;   % [Ncar x 1]

        % 2) Remove current CPE estimate (feedback)
        sym_corr = sym_static .* exp(-1j * theta_hat);

        % 3) Viterbi–Viterbi estimate of residual phase
        %    NOTE: minus sign compensates for Gray QPSK at ±π/4
        vv_sum   = sum(-(sym_corr).^4);    % multiply by -1 AFTER ^4
        delta_th = 0.25 * angle(vv_sum);   % small residual increment

        % 4) LPF update
        theta_hat = theta_hat + alpha_cpe * delta_th;
        theta_hat = mod(theta_hat + pi, 2*pi) - pi;

        theta_hist(k) = theta_hat;

        % 5) Final correction
        eq_syms(:,k) = sym_static .* exp(-1j * theta_hat);
    end

    debug.theta_hat = theta_hist;

elseif conf.phase.avg == 2
    % -------------------------------------------------------------
    % (B) Per-subcarrier Viterbi–Viterbi + LPF tracking
    %     (vector theta_hat, one per subcarrier)
    % -------------------------------------------------------------
    H = H0;                                 % static LS estimate per subcarrier
    eq_syms   = zeros(Ncar, Ndata_sym_eff);
    theta_hat = zeros(Ncar, 1);             % current phase per subcarrier
    theta_hist = zeros(Ncar, Ndata_sym_eff);

    if isfield(conf, 'phase') && conf.phase.tracking
        alpha = conf.phase.lpf_factor;      % e.g. 0.9–0.999

        for k = 1:Ndata_sym_eff
            % A. Equalize using static H (Amplitude + Initial Phase)
            sym_static = rx_data(:, k) ./ H;     % [Ncar x 1]

            % B. Correct using PREVIOUS estimate
            sym_corr = sym_static .* exp(-1j * theta_hat);   % [Ncar x 1]

            % C. Viterbi–Viterbi residual phase per subcarrier
            vv_vec   = -(sym_corr.^4);                       % [Ncar x 1]
            delta_th = 0.25 * angle(vv_vec);              % [Ncar x 1]

            % D. LPF update
            theta_hat = theta_hat + alpha * delta_th;
            theta_hat = mod(theta_hat + pi, 2*pi) - pi;   % wrap per subcarrier

            % Store for plotting
            theta_hist(:, k) = theta_hat;

            % E. Apply final correction
            eq_syms(:, k) = sym_static .* exp(-1j * theta_hat);
        end
    else
        % No tracking → pure static equalization
        eq_syms    = rx_data ./ repmat(H, 1, Ndata_sym_eff);
        theta_hist = zeros(Ncar, Ndata_sym_eff);
    end

    debug.theta_hat = theta_hist;     % [Ncar x Ndata_sym_eff]

    
else
    % -------------------------------------------------------------
    % (B) Per-subcarrier decision-directed tracking (Joelle)
    % -------------------------------------------------------------
    H_hat  = zeros(Ncar, Nsym);
    H_hat(:,1) = H0;


    for s = 2:Nsym
        Z_s   = Z(:,s);            % received subcarriers for symbol s
        H_prev = H_hat(:,s-1);     % previous channel estimate (per subcarrier)

        % One-tap equalization with previous estimate
        tmp_eq = Z_s ./ H_prev;

        % Hard decision to nearest QPSK Gray symbol
        Lsym  = length(tmp_eq);
        cand  = repmat(tmp_eq, 1, M);     % [Ncar x 4]
        grayM = repmat(GrayMap, Lsym, 1); % [Ncar x 4]
        D     = abs(cand - grayM).^2;
        [~, ind] = min(D, [], 2);
        a_hat = GrayMap(ind).';           % [Ncar x 1]

        % Decision-directed channel update per subcarrier
        H_new_est = Z_s ./ a_hat;
        H_hat(:,s) = conf.phase.alpha_dd * H_prev + (1-conf.phase.alpha_dd) * H_new_est;

        % Final equalization using updated channel for this symbol
        eq_syms(:,s-1) = Z_s ./ H_hat(:,s);
    end

    % For consistency in debug, we can store a per-subcarrier phase history
    debug.theta_hat = angle(H_hat(:,1:end)); % crude per-subcarrier phase evolution wrapped between -pi and pi
end

% Save equalized constellation for plotting
debug.rx_constellation = eq_syms(:);
debug.tx_constellation = GrayMap(:);
debug.rx_baseband      = rx_bb;
debug.rx_filtered      = rx_bb_filt;

% After phase tracking / equalization, build per-symbol channel matrix
% Ndata_sym_eff is the number of data OFDM symbols (excluding the known one)

switch conf.phase.avg
    case 1  % CPE-only VV: channel = H0 * e^{j theta_hat(k)}
        % theta_hist is 1 x Ndata_sym_eff
        H_eff = H0 .* exp(1j * theta_hist);         % [Ncar x Ndata_sym_eff]

    case 2  % per-subcarrier VV: channel = H .* e^{j theta_hat(:,k)}
        % theta_hist is Ncar x Ndata_sym_eff
        H_eff = H .* exp(1j * theta_hist);          % [Ncar x Ndata_sym_eff]

    otherwise  % 0 → DD tracking (Joelle)
        % H_hat is Ncar x Nsym, and we used symbols 2..Nsym for data
        H_eff = H_hat(:, 2:Nsym);                   % [Ncar x Ndata_sym_eff]
end

debug.H0   = H0;        % keep initial estimate
debug.Heff = H_eff;     % time-varying effective channel
debug.eq   = eq_syms;   % store equalized symbols for later SNR plot

% -------------------------------------------------------------------------
% 10) QPSK Gray demapping → bits
% -------------------------------------------------------------------------
% Flatten equalized symbols column-wise
eq_vec = eq_syms(:);              % complex vector, length Ncar * Ndata_sym_eff

Lbitsym = length(eq_vec);

% Distances to GrayMap for demapper
cand2  = repmat(eq_vec, 1, M);    % [Lbitsym x 4]
grayM2 = repmat(GrayMap, Lbitsym, 1);
D2     = abs(cand2 - grayM2).^2;
[~,ind2] = min(D2, [], 2);

demappedGray = de2bi(ind2-1, 2, 'left-msb');  % [Lbitsym x 2]

% Serialize bits
rxbit_vec = demappedGray.';       % [2 x Lbitsym]
rxbits_all = rxbit_vec(:);        % column vector

% Trim or pad to conf.nbits just in case
if length(rxbits_all) < conf.nbits
    warning('rxofdm:bitLength', ...
        'Received fewer bits (%d) than conf.nbits (%d).', length(rxbits_all), conf.nbits);
    rxbits = [rxbits_all; zeros(conf.nbits - length(rxbits_all),1)];
else
    rxbits = rxbits_all(1:conf.nbits);
end

% -------------------------------------------------------------------------
% Physical Channel Impulse Response at 48 kHz (linear CIR)
%   - Uses known OFDM training symbol in time domain
%   - Deconvolution in frequency domain: H = R/S
% -------------------------------------------------------------------------
if nargout >= 3 && isfield(debug, 'A_known') && exist('ofdm_48k', 'var')

    Ncar    = conf.ofdm.ncarrier;
    Ncp     = conf.ofdm.cplen;
    Nfft    = Ncar;
    fs_phys = conf.f_s;          % 48 kHz (physical baseband sampling rate)

    % 1) Rebuild known OFDM symbol in time at fs_ofdm = conf.ofdm.bandwidth
    A_known = debug.A_known;     % [Ncar x 1]
    sym_td  = ifft(A_known, Nfft);               % [Nfft x 1]
    sym_cp  = [sym_td(end-Ncp+1:end); sym_td];   % add CP → [Nfft+Ncp x 1]

    % Normalize like TX payload (unit average power)
    sym_cp = sym_cp / sqrt(mean(abs(sym_cp).^2));

    % 2) Up-sample this training symbol to 48 kHz exactly like TX
    sym_48k = ofdm_tx_resampling(sym_cp, conf);  % complex baseband @ 48 kHz

    % 3) Extract the corresponding received segment at 48 kHz
    %    ofdm_48k starts at the first OFDM sample after frame sync
    Ltrain = length(sym_48k);
    if length(ofdm_48k) < Ltrain
        Ltrain = length(ofdm_48k);      % safety
        sym_48k = sym_48k(1:Ltrain);
    end
    rx_train_48k = ofdm_48k(1:Ltrain);

    % 4) Linear channel via frequency-domain LS deconvolution
    Lfft = 2^nextpow2(2*Ltrain - 1);      % length for (approx) linear conv
    S_tx = fft(sym_48k, Lfft);
    S_rx = fft(rx_train_48k, Lfft);

    H_lin = S_rx ./ (S_tx + 1e-12);       % avoid division by zero
    h_lin = ifft(H_lin);                  % linear CIR @ 48 kHz

    % 5) Shift so that main tap is at index 1 (for nicer plotting)
    [~, idx0] = max(abs(h_lin));
    h_lin = circshift(h_lin, 1 - idx0);

    % Store in debug
    debug.h_phys      = h_lin;
    debug.fs_phys     = fs_phys;
end


end
