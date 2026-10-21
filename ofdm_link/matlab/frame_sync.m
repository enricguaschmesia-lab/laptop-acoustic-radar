function [beginning_of_data, fs_debug] = frame_sync(rx_signal, conf)
%{
FRAME_SYNC  Detect the start of the OFDM payload using a known BPSK preamble.

[beginning_of_data, fs_debug] = FRAME_SYNC(rx_signal, conf) performs frame synchronization by
computing a normalized correlation metric between the received (matched-filtered) signal and a
known BPSK preamble generated via lfsr_framesync(). The metric is evaluated over all possible
alignment positions using one sample per symbol (step = conf.sc.os_factor).

The function selects the global maximum of the metric as the detected preamble end and returns the
corresponding sample index (beginning_of_data) for the first payload sample after the preamble.
It also computes a dynamic detection threshold (noise-floor based and peak-relative) and returns
debug information useful for plotting.

Inputs:
  - rx_signal : complex baseband signal (typically after matched filtering)
  - conf      : configuration struct (uses conf.sc.nsyms and conf.sc.os_factor)

Outputs:
  - beginning_of_data : sample index (in rx_signal) corresponding to the detected payload start
  - fs_debug          : struct with fields for visualization (T_vals, n_start, idx_peak, peak_val, threshold)
%}

    if (rx_signal(1) == 0)
        warning('Signal appears to be noise-free. The frame synchronizer may not function correctly.');
    end

    % --- Parameters from configuration ---
    frame_sync_length = conf.sc.nsyms;   % Number of BPSK preamble symbols
    L                 = conf.sc.os_factor;  % Oversampling factor

    % BPSK frame synchronization sequence: 0 maps to +1, 1 maps to -1
    frame_sync_sequence = 1 - 2 * lfsr_framesync(frame_sync_length);

    % --- Compute normalized correlation metric T(i) over the entire signal ---
    n_start = L * frame_sync_length + 1;
    n_end   = length(rx_signal);

    if n_end <= n_start
        beginning_of_data = -1;
        warning('Frame sync: received signal is too short.');
        return;
    end

    num_positions = n_end - n_start + 1;
    T_vals = zeros(num_positions, 1);

    for idx = 1:num_positions
        i = n_start + idx - 1;

        % Take one sample per symbol (oversampled by L)
        r = rx_signal(i - L * frame_sync_length : L : i - L);

        % Correlation with the known preamble
        c = frame_sync_sequence' * r;

        % Normalized correlation metric (robust against amplitude variations)
        denom = abs(r' * r);
        if denom < eps
            T_vals(idx) = 0;
        else
            T_vals(idx) = abs(c)^2 / denom;
        end
    end

    % --- Design dynamic threshold ---
    
    % 1) Identify the global peak of the correlation metric
    [peak_val, idx_peak] = max(T_vals);

    % 2) Estimate noise floor from the first 20% of positions
    noise_len = max(10, round(0.2 * num_positions));   % Ensure at least 10 samples
    noise_region = T_vals(1:noise_len);
    mu_noise     = mean(noise_region);
    sigma_noise  = std(noise_region);

    % Noise-based threshold: a few standard deviations above noise
    thr_noise = mu_noise + 8 * sigma_noise;

    % Relative threshold: a fraction of the peak value
    thr_rel = 0.4 * peak_val;

    % Final dynamic threshold: maximum of noise-based and relative-to-peak thresholds
    detection_threshold = max(thr_noise, thr_rel);
    %disp(detection_threshold)

    % --- Determine the beginning of data at the global maximum of T ---
    beginning_of_data = n_start + idx_peak - 1;

     % --- Debug info for plotting ---
    fs_debug = struct();
    fs_debug.T_vals      = T_vals;
    fs_debug.n_start     = n_start;
    fs_debug.idx_peak    = idx_peak;
    fs_debug.peak_val    = peak_val;
    fs_debug.threshold   = detection_threshold;

end
