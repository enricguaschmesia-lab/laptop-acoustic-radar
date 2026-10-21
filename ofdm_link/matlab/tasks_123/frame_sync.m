function [beginning_of_data, fs_debug] = frame_sync(rx_signal, conf)
% FRAME_SYNC  Synchronizes frames in a received signal using dynamic thresholding.
%
%   Inputs:
%       rx_signal:     Output of the RRC matched filter at fs = conf.f_s
%                      containing the BPSK preamble followed by data.
%       conf:          Configuration structure containing parameters for synchronization.
%                       conf.sc.nsyms: Number of BPSK preamble symbols.
%                       conf.sc.os_factor: Oversampling factor.
%
%   Outputs:
%       beginning_of_data: Index in rx_signal indicating the start of the first OFDM sample.

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
    disp(detection_threshold)

    % --- Sanity check using the dynamic threshold ---
    if peak_val < detection_threshold
        warning('Frame sync: peak value is not significantly above the dynamic threshold (SNR may be too low).');
    end

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
