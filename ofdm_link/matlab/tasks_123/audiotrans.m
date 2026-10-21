% % % % %
% Wireless Receivers: algorithms and architectures
% Audio Transmission Framework 

clear all;
close all;
% clc;

% Configuration Values

% Options for transmission are : 
% emulator: use a channel emulator with 5 different configurations.
% audio: use the loudspeaker and microphone for the data transmission
conf.audiosystem = 'emulator'; 

% Emulator configuration
% Use Channel ID 2 to test Phase Tracking (Oscillator Mismatch)
conf.emulator_idx = 3;
conf.emulator_snr = 100;

% General parameters 
conf.nbits   = 512*2*50;
conf.f_c     = 8000;

% Preamble
conf.sc.f_sym = 1000;  % symbol rate
conf.sc.nsyms = 500;

% OFDM
conf.ofdm.bandwidth = 2000; 
conf.ofdm.ncarrier  = 512;
conf.ofdm.cplen     = 256;
conf.modulation_order = 2; % QPSK
conf.constellation = [1+1j -1+1j 1-1j -1-1j] * 1/sqrt(2);
conf.ofdm.nsym = conf.nbits / (conf.ofdm.ncarrier * conf.modulation_order);


% Phase Tracking Configuration
conf.phase.tracking   = false;

% Default mode:
%   0 → per-subcarrier DD tracking (Joelle)
%   1 → CPE-only Viterbi–Viterbi (global)
%   2 → per-subcarrier Viterbi–Viterbi
conf.phase.avg        = 2;      

% Filter / smoothing factors (to be tuned per channel)
conf.phase.alpha_dd   = 0;   % smoothing for DD channel tracking (Joelle)
conf.phase.alpha_cpe  = 0.84;   % smoothing for global CPE tracker
conf.phase.lpf_factor = 0.01;   % smoothing for per-subcarrier VV (avg == 2)

% Audio settings 
conf.f_s = 48000;
conf.bitsps = 16; 

% Pre-calculations
conf.ofdm.spacing  = conf.ofdm.bandwidth/conf.ofdm.ncarrier;
conf.sc.os_factor  = conf.f_s/conf.sc.f_sym;
conf.ofdm.os_factor = conf.f_s/(conf.ofdm.ncarrier*conf.ofdm.spacing);

% Pre-generate Preamble Pulse
conf.sc.txpulse_length = 20*conf.sc.os_factor;
conf.sc.txpulse    = rrc(conf.sc.os_factor, 0.22, conf.sc.txpulse_length);

disp('Start OFDM Transmission')

% Generate Data
txbits = randi([0 1], conf.nbits, 1);

% Transmit
[txsignal, conf] = txofdm(txbits, conf);

rawtxsignal = [ zeros(conf.f_s,1) ; txsignal ; zeros(conf.f_s,1) ];
rawtxsignal = [ rawtxsignal zeros(size(rawtxsignal)) ];

% Transmission Mode
switch(conf.audiosystem)
    case 'emulator'
        rxsignal = channel_emulator(rawtxsignal(:,1), conf);
    case 'audio'
        txdur = length(rawtxsignal)/conf.f_s;
        audiowrite('out.wav', rawtxsignal, conf.f_s);
        playobj = audioplayer(rawtxsignal, conf.f_s, conf.bitsps);
        recobj  = audiorecorder(conf.f_s, conf.bitsps, 1);
        record(recobj);
        pause(2);
        disp('Recording...');
        playblocking(playobj);
        pause(2);
        stop(recobj);
        disp('Recording ended');
        rawrxsignal = getaudiodata(recobj, 'int16');
        rawrxsignal = double(rawrxsignal(1:end))/double(intmax('int16'));
        rxsignal = rawrxsignal;
end
     
% Receive
[rxbits, conf, debug] = rxofdm(rxsignal, conf);

% Calculate BER
res.biterrors = sum(rxbits ~= txbits);
ber = res.biterrors/length(rxbits);
fprintf('BER: %.4f\n', ber);

% Calculate the total number of correct symbols received
total_symbols = length(rxbits) / conf.modulation_order;
correct_symbols = total_symbols - res.biterrors / conf.modulation_order;
fprintf('Total Correct Symbols: %.0f\n', correct_symbols);
fprintf('Total Number of Symbols: %.0f\n', conf.nbits/conf.modulation_order);

%% VISUALIZATION AND VERIFICATION
% 1. Preamble chain: matched filter output + correlation metric
if exist('debug', 'var') && isfield(debug, 'fs') && isfield(debug, 'rx_preamble_mf')
    figure('Name', 'Preamble & Frame Sync');

    % a) Correlation metric T(i)
    subplot(2,1,1);
    T_vals   = debug.fs.T_vals;
    n_start  = debug.fs.n_start;
    idx_peak = debug.fs.idx_peak;
    thr      = debug.fs.threshold;

    n_axis = (n_start : n_start + numel(T_vals)-1);
    plot(n_axis, T_vals, 'LineWidth', 1.0); hold on;
    yline(thr, '--', 'Threshold');
    xline(n_start + idx_peak - 1, 'r--', 'Max peak');
    hold off;
    xlabel('Sample index (48 kHz)');
    ylabel('T(i)');
    title('Normalized frame-sync metric T(i)');
    grid on;

    % b) Matched-filter output magnitude around the detected index
    subplot(2,1,2);
    mf = debug.rx_preamble_mf;
    idx0 = n_start + idx_peak - 1;
    win  = 2000;   % samples around detection to show (adjust if you want)
    lo   = max(1, idx0 - win);
    hi   = min(length(mf), idx0 + win);

    plot(lo:hi, abs(mf(lo:hi)), 'LineWidth', 1.0); hold on;
    xline(idx0, 'r--', 'Start of OFDM');
    hold off;
    xlabel('Sample index (48 kHz)');
    ylabel('|Matched-filter output|');
    title('RRC matched-filter output around sync point');
    grid on;
end

% 2. Phase Tracking Plot (New for Phase 3)
if exist('debug', 'var') && isfield(debug, 'theta_hat')
    figure('Name', 'Phase Drift and Tracking Evolution');
    
    % Subplot for Accumulated Common Phase Correction
    subplot(2,1,1);
    if conf.phase.avg == 1
        % CPE scalar per symbol
        plot(debug.theta_hat, 'LineWidth', 1.5);
        title('Accumulated Common Phase Correction \theta(k)');
        xlabel('Symbol Index'); ylabel('Phase [rad]');
        grid on;
    else
        % Per-subcarrier phase history
        sc_indices = round(linspace(1, conf.ofdm.ncarrier, 4));
        plot(debug.theta_hat(sc_indices, :).');
        title('Per-subcarrier Accumulated Phase Correction \theta(k)');
        xlabel('Symbol Index'); ylabel('Accumulated Phase Correction [rad]');
        xlim([0, 50]);
        legend(strcat('Subcarrier ', string(sc_indices)));
        grid on;
    end
    
    % Subplot for Symbol-to-Symbol Phase Correction
    subplot(2,1,2);
    if conf.phase.avg == 1
        % CPE scalar per symbol
        plot(mod((debug.theta_hat(2:end)-debug.theta_hat(1:end-1) + pi), 2*pi) - pi, 'LineWidth', 1.5);
        title('Symbol-to-Symbol Phase Correction \Delta\theta(k)');
        xlabel('Symbol Index'); ylabel('Phase [rad]');
        grid on;
    else
        % Per-subcarrier phase history
        sc_indices = round(linspace(1, conf.ofdm.ncarrier, 4));
        plot((mod((debug.theta_hat(sc_indices, 2:end)-debug.theta_hat(sc_indices, 1:end-1) + pi), 2*pi) - pi).');
        title('Per-subcarrier Symbol-to-Symbol Phase Correction \Delta\theta(k)');
        xlabel('Symbol Index'); ylabel('Phase Difference [rad]');
        legend(strcat('Subcarrier ', string(sc_indices)));
        grid on;
    end
end


% 3. Constellation Plot (The "Clouds")
if exist('debug', 'var')
    figure('Name', 'Constellation Diagram');
    subplot(1,2,1);
    plot(real(debug.rx_data_raw), imag(debug.rx_data_raw), '.');
    title('Unequalized Rx Symbols');
    grid on; axis square;
    xlabel('In-Phase'); ylabel('Quadrature');
    xlim([-1.5 1.5]); ylim([-1.5 1.5]);

    subplot(1,2,2);
    hold on;
    % Plot received symbols
    plot(real(debug.rx_constellation), imag(debug.rx_constellation), 'b.','MarkerSize', 2, 'LineWidth', 2);
    % Plot ideal constellation points
    plot(real(conf.constellation), imag(conf.constellation), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;
    title('Equalized Rx Symbols (QPSK)');
    legend('Rx Symbols', 'Ideal QPSK');
    grid on; axis square;
    xlabel('In-Phase'); ylabel('Quadrature');
    xlim([-1.5 1.5]); ylim([-1.5 1.5]);
end

% 4. Channel Estimate Magnitude
if exist('debug', 'var')
    figure('Name', 'Channel Estimate');
    plot(20*log10(abs(debug.H)));
    title('Channel Frequency Response Magnitude');
    xlabel('Subcarrier Index');
    ylabel('Magnitude (dB)');
    grid on;
end

% 5. Channel Phase
if exist('debug', 'var') && isfield(debug, 'H')
    figure('Name', 'Channel Phase');
    plot(unwrap(angle(debug.H)));
    title('Channel Frequency Response Phase');
    xlabel('Subcarrier Index');
    ylabel('Phase [rad]');
    grid on;
end

% -------------------------------------------------------------------------
% Channel Impulse Response (CIR) and Power Delay Profile (PDP)
% -------------------------------------------------------------------------
if exist('debug', 'var') && isfield(debug, 'H')
    H0    = debug.H(:);                 % [Nfft x 1] complex FR estimate
    Nfft  = conf.ofdm.ncarrier;         % 512
    Ncp   = conf.ofdm.cplen;            % 256

    % 1) Estimate CIR via IFFT of the frequency response
    %    (H0 is in the same bin order as the tx IFFT, so plain ifft is OK)
    h_est = ifft(H0, Nfft);             % [Nfft x 1]

    % Time axis in seconds (or ms) for plotting
    n_taps = 1*Ncp;                       % show only first CP-length taps
    n      = 0:n_taps-1;
    tau    = n / conf.f_s;               % delay in seconds
    tau_ms = 1e3 * tau;                 % delay in ms (nicer axis)

    % 2) Channel Impulse Response
    figure('Name', 'Channel Impulse Response');
    stem(tau_ms, abs(h_est(1:n_taps)), 'filled');
    xlabel('Delay [ms]');
    ylabel('|h[n]|');
    title('Estimated Channel Impulse Response');
    grid on;

    % 3) Power Delay Profile
    figure('Name', 'Power Delay Profile');
    stem(tau_ms, abs(h_est(1:n_taps)).^2, 'filled');
    xlabel('Delay [ms]');
    ylabel('Power');
    title('Estimated Power Delay Profile');
    grid on;
end

% -------------------------------------------------------------------------
% Physical CIR & PDP at 48 kHz (true echo delays)
% -------------------------------------------------------------------------
if exist('debug', 'var') && isfield(debug, 'h_phys')
    h_lin  = debug.h_phys;
    fs_phys = debug.fs_phys;   % 48 kHz

    N       = length(h_lin);
    n       = 0:N-1;
    tau_sec = n / fs_phys;
    tau_ms  = 1e3 * tau_sec;

    % CP duration and multiples in ms (for reference)
    Tcp_sec = conf.ofdm.cplen / conf.ofdm.bandwidth;  % same CP as before
    Tcp_ms  = 1e3 * Tcp_sec;
    T2cp_ms = 2 * Tcp_ms;

    % 1) Physical CIR
    figure('Name','Physical Channel Impulse Response (48 kHz)');
    stem(tau_ms, abs(h_lin), 'filled'); hold on;
    xline(Tcp_ms,  'r--', sprintf('CP = %.2f ms',  Tcp_ms), 'LineWidth', 1.2);
    xline(T2cp_ms, 'm--', sprintf('2CP = %.2f ms', T2cp_ms), 'LineWidth', 1.2);
    hold off;
    xlabel('Delay [ms]');
    ylabel('|h[n]|');
    title('Estimated Physical CIR (linear, 48 kHz)');
    grid on;

    % 2) Physical PDP
    figure('Name','Physical Power Delay Profile (48 kHz)');
    stem(tau_ms, abs(h_lin).^2, 'filled'); hold on;
    xline(Tcp_ms,  'r--', sprintf('CP = %.2f ms',  Tcp_ms), 'LineWidth', 1.2);
    xline(T2cp_ms, 'm--', sprintf('2CP = %.2f ms', T2cp_ms), 'LineWidth', 1.2);
    hold off;
    xlabel('Delay [ms]');
    ylabel('Power');
    title('Estimated Physical PDP (linear, 48 kHz)');
    grid on;
end

% -------------------------------------------------------------------------
% Time-varying CIR (waterfall) from per-symbol Heff
% -------------------------------------------------------------------------
if exist('debug', 'var') && isfield(debug, 'Heff')
    H_eff   = debug.Heff;                  % [Ncar x Nsym_data]
    [Nfft, Nsym_data] = size(H_eff);
    Ncp     = conf.ofdm.cplen;
    fs_ofdm = conf.ofdm.bandwidth;        % 2000 Hz

    % IFFT per symbol -> h_s[n]
    h_all = ifft(H_eff, Nfft, 1);         % [Nfft x Nsym_data]

    % Use only first Ncp taps (within CP)
    n_taps = Ncp;
    tau    = (0:n_taps-1)/fs_ofdm;        % delay (s)
    tau_ms = 1e3 * tau;

    figure('Name','Time-varying CIR');
    imagesc(1:Nsym_data, tau_ms, abs(h_all(1:n_taps,:)));
    axis xy;
    xlabel('OFDM data symbol index');
    ylabel('Delay [ms]');
    title('Time-varying Channel Impulse Response |h(n,k)|');
    colorbar;
end

% -------------------------------------------------------------------------
% Per-subcarrier SNR estimate from equalized symbols
% -------------------------------------------------------------------------
if exist('debug', 'var') && isfield(debug, 'eq')
    eq_syms = debug.eq;                  % [Ncar x Nsym_data]
    [Ncar, Nsym_data] = size(eq_syms);

    % Recreate QPSK Gray map (same as in rxofdm)
    M      = 4;
    GrayMap = (1/sqrt(2)) * [ ...
        +1 + 1j;   % 00
        -1 + 1j;   % 01
        -1 - 1j;   % 11
        +1 - 1j];  % 10

    % Hard decisions for each equalized symbol
    Lsym  = Ncar * Nsym_data;
    cand  = repmat(eq_syms(:), 1, M);        % [Ncar*Nsym x 4]
    grayM = repmat(GrayMap.', Lsym, 1);      % same size
    D     = abs(cand - grayM).^2;
    [~, idx] = min(D, [], 2);
    a_hat_vec = GrayMap(idx);                % [Ncar*Nsym x 1]
    a_hat = reshape(a_hat_vec, Ncar, Nsym_data);

    % Signal and noise powers per subcarrier
    sig_pow  = mean(abs(a_hat).^2, 2);            % ≈ 1
    noise_pow = mean(abs(eq_syms - a_hat).^2, 2) + eps;

    SNR_lin  = sig_pow ./ noise_pow;
    SNR_dB   = 10*log10(SNR_lin);

    figure('Name','Per-subcarrier SNR');
    plot(1:Ncar, SNR_dB, '.-');
    xlabel('Subcarrier index');
    ylabel('SNR [dB]');
    title('Estimated Per-subcarrier SNR');
    grid on;
end


