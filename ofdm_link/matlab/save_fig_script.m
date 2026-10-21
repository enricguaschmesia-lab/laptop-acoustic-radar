%{
SCRIPT: Single-run OFDM transmission + diagnostics + PDP/CIR estimation.

This script runs one end-to-end OFDM transmission (random QPSK payload over either the audio path
or the provided channel emulator), computes the BER, and saves a set of diagnostic figures to
fig_out_dir (frame-sync metric, phase-tracking evolution, constellations, and channel magnitude/
phase).

In addition, it estimates the channel Power Delay Profile (PDP) by repeating the transmission over
multiple independent realizations (Nreal_pdp), extracting the frequency-domain channel estimate
from the receiver debug output, transforming it to the time domain via IFFT, accumulating |h[n]|^2,
and plotting/saving both averaged and single-realization CIR/PDP (in samples and in milliseconds).

Outputs:
  - BER printed to console (single run)
  - Multiple .png/.fig figures saved under fig_out_dir (PDP/CIR + receiver diagnostics)
%}

clear 
close all
clc

% Directory where all figures will be saved

fig_out_dir = 'task_ADD';   % <<< change for every channel or scenario PLEASE!
if ~exist(fig_out_dir, 'dir')
    mkdir(fig_out_dir);
end

% Configuration Values

% Options for transmission are : 
% emulator: use a channel emulator with 5 different configurations.
% audio: use the loudspeaker and microphone for the data transmission
conf.audiosystem = 'audio'; 

% Emulator configuration
conf.emulator_idx = 5;    % 1 to 5 yields different channels
conf.emulator_snr = 100;  % we want to see the channel not noise

% General parameters 
conf.nbits   = 512*2*10;  % number of bits 
conf.f_c     = 8000;      % center carrier frequency [Hz]

%chanel parameter
conf.channel_tracking = 1 ; %1 if we want tracking, 0 otherwise
                      

% Channel/phase tracking: (only works if conf.channel_tracking is set to 1)
%   0 → channel tracking (amplitude and phase)
%   1 → CPE-only Viterbi–Viterbi (global)
%   2 → per-subcarrier Viterbi–Viterbi
conf.phase.avg        = 0; 

% Filter / smoothing factors (to be tuned per channel)
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


disp('Start OFDM Transmission (single run)')

% Generate random data
txbits = randi([0 1],conf.nbits,1);

% Transmit 
[txsignal, conf] = txofdm(txbits,conf);

rawtxsignal = [ zeros(conf.f_s,1) ; txsignal ; zeros(conf.f_s,1) ];
rawtxsignal = [  rawtxsignal  zeros(size(rawtxsignal)) ];

% MATLAB audio/emulator mode
switch(conf.audiosystem)
    case 'emulator'
        rxsignal = channel_emulator(rawtxsignal(:,1),conf);
    case 'audio'
        % % % % % % % % % % % % %
        % Begin
        % Audio Transmission    

        txdur       = length(rawtxsignal)/conf.f_s; % calculate length of transmitted signal
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

        %
        % End
        % Audio Transmission   
        % % % % % % % % % % % %
end

% Receive Function
[rxbits, conf, debug] = rxofdm(rxsignal,conf);

res.biterrors    = sum(rxbits ~= txbits);
ber = res.biterrors/length(rxbits);
fprintf('BER (single run): %.4f\n', ber);

% PDP ESTIMATION OVER MANY REALIZATIONS AT FIXED SNR (EMULATOR)

% 2) PDP estimation loop (fixed SNR, many realizations)

Nreal_pdp     = 1;                          % number of realizations to average
Nfft_pdp      = conf.ofdm.ncarrier;
pdp_accum     = zeros(Nfft_pdp, 1);          % accumulate |h[n]|^2
pdp_num_runs  = 0;

% Will store one single-realization channel estimate for inspection
H_one_real   = [];
h_one_real   = [];
pdp_one_real = [];

fprintf('Estimating PDP at SNR = %d dB over %d realizations...\n', ...
    conf.emulator_snr, Nreal_pdp);

for r = 1:Nreal_pdp
    % New random data each time
    txbits_pdp = randi([0 1],conf.nbits,1);
    conf.nbits = length(txbits_pdp);

    % Transmit
    [txsignal_pdp, conf] = txofdm(txbits_pdp,conf);
    rawtxsignal_pdp = [ zeros(conf.f_s,1) ; txsignal_pdp ; zeros(conf.f_s,1) ];
    rawtxsignal_pdp = [  rawtxsignal_pdp  zeros(size(rawtxsignal_pdp)) ];

    % Channel
    switch(conf.audiosystem)
        case 'emulator'
            rxsignal_pdp = channel_emulator(rawtxsignal_pdp(:,1),conf);
        case 'audio'
            txdur       = length(rawtxsignal_pdp)/conf.f_s;
            audiowrite('out.wav',rawtxsignal_pdp,conf.f_s)
            playobj = audioplayer(rawtxsignal_pdp,conf.f_s,conf.bitsps);
            recobj  = audiorecorder(conf.f_s,conf.bitsps,1);
            record(recobj);
            pause(2);
            playblocking(playobj)
            pause(2);
            stop(recobj);
            rawrxsignal_pdp  = getaudiodata(recobj,'int16');
            rawrxsignal_pdp  = double(rawrxsignal_pdp(1:end))/double(intmax('int16')) ;
            rxsignal_pdp = rawrxsignal_pdp;
    end

    % Receive to get channel estimate
    [~, conf, debug_pdp] = rxofdm(rxsignal_pdp,conf);

    % Extract H and compute |h[n]|^2
    if exist('debug_pdp','var') && isfield(debug_pdp,'H')
        H0 = debug_pdp.H(:);              % frequency response (512 x 1)

       

        h_est = ifft(H0, Nfft_pdp);       % CIR estimate

        pdp_accum    = pdp_accum + abs(h_est).^2;
        pdp_num_runs = pdp_num_runs + 1;

        % save one realization for inspection
        if isempty(H_one_real)
            H_one_real   = H0;
            h_one_real   = h_est;
            pdp_one_real = abs(h_est).^2;
        end
    else
        warning('No debug_pdp.H found in realization %d, skipping.', r);
    end
end

% 3) Average PDP + single realization plotting

if pdp_num_runs > 0
    fs_ofdm = conf.ofdm.bandwidth;      % OFDM baseband sampling rate (Hz)
    n       = 0:Nfft_pdp-1;
    tau     = n / fs_ofdm;              % delay in seconds
    tau_ms  = 1e3 * tau;                % delay in ms

    % Average PDP over all runs
    pdp_mean = pdp_accum / pdp_num_runs;

    Lplot = min(conf.ofdm.cplen, Nfft_pdp); % show first CP-length taps (256)

    %  Average Channel Impulse Response magnitude (samples) 
    figure('Name', 'Average Channel Impulse Response (samples)');
    stem(n(1:Lplot), sqrt(pdp_mean(1:Lplot)), 'filled');
    xlabel('Tap index n');
    ylabel('|h[n]| (avg)');
    title(sprintf('Average CIR magnitude (SNR = %d dB, %d runs)', ...
           conf.emulator_snr, pdp_num_runs));
    grid on;
    saveas(gcf, fullfile(fig_out_dir, 'avg_chan_undertable.fig'));
    print(gcf, fullfile(fig_out_dir, 'avg_chan_undertable.png'), '-dpng', '-r300'); 

    %  Average Power Delay Profile (time) 
    figure('Name', 'Average Power Delay Profile (time)');
    stem(tau_ms(1:Lplot), pdp_mean(1:Lplot), 'filled');
    xlabel('Delay [ms]');
    ylabel('Power');
    title(sprintf('PDP (SNR = %d dB, %d runs)', ...
           conf.emulator_snr, pdp_num_runs));
    grid on;
    saveas(gcf, fullfile(fig_out_dir, 'pdp_acoustics_tau_under_table.fig'));
    print(gcf, fullfile(fig_out_dir, 'pdp_acoustics_tau_under_table.png'), '-dpng', '-r300');  

    %  Average CIR magnitude (time) 
    figure('Name', 'Average Channel Impulse Response (time)');
    stem(tau_ms(1:Lplot), sqrt(pdp_mean(1:Lplot)), 'filled');
    xlabel('Delay [ms]');
    ylabel('|h[n]| (avg)');
    title(sprintf('Average CIR magnitude (SNR = %d dB, %d runs)', ...
           conf.emulator_snr, pdp_num_runs));
    grid on;
    saveas(gcf, fullfile(fig_out_dir, 'avg_chan_tau_under_table.fig'));
    print(gcf, fullfile(fig_out_dir, 'avg_chan_tau_under_table.png'), '-dpng', '-r300');  

    %  Average Power Delay Profile (samples) 
    figure('Name', 'Average Power Delay Profile (samples)');
    stem(n(1:Lplot), pdp_mean(1:Lplot), 'filled');
    xlabel('Tap index n');
    ylabel('Power');
    title(sprintf('PDP (SNR = %d dB, %d runs)', ...
           conf.emulator_snr, pdp_num_runs));
    grid on;
    saveas(gcf, fullfile(fig_out_dir, 'pdp_acoustics.fig'));
    print(gcf, fullfile(fig_out_dir, 'pdp_acoustics.png'), '-dpng', '-r300'); 

    %  Single-realization CIR magnitude (time) 
    if ~isempty(h_one_real)
        figure('Name', 'Single-Realization CIR magnitude (time)');
        stem(tau_ms(1:Lplot), abs(h_one_real(1:Lplot)), 'filled');
        xlabel('Delay [ms]');
        ylabel('|h[n]|');
        title(sprintf('Single-Realization CIR magnitude (SNR = %d dB)', ...
               conf.emulator_snr));
        grid on;
        saveas(gcf, fullfile(fig_out_dir, 'single_chan_tau_under_table.fig'));
        print(gcf, fullfile(fig_out_dir, 'single_chan_tau_under_table.png'), '-dpng', '-r300');
    end

    %  Single-realization PDP (time) 
    if ~isempty(pdp_one_real)
        figure('Name', 'Single-Realization Power Delay Profile (time)');
        stem(tau_ms(1:Lplot), pdp_one_real(1:Lplot), 'filled');
        xlabel('Delay [ms]');
        ylabel('Power');
        title(sprintf('Single-Realization PDP (SNR = %d dB)', ...
               conf.emulator_snr));
        grid on;
        saveas(gcf, fullfile(fig_out_dir, 'pdp_single_tau_under_table.fig'));
        print(gcf, fullfile(fig_out_dir, 'pdp_single_tau_under_table.png'), '-dpng', '-r300');
    end    

    fprintf('PDP estimated from %d valid realizations.\n', pdp_num_runs);
else
    warning('No valid channel estimates collected → PDP not available.');
end




% VISUALIZATION AND VERIFICATION

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

% 2. Phase Tracking Plot
if exist('debug', 'var') && isfield(debug, 'theta_hat')
    figure('Name', 'Phase Tracking Evolution');

    % CASE 1: Common Phase Error VV
    if conf.phase.avg == 1
        % CPE scalar per symbol
        plot(debug.theta_hat, 'LineWidth', 1.5);
        title('Estimated Common Phase Error \theta(k)');
        xlabel('Symbol Index'); ylabel('Phase [rad]');
        grid on;

    % CASE 2: Per-subcarrier VV tracking (vector)
    elseif conf.phase.avg == 2
        sc_indices = round(linspace(1, conf.ofdm.ncarrier, 4));
        plot(debug.theta_hat(sc_indices, :).');
        title('Per-subcarrier Phase Evolution (VV)');
        xlabel('Symbol Index'); ylabel('Phase Difference [rad]');
        legend(strcat('Subcarrier ', string(sc_indices)));
        grid on;

    % CASE 3: Channel Tracking (avg=0)
    %  → plot BOTH phase drift AND magnitude drift
    elseif conf.phase.avg == 0
        % ----- Phase Drift Plot -----
        subplot(2,1,1)
        sc_indices = round(linspace(1, conf.ofdm.ncarrier, 4));
        plot(debug.theta_hat(sc_indices, :).');
        title('Per-subcarrier Phase Evolution (DD tracking)');
        xlabel('Symbol Index'); ylabel('Phase Difference [rad]');
        legend(strcat('Subcarrier ', string(sc_indices)));
        grid on;

        % ----- Magnitude Evolution -----
        subplot(2,1,2)

        if isfield(debug, 'H_mag')
            plot(debug.H_mag(sc_indices, :).');
            title('Per-subcarrier Magnitude Evolution |H(m,k)|');
            xlabel('Symbol Index'); ylabel('Magnitude');
            legend(strcat('Subcarrier ', string(sc_indices)));
            grid on;
        else
            text(0.1,0.5,'Magnitude history not stored','FontSize',12);
        end
    end
end


% 3. Constellation Plot 
if exist('debug', 'var')
    figure('Name', 'Constellation Diagram');
    subplot(1,2,1);
    plot(real(debug.rx_data_raw), imag(debug.rx_data_raw), '.');
    title('Unequalized Rx Symbols');
    grid on; axis square;
    xlabel('In-Phase'); ylabel('Quadrature');


    subplot(1,2,2);
    hold on;
    % Plot received symbols
    plot(real(debug.rx_constellation), imag(debug.rx_constellation), 'b.');
    % Plot ideal constellation points
    plot(real(conf.constellation), imag(conf.constellation), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
    hold off;
    title('Equalized Rx Symbols (QPSK)');
    legend('Rx Symbols', 'Ideal QPSK');
    grid on; axis square;
    xlabel('In-Phase'); ylabel('Quadrature');
    xlim([-1.5 1.5]); ylim([-1.5 1.5]);
    % (add save here if you want)
end


%% 3b) Unequalized constellation evolution (shows microphone-motion impact)
if exist('debug', 'var') && isfield(debug, 'rx_data_raw') && ~isempty(debug.rx_data_raw)

    Ncar  = conf.ofdm.ncarrier;
    xraw  = debug.rx_data_raw(:);

    % Reshape into [subcarrier x OFDM_symbol]
    Ns = floor(numel(xraw)/Ncar);
    xraw = xraw(1:Ns*Ncar);
    Xraw = reshape(xraw, Ncar, Ns);

    % ------------------------------------------------------------
    % (A) "Time-colored" unequalized constellation (each color = OFDM symbol index)
    % ------------------------------------------------------------
    sym_idx = repelem(1:Ns, Ncar).';  % one symbol index per constellation point

    figure('Name','Unequalized Constellation Evolution (colored by OFDM symbol)');
    scatter(real(xraw), imag(xraw), 8, sym_idx, 'filled'); hold on;
    plot(real(conf.constellation), imag(conf.constellation), 'kx', ...
         'MarkerSize', 10, 'LineWidth', 1.5);
    hold off; grid on; axis square;
    xlabel('In-Phase'); ylabel('Quadrature');
    title('Unequalized Rx constellation over time (color = OFDM symbol index)');
    cb = colorbar; cb.Label.String = 'OFDM symbol index';

    saveas(gcf, fullfile(fig_out_dir, 'uneq_constellation_time_color.png'));
    savefig(fullfile(fig_out_dir, 'uneq_constellation_time_color.fig'));

    % ------------------------------------------------------------
    % (B) Early / Middle / Late unequalized constellations (side-by-side)
    % ------------------------------------------------------------
    win = min(5, Ns);  % number of OFDM symbols per window (increase if you want)
    idx1 = 1:win;

    mid_start = max(1, round(Ns/2 - win/2));
    idx2 = mid_start : min(Ns, mid_start + win - 1);

    idx3 = max(1, Ns-win+1):Ns;

    figure('Name','Unequalized Constellations: early / middle / late');

    subplot(1,3,1);
    tmp = Xraw(:, idx1);
    plot(real(tmp(:)), imag(tmp(:)), '.'); hold on;
    plot(real(conf.constellation), imag(conf.constellation), 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);
    hold off; grid on; axis square;
    title(sprintf('Early (sym %d-%d)', idx1(1), idx1(end)));
    xlabel('I'); ylabel('Q');

    subplot(1,3,2);
    tmp = Xraw(:, idx2);
    plot(real(tmp(:)), imag(tmp(:)), '.'); hold on;
    plot(real(conf.constellation), imag(conf.constellation), 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);
    hold off; grid on; axis square;
    title(sprintf('Middle (sym %d-%d)', idx2(1), idx2(end)));
    xlabel('I'); ylabel('Q');

    subplot(1,3,3);
    tmp = Xraw(:, idx3);
    plot(real(tmp(:)), imag(tmp(:)), '.'); hold on;
    plot(real(conf.constellation), imag(conf.constellation), 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);
    hold off; grid on; axis square;
    title(sprintf('Late (sym %d-%d)', idx3(1), idx3(end)));
    xlabel('I'); ylabel('Q');

    % Link axes so scaling doesn’t “hide” drift across panels
    ax = findall(gcf,'Type','axes');
    linkaxes(ax,'xy');

    saveas(gcf, fullfile(fig_out_dir, 'uneq_constellation_early_mid_late.png'));
    savefig(fullfile(fig_out_dir, 'uneq_constellation_early_mid_late.fig'));
end


% 4. Channel Estimate Magnitude
if exist('debug', 'var') && isfield(debug, 'H')
    H_plot = debug.H;
    N_H    = size(H_plot,1);

    % Frequency axis from f_c - BW/2 to f_c + BW/2
    f_axis = linspace(conf.f_c - conf.ofdm.bandwidth/2, ...
                      conf.f_c + conf.ofdm.bandwidth/2, N_H).';

    figure('Name', 'Channel Estimate Magnitude');
    plot(f_axis, 20*log10(abs(H_plot)));
    title('Channel Frequency Response Magnitude');
    xlabel('Frequency [Hz]');
    ylabel('Magnitude (dB)');
    grid on;

    % Save figure
    saveas(gcf, fullfile(fig_out_dir, 'channel_estimate_mag.png'));   
    savefig(fullfile(fig_out_dir, 'channel_estimate_mag.fig'));    
end

% 5. Channel Phase 
if exist('debug', 'var') && isfield(debug, 'H')
    H_plot = debug.H;
    N_H    = size(H_plot,1);

    f_axis = linspace(conf.f_c - conf.ofdm.bandwidth/2, ...
                      conf.f_c + conf.ofdm.bandwidth/2, N_H).';

    figure('Name', 'Channel Estimate Phase');
    plot(f_axis, unwrap(angle(H_plot)));
    title('Channel Frequency Response Phase');
    xlabel('Frequency [Hz]');
    ylabel('Phase [rad]');
    grid on;

    % Save figure
    saveas(gcf, fullfile(fig_out_dir, 'channel_estimate_phase.png')); 
    savefig(fullfile(fig_out_dir, 'channel_estimate_phase.fig'));     
end
