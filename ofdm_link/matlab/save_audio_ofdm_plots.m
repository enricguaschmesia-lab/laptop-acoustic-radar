function save_audio_ofdm_plots(debug, conf, out_dir, tag)
%{
SAVE_AUDIO_OFDM_PLOTS  Save standard OFDM receiver diagnostics (channel, PDP, evolution, constellation).

save_audio_ofdm_plots(debug, conf, out_dir, tag) generates and saves a set of figures from the
receiver debug struct and configuration:
  - Channel magnitude and phase from the initial training estimate (H0) vs. frequency around f_c
  - Time-domain impulse response and PDP from H0 (first N_CP taps)
  - If per-symbol channel estimates are available (debug.H_hat): heatmaps of channel magnitude,
    normalized magnitude drift, phase drift, and PDP evolution over time
  - If equalized symbols are available (debug.rx_constellation): constellation scatter and RMS EVM

All figures are written as PNG files into out_dir. The optional string tag is appended to filenames
to distinguish different runs/scenarios.

Inputs:
  - debug   : struct containing fields like H0, H_hat, rx_constellation (as available)
  - conf    : configuration struct (OFDM params, carrier frequency, sampling rate, constellation)
  - out_dir : output folder for figures
  - tag     : (optional) filename suffix to label the run
%}

if nargin < 4, tag = ""; end
if ~exist(out_dir,'dir'); mkdir(out_dir); end

Ncar = conf.ofdm.ncarrier;
Ncp  = conf.ofdm.cplen;

% --- Robust OFDM baseband sampling rate ---
if isfield(conf.ofdm,'os_factor') && ~isempty(conf.ofdm.os_factor)
    f_ofdm = conf.f_s / conf.ofdm.os_factor;
else
    f_ofdm = conf.ofdm.bandwidth; % fallback
end

% Subcarrier spacing
if isfield(conf.ofdm,'spacing') && ~isempty(conf.ofdm.spacing)
    df = conf.ofdm.spacing;
else
    df = f_ofdm / Ncar;
end

Ts   = 1/f_ofdm;
Tsym = (Ncar + Ncp) / f_ofdm;

%  Frequency axis (passband, centered around f_c) 
f_base = (-Ncar/2 : Ncar/2-1) * df;      
f_pass = conf.f_c + f_base;             
f_pass_kHz = f_pass/1e3;

%  1) Frequency response (training) 
H0 = debug.H0(:);
H0s = fftshift(H0);  

f1 = figure('Visible','off'); grid on;
plot(f_pass_kHz, 20*log10(abs(H0s)+1e-12));
xlabel('Frequency (kHz)'); ylabel('|H_0(f)| (dB)');
title('Channel magnitude from training symbol');
exportgraphics(f1, fullfile(out_dir, "H0_mag_" + tag + ".png"), 'Resolution', 200);
close(f1);

f2 = figure('Visible','off'); grid on;
plot(f_pass_kHz, unwrap(angle(H0s)));
xlabel('Frequency (kHz)'); ylabel('Phase(H_0(f)) (rad)');
title('Channel phase from training symbol');
exportgraphics(f2, fullfile(out_dir, "H0_phase_" + tag + ".png"), 'Resolution', 200);
close(f2);

%  2) Impulse response + PDP 
h0 = ifft(H0, Ncar);
tau_ms = (0:Ncp-1) * Ts * 1e3;

f3 = figure('Visible','off'); grid on;
stem(0:Ncp-1, abs(h0(1:Ncp)), 'filled');
xlabel('Tap index n'); ylabel('|h_0[n]|');
title('Impulse response magnitude (first N_{CP} taps)');
exportgraphics(f3, fullfile(out_dir, "h0_firstNcp_" + tag + ".png"), 'Resolution', 200);
close(f3);

P0 = abs(h0).^2;
f4 = figure('Visible','off'); grid on;
plot(tau_ms, 10*log10(P0(1:Ncp)+1e-15));
xlabel('Delay (ms)'); ylabel('PDP (dB)');
title('PDP from training symbol (first N_{CP} taps)');
exportgraphics(f4, fullfile(out_dir, "PDP0_" + tag + ".png"), 'Resolution', 200);
close(f4);

%  3–5) Channel evolution heatmaps 
if isfield(debug,'H_hat')
    Hhat = debug.H_hat;
    Nsym = size(Hhat,2);
    t = (0:Nsym-1) * Tsym;

    % Shift along subcarrier dimension so rows correspond to f_pass
    Hs = fftshift(Hhat, 1);

    % (3) Raw magnitude evolution
    f5 = figure('Visible','off');
    imagesc(t, f_pass_kHz, 20*log10(abs(Hs)+1e-12));
    axis xy; colorbar;
    xlabel('Time (s)'); ylabel('Frequency (kHz)');
    title('Channel evolution: 20log_{10}|Ĥ(f,s)|');
    exportgraphics(f5, fullfile(out_dir, "H_evolution_mag_" + tag + ".png"), 'Resolution', 200);
    close(f5);

    % (4) Normalized magnitude drift
    Hnorm = abs(Hs) ./ (abs(Hs(:,1))+1e-12);
    f6 = figure('Visible','off');
    imagesc(t, f_pass_kHz, 20*log10(Hnorm+1e-12));
    axis xy; colorbar;
    caxis([-3 3]);
    xlabel('Time (s)'); ylabel('Frequency (kHz)');
    title('Normalized magnitude drift: 20log_{10}|Ĥ(f,s)/Ĥ(f,1)|');
    exportgraphics(f6, fullfile(out_dir, "H_evolution_norm_" + tag + ".png"), 'Resolution', 200);
    close(f6);

    % (5) Phase drift relative to first symbol
    phi_rel = angle(Hs .* conj(Hs(:,1)));
    phi_rel = unwrap(phi_rel, [], 2);
    f7 = figure('Visible','off');
    imagesc(t, f_pass_kHz, phi_rel);
    axis xy; colorbar;
    xlabel('Time (s)'); ylabel('Frequency (kHz)');
    title('Phase drift: unwrap(angle(Ĥ(f,s)·Ĥ*(f,1)))');
    exportgraphics(f7, fullfile(out_dir, "H_phase_drift_" + tag + ".png"), 'Resolution', 200);
    close(f7);

    % PDP evolution 
    hhat = ifft(Hhat, Ncar, 1);
    P = abs(hhat).^2;
    Pncp = P(1:Ncp,:);
    f8 = figure('Visible','off');
    imagesc(t, tau_ms, 10*log10(Pncp+1e-15));
    axis xy; colorbar;
    xlabel('Time (s)'); ylabel('Delay (ms)');
    title('PDP evolution over time (first N_{CP} taps)');
    exportgraphics(f8, fullfile(out_dir, "PDP_evolution_" + tag + ".png"), 'Resolution', 200);
    close(f8);
end

%  6) Constellation + EVM 
if isfield(debug,'rx_constellation')
    rx = debug.rx_constellation(:);
    if numel(rx) > 6000
        rxp = rx(1:floor(numel(rx)/6000):end);
    else
        rxp = rx;
    end

    const = conf.constellation(:);
    D = abs(rxp - const.').^2;
    [~,idx] = min(D, [], 2);
    ref = const(idx);
    evm_rms = sqrt(mean(abs(rxp-ref).^2) / mean(abs(ref).^2));

    f11 = figure('Visible','off'); grid on;
    plot(real(rxp), imag(rxp), '.', 'MarkerSize', 8); hold on;
    plot(real(const), imag(const), 'x', 'LineWidth', 2);
    axis equal;
    xlabel('I'); ylabel('Q');
    title(sprintf('Equalized constellation (EVM_{rms}=%.2f%%)', 100*evm_rms));
    exportgraphics(f11, fullfile(out_dir, "Constellation_EVM_" + tag + ".png"), 'Resolution', 200);
    close(f11);
end

end
