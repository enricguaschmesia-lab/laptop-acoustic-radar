% MAKE_FIXTURES  Dump reference outputs of the original MATLAB link (EE-442, Group 20)
% so the Python port can be checked against them (tests/test_port.py).

% The original link code lives in ofdm_link/matlab of this repository.
ref = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'ofdm_link', 'matlab');
addpath(ref);
out = fileparts(mfilename('fullpath'));

% PN sequence: 500 preamble bits + 1024 training bits, as in txofdm.m
pn = lfsr_framesync(1524);

% Preamble configuration, as in audiotrans.m
conf.f_s = 48000;
conf.sc.f_sym = 1000;
conf.sc.nsyms = 500;
conf.sc.os_factor = conf.f_s/conf.sc.f_sym;
conf.sc.txpulse_length = 20*conf.sc.os_factor;
pulse = rrc(conf.sc.os_factor, 0.22, conf.sc.txpulse_length);

% Preamble waveform, exactly as in txofdm.m (step 6)
bpsk = -2*pn(1:conf.sc.nsyms) + 1;
ups = upsample(bpsk, conf.sc.os_factor);
pp = conv(ups, pulse, 'full');
pre = pp(1+conf.sc.txpulse_length:end-conf.sc.txpulse_length);
pre = pre / sqrt(mean(abs(pre).^2));

% Synthetic received baseband: delay, attenuation, phase rotation and noise
rng(7);
delay = 12345;
rx = [zeros(delay,1); 0.3*exp(1j*0.7)*pre; zeros(20000,1)];
rx = rx + 0.05*(randn(size(rx)) + 1j*randn(size(rx)))/sqrt(2);

% Matched filter + frame sync, exactly as in rxofdm.m (step 3)
h_mf = conj(flipud(pulse(:)));
mf = conv(rx, h_mf, 'same');
[bod, dbg] = frame_sync(mf, conf);
T_vals = dbg.T_vals;

save(fullfile(out, 'matlab_reference.mat'), 'pn', 'pulse', 'pre', 'rx', 'mf', 'bod', 'T_vals', '-v7');
fprintf('saved fixtures, beginning_of_data = %d\n', bod);
