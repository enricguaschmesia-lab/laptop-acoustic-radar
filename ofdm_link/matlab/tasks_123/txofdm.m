function [txsignal, conf] = txofdm(txbits, conf)
% Digital Transmitter Function
% This function generates a transmitted OFDM signal from input bits.
% 
% Inputs:
%   txbits - Input binary data to be transmitted.
%   conf   - Configuration structure containing parameters for transmission,
%            including modulation order, OFDM parameters, and pulse shaping.
%
% Outputs:
%   txsignal - The transmitted OFDM signal in the time domain.
%   conf     - The configuration structure used for transmission.
% 
% The process includes the following steps:
% 1. Generate a preamble for synchronization.
% 2. Map input bits to QPSK symbols.
% 3. Construct the OFDM grid, which includes training and data symbols.
% 4. Perform IFFT and add a cyclic prefix to each OFDM symbol.
% 5. Convert the parallel OFDM sequences into a serial payload.
% 6. Normalize the energy of the payload.
% 7. Concatenate the normalized preamble and payload.
% 8. Upconvert the baseband signal to the desired carrier frequency.

    % 1. Generate Preamble
    % Utilize the provided lfsr_framesync function for synchronization.
    preamble_seq = lfsr_framesync(conf.sc.nsyms); % conf.sc.nsyms is defined in audiotrans
    bpsk_preamble = 1 - 2 * preamble_seq;
    
    % Apply pulse shaping to the preamble using a Root Raised Cosine (RRC) filter.
    preamble_upsamp = conv(upsample(bpsk_preamble, conf.sc.os_factor), conf.sc.txpulse, 'same');
    
    % Normalize the energy of the preamble.
    norm_preamble = preamble_upsamp / sqrt(mean(abs(preamble_upsamp).^2));

    % 2. Map Data Bits to QPSK Symbols (unchanged)
    bits_reshaped  = reshape(txbits, conf.modulation_order, []).';  % [Nsym x 2]
    symbol_indices = bi2de(bits_reshaped, 'left-msb') + 1;
    qpsk_data      = conf.constellation(symbol_indices);
    
    % 2b. Generate PN-based known OFDM symbol (must match rxofdm)
    pn_len  = conf.sc.nsyms + 2*conf.ofdm.ncarrier;
    pn_seq  = lfsr_framesync(pn_len);                 % 0/1
    known_bits = pn_seq(conf.sc.nsyms+1 : conf.sc.nsyms+2*conf.ofdm.ncarrier);
    known_bits = reshape(known_bits, 2, []).';        % [Ncar x 2]
    
    GrayMap   = conf.constellation;                   % same order as in rxofdm
    A_known   = GrayMap(bi2de(known_bits, 'left-msb') + 1).';   % [Ncar x 1]
    
    % 3. Construct the OFDM Grid (unchanged)
    total_ofdm_symbols = conf.ofdm.nsym + 1;
    ofdm_grid = zeros(conf.ofdm.ncarrier, total_ofdm_symbols);
    
    ofdm_grid(:,1)     = A_known;                                       % training
    ofdm_grid(:,2:end) = reshape(qpsk_data, conf.ofdm.ncarrier, []);    % data
    
    % 4. Perform IFFT at baseband (fs_ofdm = conf.ofdm.bandwidth = 2000 Hz)
    Nfft     = conf.ofdm.ncarrier;      % 512
    Ncp      = conf.ofdm.cplen;         % 256
    sym_len  = Nfft + Ncp;              % 768 samples at 2 kHz
    
    tx_ofdm_sequences = zeros(sym_len, total_ofdm_symbols);
    
    for i = 1:total_ofdm_symbols
        % Standard N-point IFFT (no oversampling)
        time_domain_symbol = ifft(ofdm_grid(:, i), Nfft);   % size [512 x 1]
        
        % Add CP in baseband samples (no os_factor!)
        cp_samples = time_domain_symbol(end-Ncp+1 : end);   % [256 x 1]
        tx_ofdm_sequences(:, i) = [cp_samples; time_domain_symbol];
    end
    
    % 5. Parallel-to-serial baseband OFDM payload (fs_ofdm = 2 kHz)
    tx_payload_ofdm = tx_ofdm_sequences(:);
    
    % Normalize the OFDM payload power
    tx_payload_ofdm = tx_payload_ofdm / sqrt(mean(abs(tx_payload_ofdm).^2));
    
    % 6. Resample OFDM payload from 2 kHz → 48 kHz using ofdm_tx_resampling
    tx_payload_48k = ofdm_tx_resampling(tx_payload_ofdm, conf);
    
    % 7. Concatenate preamble (already at 48 kHz) and OFDM payload at 48 kHz
    tx_baseband = [norm_preamble; tx_payload_48k];
    
    % 8. Upconvert tx_baseband to carrier
    time = (0:length(tx_baseband)-1).' / conf.f_s;
    txsignal = real(tx_baseband .* exp(1j * 2 * pi * conf.f_c * time));


end