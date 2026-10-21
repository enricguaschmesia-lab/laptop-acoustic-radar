function [beginning_of_data] = frame_sync_global(rx_signal, conf)
% FRAME_SYNC_GLOBAL - Performs robust global peak search for frame synchronization.
% 
% Inputs:
%   rx_signal - A vector representing the noisy received signal.
%   conf - A structure containing configuration parameters, including:
%       conf.sc.nsyms - Number of symbols in the frame synchronization sequence.
%       conf.sc.os_factor - Oversampling factor.
%
% Outputs:
%   beginning_of_data - The index of the first data symbol in rx_signal. 
%                       Returns -1 if no synchronization sequence is found.

% Check for empty or flat signal
if (rx_signal(1) == 0)
    warning('Signal appears to be noise-free. The frame synchronizer may not function correctly in this case.');
end

frame_sync_length = conf.sc.nsyms;      % Length of the frame synchronization sequence
L = conf.sc.os_factor;                  % Oversampling factor

% Generate the frame synchronization sequence and map it to BPSK: 0 -> +1, 1 -> -1
% This sequence serves as the reference for correlation with the received signal
frame_sync_sequence = 1 - 2*lfsr_framesync(frame_sync_length); 

% Initialize variables for the global search
current_max_metric = 0;  % Variable to track the maximum metric found
best_index = -1;         % Variable to store the index of the best match

% Define the search range
% Start at an index where there are enough previous samples to fill the buffer
start_search = L * frame_sync_length + 1;
end_search = length(rx_signal);

% Loop through the entire signal to find the best match
for i = start_search : end_search
    
    % Extract the candidate vector (downsampled by L)
    r = rx_signal(i - L * frame_sync_length : L : i - L);
    
    % Calculate the correlation (dot product) with the frame synchronization sequence
    c = frame_sync_sequence' * r;
    
    % Calculate the energy of the received segment
    energy = abs(r' * r);
    
    % Calculate the normalized metric T
    % Guard against divide-by-zero if the signal is pure silence
    if energy > 0
        T = abs(c)^2 / energy;
    else
        T = 0;
    end
    
    % Update the global maximum if the current metric is greater
    if T > current_max_metric
        current_max_metric = T;
        best_index = i;
    end
    
end

% Return the index of the global maximum found
if best_index ~= -1
    beginning_of_data = best_index;
else
    % This typically occurs if the signal was completely empty
    warning('No synchronization sequence found.');
    beginning_of_data = -1;
end

end