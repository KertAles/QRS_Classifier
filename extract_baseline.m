
function [fil_sig] = extract_baseline(signal, Fs)

    lw = 1/(Fs/2);
    [b,a] = butter(4,lw,'high');

    % Forward filter
    fil_sig = filter(b,a,signal);
    % Flip the result for backward filtering
    fil_sig = flipud(fil_sig);
    % And now filter again (backward)
    fil_sig = filter(b,a,fil_sig);
    % Re-flip the signal (return to forward)
    fil_sig = flipud(fil_sig);
end