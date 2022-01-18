function [ECG, ann, anntype, QRS_length] = load_data(filename)
    [signal, fs, time] = rdsamp(append('mit-bih/', filename));
    [ann, anntype, subtype, chan, num, comments] = rdann(append('mit-bih/', filename),'atr');

    ECG_raw = signal(:, 1);
    ECG = extract_baseline(ECG_raw, fs);
    QRS_length = round(fs * 0.16);
end