function [QRS_data] = extract_signal(ECG, curr_QRS, QRS_length)
    data_length = length(ECG);

    QRS_extract = ECG(max(1, curr_QRS - round(QRS_length/2) + 1): min(data_length, curr_QRS + round(QRS_length/2)));

    QRS_data = zeros(1, QRS_length);
    QRS_data(:, 1:length(QRS_extract)) = QRS_extract;
end