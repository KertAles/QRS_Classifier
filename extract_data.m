
function [QRS_complexes, labels, anns] = extract_data(filename, try_Ns, use_nth)
    [ECG, ann, anntype, QRS_length] = load_data(filename);
 
    QRS_complexes = zeros(1, QRS_length);
    labels = [];
    anns = [];
    num_of_ns = -1;

    for i = 1:length(ann)
        if anntype(i) == 'N' || anntype(i) == 'V'
            if try_Ns && anntype(i) == 'N'
                num_of_ns = num_of_ns + 1;
                if mod(num_of_ns, use_nth) ~= 0
                   continue 
                end
            end
        
            curr_QRS = ann(i);
            QRS_complexes = cat(1, QRS_complexes, extract_signal(ECG, curr_QRS, QRS_length)); 
            labels = cat(1, labels, anntype(i));
            anns = cat(1, anns, ann(i));
        end
    end
    
    QRS_complexes = QRS_complexes(2:end, :);
end
