function [predictions, anns] = QRSDetect(fileName)
RECORDS="100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 122 123 124 200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 222 223 228 230 231 232 233 234";

record_list = split(RECORDS, ' ');

files_1 = record_list(1:2:end);
files_2 = record_list(2:2:end);

use_first_model = ismember(fileName, files_1, 'legacy');


[data_raw, ~, anns] = extract_data(fileName, true, 1);

norm_type = 'scale';
data_arr = data_raw;
data_arr = normalize(data_arr, 2, norm_type);
data_arr = data_arr';

data = {};
for i = 1:length(anns)
   data{i} = (data_arr(:, i));  
end
data = data';

if use_first_model
    load ('models/CNN_reg_file_1.mat', 'net3')
else
    load ('models/CNN_reg_file_2.mat', 'net3')
end


predictions = classify(net3, data);

end
