
RECORDS="100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 122 123 124 200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 222 223 228 230 231 232 233 234";

record_list = split(RECORDS, ' ');

train_files = record_list(2:2:end);
test_files = record_list(1:2:end);

train_files = record_list;


%%
train_data_raw = [];
train_labels_raw = [];

for i = 1:length(train_files)
    [dat, lab, ~] = extract_data(convertStringsToChars(train_files(i)), true, 1);
    
    train_data_raw = cat(1, train_data_raw, dat);
    train_labels_raw = cat(1, train_labels_raw, lab);
end

test_data_raw = [];
test_labels_raw = [];

for i = 1:length(test_files)
    [dat, lab, ~] = extract_data(convertStringsToChars(test_files(i)), false, 1);
    
    test_data_raw = cat(1, test_data_raw, dat);
    test_labels_raw = cat(1, test_labels_raw, lab);
end


'Loaded data'

%%
if false
figure;
hold on;
plot(train_data_raw(3010, :), 'r')
plot(train_data_raw(3120, :), 'g')
plot(train_data_raw(3126, :), 'b')
plot(train_data_raw(3128, :), 'k')
train_labels_raw(3010)
train_labels_raw(3120)
train_labels_raw(3126)
train_labels_raw(3128)
end

%train_labels = array2table(cellstr(train_labels));
%test_labels = array2table(cellstr(test_labels));

%%

if true
rng("default") % For reproducibility of the partition
c = cvpartition(length(train_data_raw),"Holdout", 0.20);
trainingIndices = training(c); % Indices for the training set
testIndices = test(c); % Indices for the test set

test_data_raw = train_data_raw(testIndices, :);
test_labels_raw = train_labels_raw(testIndices, :);
train_data_raw = train_data_raw(trainingIndices, :);
train_labels_raw = train_labels_raw(trainingIndices, :);
end

%%
if true

pvc_indices = train_labels_raw == 'V';

pvc_labels = cat(1, train_labels_raw(pvc_indices, :), train_labels_raw(pvc_indices, :));
pvc_labels = cat(1, pvc_labels, train_labels_raw(pvc_indices, :));
pvc_labels = cat(1, pvc_labels, pvc_labels);

pvc_data = cat(1, train_data_raw(pvc_indices, :), train_data_raw(pvc_indices, :));
pvc_data = cat(1, pvc_data, train_data_raw(pvc_indices, :));
pvc_data = cat(1, pvc_data, pvc_data);

train_data_raw = cat(1, train_data_raw, pvc_data);
train_labels_raw = cat(1, train_labels_raw, pvc_labels);

end


%%

norm_type = 'scale';

train_data_arr = train_data_raw;
train_data_arr = normalize(train_data_arr, 2, norm_type);

test_data_arr = test_data_raw;
test_data_arr = normalize(test_data_arr, 2, norm_type);

train_data_arr = train_data_arr';
test_data_arr = test_data_arr';


%%
train_data = {};

for i = 1:length(train_data_arr)
   train_data{i} = (train_data_arr(:, i));  
end

train_data = train_data';

test_data = {};

for i = 1:length(test_data_arr)
   test_data{i} = (test_data_arr(:, i)); 
end

test_data = test_data';
%%

train_labels_arr = categorical(cellstr(train_labels_raw));
test_labels_arr = categorical(cellstr(test_labels_raw));

train_labels = {};

for i = 1:length(train_labels_arr)
   train_labels{i} = (train_labels_arr(i));  
end

train_labels = train_labels';

test_labels = {};

for i = 1:length(test_labels_arr)
   test_labels{i} = (test_labels_arr(i)); 
end

test_labels = test_labels';

%%
train_labels_char = {};

for i = 1:length(train_labels_raw)
   train_labels_char{i} = (train_labels_raw(i));  
end

train_labels_char = train_labels_char';

test_labels_char = {};

for i = 1:length(test_labels_raw)
   test_labels_char{i} = (test_labels_raw(i)); 
end

test_labels_char = test_labels_char';


%%

layers_mlp = [ ...
    sequenceInputLayer(58)
    fullyConnectedLayer(40)
    leakyReluLayer
    fullyConnectedLayer(20)
    leakyReluLayer
    fullyConnectedLayer(10)
    leakyReluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

layers_lstm = [ ...
    sequenceInputLayer(58)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(10)
    leakyReluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

layers_cnn = [
    sequenceInputLayer(58)
    convolution1dLayer(11, 8, "Padding","same")
    leakyReluLayer
    %maxPooling1dLayer(3, Stride=2, Padding="same")
    convolution1dLayer(11, 16, "Padding","same")
    leakyReluLayer
    %maxPooling1dLayer(3, Stride=2, Padding="same")
    convolution1dLayer(11, 32, "Padding","same")
    globalMaxPooling1dLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];


layers_aleshnet = [
    sequenceInputLayer(58)
    convolution1dLayer(5, 8, "Padding","same")
    leakyReluLayer
    maxPooling1dLayer(3, Stride=2, Padding="same")
    convolution1dLayer(5, 16, "Padding","same")
    leakyReluLayer
    maxPooling1dLayer(3, Stride=2, Padding="same")
    convolution1dLayer(5, 32, "Padding","same")
    convolution1dLayer(5, 32, "Padding","same")
    convolution1dLayer(5, 24, "Padding","same")
    globalMaxPooling1dLayer
    fullyConnectedLayer(20)
    leakyReluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
%%

options_mlp = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'Shuffle', 'once', ...
    'plots','training-progress', ...
    'Verbose',true);

options_lstm = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'Shuffle', 'once', ...
    'plots','training-progress', ...
    'Verbose',true);

options_cnn = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'Shuffle', 'once', ...
    'plots','training-progress', ...
    'Verbose',true);

options_alesh = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 512, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'Shuffle', 'once', ...
    'plots','training-progress', ...
    'Verbose',true);
%%

net1 = trainNetwork(train_data, train_labels, layers_mlp, options_mlp);
net2 = trainNetwork(train_data, train_labels_arr, layers_lstm, options_lstm);
net3 = trainNetwork(train_data, train_labels_arr, layers_cnn, options_cnn);
net4 = trainNetwork(train_data, train_labels_arr, layers_aleshnet, options_alesh);



save ('models/MLP_reg_file_2.mat', 'net1')
save ('models/LSTM_reg_file_2.mat', 'net2')
save ('models/CNN_reg_file_2.mat', 'net3')
save ('models/alesnet_reg_file_2.mat', 'net4')
%%

load ('models/MLP_reg_whole.mat', 'net1')
load ('models/LSTM_reg_whole.mat', 'net2')
load ('models/CNN_reg_whole.mat', 'net3')
load ('models/alesnet_reg_whole.mat', 'net4')
%%

if false
trainPred = classify(net2, train_data);

figure;
confusionchart(train_labels_arr, trainPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM');
end
%%
testPred = classify(net4, test_data);

figure;
confusionchart(test_labels_arr, testPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM');

%%

testPred = classify(net1, test_data);
if true
testPredArr = [];

for i = 1:length(testPred)
    testPredArr(i) = char(string(testPred(i)));
end

testPredArr = char(testPredArr');

confusionchart(test_labels_raw, testPredArr,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM');
end


%%

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

function [QRS_complexes, labels] = extract_data_der(filename, try_Ns, use_nth)
    [ECG, ann, anntype, QRS_length] = load_data(filename);
    
    QRS_complexes = zeros(1, QRS_length);
    labels = [];
    num_of_ns = -1;

    ECG = gradient(ECG);

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
        end
    end
    
    QRS_complexes = QRS_complexes(2:end, :);
end


function [QRS_complexes, labels] = extract_data_der_comb(filename, try_Ns, use_nth)
    [ECG, ann, anntype, QRS_length] = load_data(filename);
    
    QRS_complexes = zeros(1, QRS_length*2);
    labels = [];
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
            
            QRS_comb = cat(2, extract_signal(ECG, curr_QRS, QRS_length), ...
                              extract_signal_der(ECG, curr_QRS, QRS_length));

            QRS_complexes = cat(1, QRS_complexes, QRS_comb);
            labels = cat(1, labels, anntype(i));
        end
    end
    
    QRS_complexes = QRS_complexes(2:end, :);
end


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

function [QRS_data] = extract_signal(ECG, curr_QRS, QRS_length)
    data_length = length(ECG);

    QRS_extract = ECG(max(1, curr_QRS - round(QRS_length/2) + 1): min(data_length, curr_QRS + round(QRS_length/2)));

    QRS_data = zeros(1, QRS_length);
    QRS_data(:, 1:length(QRS_extract)) = QRS_extract;
end


function [QRS_data] = extract_signal_der(ECG, curr_QRS, QRS_length)
    data_length = length(ECG);
 
    QRS_extract_der = gradient(ECG(max(1, curr_QRS - round(QRS_length/2) + 1 + 20): min(data_length, curr_QRS + round(QRS_length/2) + 20)));
    
    QRS_data = zeros(1, QRS_length);
    QRS_data(:, 1:length(QRS_extract_der)) = QRS_extract_der;
end

function [QRS_data] = extract_signal_der_der(ECG, curr_QRS, QRS_length)
    data_length = length(ECG);
 
    QRS_extract_der = gradient(gradient(ECG(max(1, curr_QRS - round(QRS_length/2) + 1 + 20): min(data_length, curr_QRS + round(QRS_length/2) + 20))));
    
    QRS_data = zeros(1, QRS_length);
    QRS_data(:, 1:length(QRS_extract_der)) = QRS_extract_der;
end


function [ECG, ann, anntype, QRS_length] = load_data(filename)
    [signal, fs, time] = rdsamp(append('mit-bih/', filename));
    [ann, anntype, subtype, chan, num, comments] = rdann(append('mit-bih/', filename),'atr');

    ECG_raw = signal(:, 1);
    ECG = extract_baseline(ECG_raw, fs);
    QRS_length = round(fs * 0.16);
end




