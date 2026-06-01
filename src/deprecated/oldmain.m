chunkDuration = 0.5; %seconds
sampleRate = 16000;
batchSize =10;
chunkSamples = round(sampleRate * chunkDuration);

features =17;

noiseChunksPerFile =1;

noisePath = "train/noise/free-sound/";
speechPath = "train/speech/us-gov/";

% Create full noise and speech paths
noise = dir(fullfile(noisePath, '*.wav'));
speech= dir(fullfile(speechPath, '*.wav'));

noiseFiles = fullfile({noise.folder},{noise.name});
speechFiles = fullfile({speech.folder},{speech.name});

% randomize the lists
speechFiles = speechFiles(randperm(length(speechFiles))); 
noiseFiles = noiseFiles(randperm(length(noiseFiles)));

trainSpeechFiles = speechFiles(1:end-1);
evalSpeechFiles = speechFiles(end);

trainNoiseFiles = noiseFiles(1:end-100);
evalNoiseFiles = noiseFiles(end-101:end);


[Xnoise, Ynoise] =datasetBuilder( trainNoiseFiles, ...
    "noise", ...
    1, ...
    sampleRate, ...
    chunkSamples, ...
    features);

% calculate the number of SpeechChunks based on NoiseChunks
calc = length(trainNoiseFiles) * noiseChunksPerFile;

speechChunksPerFile = ceil(calc / length(trainSpeechFiles));

[Xspeech, Yspeech] =datasetBuilder( ...
    trainSpeechFiles, ...
    "speech", ...
    speechChunksPerFile, ...
    sampleRate, ...
    chunkSamples, ...
    features);


fprintf("Speech samples: %d\n", length(Yspeech));
fprintf("Noise samples : %d\n", length(Ynoise));

% combine
X = [Xspeech; Xnoise];
Y = [Yspeech; Ynoise];

% shuffle
idx = randperm(length(Y));
X = X(idx, :);
Y = Y(idx, :);


% X = chunks Y = 1:spech ,0:noise
% X = zeros(chunkSamples, batchsize);
% Y = zeros(batchsize,1);

% X=zeros(batchSize,17);
% Y=categorical(strings(batchSize,1));
% 
% speechIdx =1;
% noiseIdx =1;
% % allocate half chunks to speechSamples
% for i= 1:batchSize/2
%     file = speechFiles{speechIdx};
%     chunk = randomChunk(sampleRate,chunkSamples,file);
% 
%     feat = extractFeatures(chunk,sampleRate);
% 
%     % X(:, i) = chunk; % Store the speech chunk in X
%     % Y(i) = 1; % Mark with 1: speech
%     X(i,:) = feat;
%     Y(i) = categorical("speech");
% 
%     speechIdx = speechIdx +1;
% 
%     % Check if we reached the end of speechFile list
%     if speechIdx > length(speechFiles)
%         speechIdx = 1;
%         speechFiles = speechFiles(randperm(length(speechFiles)));
%     end
% end
% 
% % allocate half chunks to noiseSamples
% for i = 1: batchSize/2
%     file = noiseFiles{noiseIdx};
%     chunk = randomChunk(sampleRate,chunkSamples,file);
% 
%     feat = extractFeatures(chunk,sampleRate);
% 
%     idx = batchSize/2 + i;
%     X(idx,:) = feat; % Store the speech chunk in X
%     Y(idx) = categorical("noise"); % Mark with 0: noise
%     noiseIdx = noiseIdx +1;
% 
%     % Check if we reached the end of speechFile list
%     if noiseIdx > length(noiseFiles)
%         noiseIdx = 1;
%         noiseFiles = noiseFiles(randperm(length(noiseFiles)));
%     end
% end

% Randomize the order of the combined dataset
% combinedIdx = randperm(length(Y));
% X = X(combinedIdx, :);
% Y = Y(combinedIdx);

% K =5;

% Mdl = fitcknn(X,Y,OptimizeHyperparameters="auto", ...
%     HyperparameterOptimizationOptions= ...
%     struct(AcquisitionFunctionName="expected-improvement-plus"))
% result
% Best estimated feasible point (according to models):
%     NumNeighbors     Distance     Standardize
%     ____________    __________    ___________
% 
%          2          seuclidean       false   


model = fitcknn(X,Y,...
    NumNeighbors= 2, ...
    Distance= "seuclidean", ...
    Standardize= false)

% % save("knn_model.mat",model);
% testFiles = 114;
% Xtest = zeros(testFiles*2,17);
% Ytest= categorical(strings(testFiles*2,1));
% 
% testNoisePath = "eval/noise/";
% testSpeechPath = "eval/speech/";
% 
% % Create full noise and speech paths
% testNoise = dir(fullfile(testNoisePath, '*.wav'));
% testSpeech= dir(fullfile(testSpeechPath, '*.wav'));
% testSpeechFiles= fullfile({testSpeech.folder},{testSpeech.name});
% testNoiseFiles= fullfile({testNoise.folder},{testNoise.name});
% 
% for i = 1:length(testSpeechFiles)
%     file = testSpeechFiles{i};
%     chunk = randomChunk(sampleRate,chunkSamples,file);
%     feat = extractFeatures(chunk,sampleRate);
% 
%     Xtest(i,:) = feat;
%     Ytest(i,:) = categorical("speech");
%     for j = 1:10
%         chunk = randomChunk(sampleRate,chunkSamples,file);
%         feat = extractFeatures(chunk,sampleRate);
% 
%         Xtest(i+j,:) = feat;
%         Ytest(i+j) = categorical("speech");
%     end
% end
% idx = i*10-1;
% for i = 1:length(testNoiseFiles)
%     file = testNoiseFiles{i};
% 
%     chunk = randomChunk(sampleRate,chunkSamples,file);
%     feat = extractFeatures(chunk,sampleRate);
% 
%     Xtest(i+idx,:) = feat;
%     Ytest(i+idx) = categorical("noise");
% end

[XevalNoise, YevalNoise] =datasetBuilder( ...
    evalNoiseFiles, ...
    "noise", ...
    noiseChunksPerFile, ...
    sampleRate, ...
    chunkSamples, ...
    features);

% calculate the number of SpeechChunks based on NoiseChunks
calc = length(evalNoiseFiles) * noiseChunksPerFile;
speechChunksPerFile = ceil(calc / length(evalSpeechFiles));

[XevalSpeech,YevalSpeech] = datasetBuilder( ...
    evalSpeechFiles, ...
    "speech", ...
    speechChunksPerFile, ...
    sampleRate, ...
    chunkSamples, ...
    features);

Xtest = [XevalSpeech; XevalNoise];
Ytest = [YevalSpeech; YevalNoise];

Ypred = predict(model,Xtest);
Ypred = categorical(Ypred);

accuracy = sum(Ypred == Ytest) / numel(Ytest);
fprintf("Test Accuracy: %.2f%%\n",accuracy*100);

figure;
confusionchart(Ytest,Ypred);
title("K-nn Speech vs Nosie Classification");