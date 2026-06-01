function model = trainKnnModel(trainSpeechFiles, trainNoiseFiles,config)   
    [Xnoise, Ynoise] =datasetBuilder( trainNoiseFiles, ...
        "background", ...
        config.noiseChunksPerFile, ...
        config.sampleRate, ...
        config.chunkSamples, ...
        config.features);

    fprintf("Background samples : %d\n", length(Ynoise));
    % calculate the number of SpeechChunks based on NoiseChunks

    calc = length(trainNoiseFiles) * config.noiseChunksPerFile;
    speechChunksPerFile = ceil(calc / length(trainSpeechFiles));
    
    [Xspeech, Yspeech] =datasetBuilder( ...
        trainSpeechFiles, ...
        "foreground", ...
        speechChunksPerFile, ...
        config.sampleRate, ...
        config.chunkSamples, ...
        config.features);


    fprintf("Foreground samples: %d\n", length(Yspeech));
    
    % combine
    X = [Xspeech; Xnoise];
    Y = [Yspeech; Ynoise];
    
    % shuffle
    idx = randperm(length(Y));
    X = X(idx, :);
    Y = Y(idx, :);
    
    
    Knn_model = fitcknn(X,Y,...
        NumNeighbors= 3, ...
        Distance= "euclidean", ...
        Standardize= true)
    
    
    save("./models/knn_model.mat","Knn_model");
    
    model = struct;
    model.type = "knn";
    model.knn = Knn_model;
end