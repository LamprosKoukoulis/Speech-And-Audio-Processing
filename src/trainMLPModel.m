function model = trainMLPModel(trainSpeechFiles, trainNoiseFiles,config)  
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
    
    Y = categorical(Y);
    labels = categories(Y);
    
    % convert labels to One-Hot    
    onehot = onehotencode(Y,2);

    mlp = patternnet([64 32]);

    mlp.trainFcn = 'trainlm';
    
    mlp.trainParam.epochs =500;
    % mlp.trainParam.max_fail = 10;
    
    mlp.divideParam.trainRatio = 0.7;
    mlp.divideParam.valRatio = 0.15; 
    mlp.divideParam.testRatio = 0.15;

    mlp = train(mlp, X',onehot'); % Train the MLP model with the prepared data

    Ypred = mlp(X');
    loss = perform(mlp, onehot', Ypred);
    fprintf("Cross-Entropy Loss: %.4f\n", loss);
    
    model = struct();
    model.type = "mlp";
    model.mlp = mlp;
    model.labels = labels;
    
    % model = mlp;
    save(config.mlpModelFile,"model");
end