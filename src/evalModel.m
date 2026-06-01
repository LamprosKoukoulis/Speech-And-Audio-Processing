function accuracy = evalModel(model,evalSpeechFiles,evalNoiseFiles,config)
    noiseChunksPerFile =1;

    [XevalNoise, YevalNoise] =datasetBuilder( ...
        evalNoiseFiles, ...
        "background", ...
        noiseChunksPerFile, ...
        config.sampleRate, ...
        config.chunkSamples, ...
        config.features);
    
    % calculate the number of SpeechChunks based on NoiseChunks
    calc = length(evalNoiseFiles) * noiseChunksPerFile;
    speechChunksPerFile = ceil(calc / length(evalSpeechFiles));
    
    [XevalSpeech,YevalSpeech] = datasetBuilder( ...
        evalSpeechFiles, ...
        "foreground", ...
        speechChunksPerFile, ...
        config.sampleRate, ...
        config.chunkSamples, ...
        config.features);
    
    Xtest = [XevalSpeech; XevalNoise];
    Ytest = [YevalSpeech; YevalNoise];
    
    switch config.type 
        case "knn"
            Ypred = predict(model.knn,Xtest);
            Ypred = categorical(Ypred);
        case "mlp"
            Yscores = model.mlp(Xtest');

            [~ , idx] = max(Yscores, [], 1);
            Ypred= categorical(model.labels(idx));
    end

    accuracy = sum(Ypred == Ytest) / numel(Ytest);
    fprintf("Test Accuracy: %.2f%%\n",accuracy*100);
    
    figure;
    confusionchart(Ytest,Ypred);
    title(upper(model.type)+" Speech vs Nosie Classification");
end