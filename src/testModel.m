function results = testModel(model,config)
    chunkSamples = round(config.sampleRate * config.chunkDuration);
    stepChunks = round(chunkSamples * (1- config.overlap));
    
    [audio,~] = audioread(config.testFile);
    
    if size(audio,2) > 1
        fprintf("[testModel] ]Resizing Audio\n");
        audio = mean(audio,2);
    end
    
    % maxChunks = min(length(audio), maxTestDuration * sampleRate);
    % audio = audio(1:maxChunks);
    
    numChunks = floor((length(audio)- chunkSamples)/ stepChunks) + 1;
    
    Xtest = zeros(numChunks,config.features);
    startTimes = zeros(numChunks,1);
    endTimes = zeros(numChunks,1);
    
    fprintf("Processing "+numChunks+" chunks for testing\n");
    
    parfor i = 1:numChunks
        startIdx = (i-1)* stepChunks + 1;
        endIdx = startIdx + chunkSamples -1;
    
        chunk = audio(startIdx:endIdx);
        feat = extractFeatures(chunk, config.sampleRate);
        Xtest(i,:) = feat;
    
        startTimes(i) = (startIdx -1)/ config.sampleRate;
        endTimes(i) = (endIdx -1)/config.sampleRate;
    end 
    
    switch config.type 
        case "knn"
            Ypred = predict(model,Xtest);
            Ypred =categorical(Ypred);
        case "mlp"
            Yscores = model.mlp(Xtest');

            [~ , idx] = max(Yscores, [], 1);
            Ypred= categorical(model.labels(idx));
    end
    
    % window = 2;
    % 
    % % smoothing
    % for i = window+1:length(Ypred)-window
    %     segment = Ypred(i-window:i+window);
    % 
    %     foregroundCount = sum(segment == "foreground");
    %     backgroundCount = sum(segment == "background");
    % 
    %     if foregroundCount > backgroundCount
    %         Ypred(i) = categorical("foreground");
    %     else
    %         Ypred(i) = categorical("background");
    %     end
    % end
    
    results  = mergeSeqments(startTimes,endTimes,Ypred);
    AudioFile = repmat(string(config.test.name), height(results),1);
    
    % startTime = duration(0,0,results.start,'Format','hh:mm:ss');
    % endTime   = duration(0,0,results.end,'Format','hh:mm:ss');
    % 
    % resTable = table(AudioFile, startTime, endTime, string(results.label),...
    %     'VariableNames',{'Audiofile','start','end','class'});
    
    resTable = table(AudioFile, results.start, results.end, string(results.label),...
        'VariableNames',{'Audiofile','start','end','class'});
    switch config.type 
        case "knn"
            writetable(resTable,config.knnOutputFile);
        case "mlp"
            writetable(resTable, config.mlpOutputFile);
    end
end