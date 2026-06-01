function [trainSpeechFiles, trainNoiseFiles, evalSpeechFiles, evalNoiseFiles] = dataLoader(config) 
    % Create full noise and speech paths
    noise = dir(fullfile(config.noisePath, '*.wav'));
    speech= dir(fullfile(config.speechPath, '*.wav'));
    
    noiseFiles = fullfile({noise.folder},{noise.name});
    speechFiles = fullfile({speech.folder},{speech.name});
    
    % randomize the lists
    speechFiles = speechFiles(randperm(length(speechFiles))); 
    noiseFiles = noiseFiles(randperm(length(noiseFiles)));
    
    trainSpeechFiles = speechFiles(1:end-1);
    evalSpeechFiles = speechFiles(end);
    
    trainNoiseFiles = noiseFiles(1:end-100);
    evalNoiseFiles = noiseFiles(end-101:end);
    
    statsTable = table( ...
    length(trainSpeechFiles), ...
    length(trainNoiseFiles), ...
    length(evalSpeechFiles), ...
    length(evalNoiseFiles), ...
    'VariableNames', { ...
        'TrainSpeechFiles', ...
        'TrainNoiseFiles', ...
        'EvalSpeechFiles', ...
        'EvalNoiseFiles' ...
    });

    disp(statsTable);


end