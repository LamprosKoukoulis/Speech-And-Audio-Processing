function [X,Y] = fullDatasetBuilder(files, label, ...
    sampleRate, chunkSamples, features, chunksPerFile)
    % chunksPerFile =20;
    total = length(files) * chunksPerFile
    
    X = zeros(total, features);
    Y = categorical(strings(total,1));
    h= numel(length(files)/8);
    
    parfor f = 1: numel(total)
    
        file = files{f};
        audio = audioread(file);
        audio = audio(:);
    
        localX = zeros(chunksPerFile, features);
        localY = categorical(strings(chunksPerFile,1));
    
        maxStart = length(audio) - chunkSamples;
    
        if maxStart <= 0
            continue;
        end
    
        for j = 1:chunksPerFile
    
            % random safe start
            if chunksPerFile == 1
                startIdx = 1;
            else
                step = floor(maxStart / (chunksPerFile-1));
                startIdx = (j-1)*step + 1;
            end
    
            chunk = audio(startIdx:startIdx + chunkSamples - 1);
    
            feat = extractFeatures(chunk, sampleRate);
    
            localX(j,:) = feat;
            localY(j) = categorical(label);
        end
    
        X_f{f} = localX;
        Y_f{f} = localY;
    end
    
    X = vertcat(X_f{:});
    Y = vertcat(Y_f{:});

end