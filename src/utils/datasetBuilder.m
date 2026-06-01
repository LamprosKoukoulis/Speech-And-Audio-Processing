function [X,Y] = datasetBuilder(files,label,chunksPerFile, ...
    sampleRate,chunkSamples,features)
    
numFiles = numel(files);

countErrors=0;
Xc = cell(numFiles,1);

parfor fileIdx = 1:numFiles
    file = files{fileIdx};
    audio= audioread(file);
    

    if size(audio,2) > 1
        fprintf("[datasetBuilder] ]Resizing Audio\n");
        audio = mean(audio,2);
    end
    audio = audio(:);
    tempX = zeros(chunksPerFile,features,'single');

    if length(audio) < chunkSamples
        audio = [audio; zeros(chunkSamples - length(audio),1)];
        countErrors =countErrors +1;
        % error("Audio length is less than the required chunk size.");
        % continue;
    end

    len = length(audio);
    stepSize = max(1, floor((len - chunkSamples) / max(1, chunksPerFile-1)));

    for i =1:chunksPerFile
        startIdx = (i-1) * stepSize + 1;
        if startIdx + chunkSamples -1 >len
            chunk = audio(end-chunkSamples+1:end);
        else
            chunk = audio(startIdx: startIdx + chunkSamples -1);
            
        end
        feat = extractFeatures(chunk, sampleRate);
        size(feat);
        tempX(i,:) = feat;
   
    end

    Xc{fileIdx} = tempX;
end
X = vertcat(Xc{:});
Y = repmat(categorical(label), size(X,1), 1);
fprintf( "[datasetBuilder] Audio length is less than the required chunk size. "+ countErrors +" Times. \n");
end