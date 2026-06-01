function [X,Y] = datasetBuilder(files,label,chunksPerFile, ...
    sampleRate,chunkSamples,features)

    total = length(files) * chunksPerFile;

    X = zeros(total,features);
    Y= categorical(strings(total,1));

    idx = 1;
    for i = 1:length(files)
        file = files{i};

        for j = 1:chunksPerFile

            chunk = randomChunk(sampleRate,chunkSamples,file);

            feat = extractFeatures(chunk, sampleRate);

            X(idx,:) = feat;
            Y(idx) = categorical(label);

            idx = idx + 1;
        end
    end
end