function chunk = randomChunk(sampleRate,chunkSamples,audio)
   
    % if fs ~= sampleRate
    %     fprintf("Resample sound at "+fs+" Hz.\n")
    %     audio = resample(audio, sampleRate, fs);
    % end

    if size(audio,2) >1
        fprintf("[RandomChunk] resizing used");
        audio = mean(audio,2);
    end
    len = length(audio);
    if length(audio) < chunkSamples
        chunk = [audio; zeros(chunkSamples - len,1)];
        return;
    else
        idx = randi(len - chunkSamples +1);

        chunk = audio(idx:idx + chunkSamples -1);
    end
end