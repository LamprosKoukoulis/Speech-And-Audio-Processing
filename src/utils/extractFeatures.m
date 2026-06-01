function feat = extractFeatures(chunk, fs)
    chunk = chunk(:);

    % MFCCs
    mfccs = mfcc(chunk, fs, 'NumCoeffs', 13);
    mfccMean = mean(mfccs,1);
    
    % Energy
    energy = mean(chunk.^2);

    % Zero crossing rate
    zcr = zerocrossrate(chunk);
    
    % FreqZ
    [H,~]=freqz(chunk,1,256, fs);
    absH = abs(H);

    freqM = mean(absH); % Strength
    freqS = std(absH);  % Spread of the signal 
    feat = [mfccMean, energy, zcr, freqM,freqS];
    feat = feat(:);
end