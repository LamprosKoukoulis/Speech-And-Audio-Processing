function results = mergeSeqments(startTimes, endTimes, labels)
    mergedStart = [];
    mergedEnd = [];
    mergedLabel = categorical();

    curLabel = labels(1);
    curStart = startTimes(1);
    curEnd = endTimes(1);

    for i = 2:length(labels)
        if labels(i) == curLabel
            curEnd = endTimes(i);
        else
            mergedStart(end + 1,1) = curStart;
            mergedEnd(end + 1,1) = curEnd;
            mergedLabel(end + 1,1) = curLabel;
            curLabel = labels(i,1);
            curStart = startTimes(i);
            curEnd = endTimes(i);
        end
    end
    
    % Add last seqment
    mergedStart(end+1,1) = curStart;
    mergedEnd(end+1,1) = curEnd;
    mergedLabel(end+1,1) = curLabel;
    results = table(mergedStart, mergedEnd, mergedLabel, 'VariableNames', {'start', 'end', 'label'});

end