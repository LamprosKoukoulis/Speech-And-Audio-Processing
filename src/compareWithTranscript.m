json = jsondecode(fileread("../test/transcriptions/S01.json"));
config = config();


n = length(json);
jStart = zeros(n,1);
jEnd = zeros(n,1);
jLabel = strings(n,1);
trueL = strings(0,1);
predL = strings(0,1);

% convert HH:mm:ss to seconds
for i = 1:n
    jStart(i) = timeToSeconds(json(i).start_time);
    jEnd(i)   = timeToSeconds(json(i).end_time);
    jLabel(i) = inferLabel(json(i).words);
end

jsonTable = table(jStart, jEnd, jLabel, ...
    'VariableNames',{'start','end','label'});

% load predictions
switch config.type
    case "knn"
        pred = readtable(config.knnOutputFile);
        pred.class = string(pred.class);
    case "mlp"
        pred = readtable(config.mlpOutputFile);
        pred.class = string(pred.class);
end
disp("Prediction distribution:");
tabulate(categorical(pred.class))

threshold = 0.5;
correct = 0;

% usedJson for checking which segments where already matched.
usedJSON = false(height(jsonTable),1);

for i = 1:height(pred)
    bestIoU = 0;
    bestIdx = -1;

    for j = 1:height(jsonTable)

        if usedJSON(j)
            continue;
        end
        
        % Calculate the Intersection over Union (IoU) for the current prediction
        iou = overlap( ...
            pred.start(i), pred.xEnd(i), ...
            jsonTable.start(j), jsonTable.end(j));
        
        % Keep best match
        if iou > bestIoU
            bestIoU = iou;
            bestIdx = j;
        end
        
    end

    % Accept Match if above threshold
    if bestIoU >= threshold && bestIdx > 0
        
        trueLabel = string(jsonTable.label(bestIdx));
        predLabel = string(pred.class(i));
        
        if trueLabel ~= "uncertain"
            trueL(end+1,1) = trueLabel;
            predL(end+1,1) = predLabel;
            if pred.class(i) == jsonTable.label(bestIdx)
                
                correct = correct + 1;
            end
        end

        usedJSON(bestIdx) = true;
    end
end

accuracy = correct / max(numel(trueL),1);
fprintf("Alignment accuracy: %.2f%%\n", accuracy*100);
fprintf("Matched segments: %d\n", numel(trueL));

rate_results(trueL,predL)

figure;
% accuracy = max(numel(true),1);
cm =confusionchart( ...
    categorical(trueL), ...
    categorical(predL));

% cm.Normalization ="row-normalized";
title(upper(config.type)+" Confusion Matrix");


function sec = timeToSeconds(t)
    parts = split(t, ":");
    h = str2double(parts(1));
    m = str2double(parts(2));
    s = str2double(parts(3));
    sec = h*3600 + m*60 + s;
end

% Labeling from transcript
function label = inferLabel(words)

    w = lower(string(words));

    if contains(w, "noise")
        label = "background";
    elseif contains(w, "inaudible")
        label = "uncertain";
    elseif w == "" || w == "[]"
        label = "background";
    else
        label = "foreground";
    end
end

% Intersection over Union between two time seqments
% Used for aligment matching
function iou = overlap(a1,a2,b1,b2)
    inter = max(0, min(a2,b2) - max(a1,b1));
    union = max(a2,b2) - min(a1,b1);
    iou = inter / union;
end

function rate_results(trueL,predL)
    trueCat = categorical(trueL);
    predCat = categorical(predL);
    % true positive
    TP = sum(trueCat == "foreground" & predCat == "foreground"); 
    % false positive
    FP = sum(trueCat == "background" & predCat == "foreground");
    % false negative
    FN = sum(trueCat == "foreground" & predCat == "background");
    
    precision = TP / max(TP + FP,1);
    recall    = TP / max(TP + FN,1);
    
    f1 = 2 * precision * recall / max(precision + recall,eps);
    
    fprintf("Precision: %.4f\n", precision);
    fprintf("Recall: %.4f\n", recall);
    fprintf("F1 Score: %.4f\n", f1);
end