% clear
% batchSize =10;
config = config();
load("models/dataset.mat","dataset");
trainSpeechFiles = dataset.trainSpeechFiles;
trainNoiseFiles = dataset.trainNoiseFiles;
evalSpeechFiles =  dataset.evalSpeechFiles;
evalNoiseFiles = dataset.evalNoiseFiles;


testTime =tic;
switch config.type
    case "knn"
        % RESTORING LAST MODEL 
        load(config.knnModelFile,"Knn_model");
        testModel(Knn_model,config);
    case "mlp"
        load(config.mlpModelFile,"model");
        testModel(model,config);
end

testTime=toc(testTime);
fprintf("Testing time: %.2f seconds\n\n", testTime);

comTime = tic;

compareWithTranscript();

comTime = toc(comTime);
fprintf("Comparing time: %.2f seconds\n\n", comTime);