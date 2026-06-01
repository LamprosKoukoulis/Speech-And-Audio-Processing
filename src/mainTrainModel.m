clear;

totTime= tic;

config = config();
[trainSpeechFiles, trainNoiseFiles, evalSpeechFiles, evalNoiseFiles]= dataLoader(config);

trainTime =tic;

switch config.type 
    case "knn"
model = trainKnnModel(trainSpeechFiles, trainNoiseFiles,config);
    case "mlp"
model = trainMLPModel(trainSpeechFiles, trainNoiseFiles,config);
end

trainTime=toc(trainTime);
fprintf("Training time: %.2f seconds\n\n", trainTime);

accuracy = evalModel(model,evalSpeechFiles,evalNoiseFiles,config);

%           SAVING DATASET
dataset.trainSpeechFiles = trainSpeechFiles;
dataset.trainNoiseFiles = trainNoiseFiles;
dataset.evalSpeechFiles = evalSpeechFiles;
dataset.evalNoiseFiles = evalNoiseFiles;
save("models/dataset.mat","dataset");

mainEvalModel;

totTime = toc(totTime);
fprintf("Total time: %.2f seconds\n\n", totTime);