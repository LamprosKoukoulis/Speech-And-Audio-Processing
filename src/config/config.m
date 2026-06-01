function config = config()
config.noisePath = "../train/noise/free-sound/";
config.speechPath = "../train/speech/us-gov/";
config.test= dir(fullfile("../test/", 'S01_U04.CH4.wav'));
config.testFile = fullfile(config.test.folder,config.test.name);
config.noiseChunksPerFile = 10;
config.chunkDuration = 0.2; %seconds
config.overlap =0.8; %percent
config.sampleRate = 16000;
config.knnModelFile = "./models/knn_model.mat";
config.mlpModelFile = "./models/mlp_model_64_32.mat";
config.knnOutputFile = "models/k-nn_results.csv";
config.mlpOutputFile = "models/mlp_results.csv";
config.chunkSamples = round(config.sampleRate * config.chunkDuration);
config.features =18;
config.type = "knn"; % mlp or knn

end