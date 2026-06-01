clear;
close all;

audio_length = 0.25;    % 25 ms
audio_shift = 0.01;     % 10 ms

[x,fs] = audioread("train\noise\free-sound\noise-free-sound-0001.wav");
% time = (0:length(x)-1) / fs;
% plot(time,x)
% xlabel('Time (s)')
% ylabel('Amplitude')
% title('Audio Waveform')

% spectrogram(x, round(0.025*fs), round(0.015*fs), [], fs, 'yaxis')
% title('Spectrogram')


% frame_length = round(audio_length * fs);
% frame_step = round(audio_shift * fs);

% frames = buffer(x, frame_length,frame_length - frame_step, 'nodelay');

% Energy
% energy = sum(frames.^2);

% ZCR
% zcr = sum(abs(diff(sign(frames)))) / (2 * size(frames,1));

% MFCC
% coeffs = mfcc(x, fs, frame_length, frame_length - frame_step);

% Συνένωση features
% features = [energy' zcr' coeffs];

