%mnist practice code
clc;
clear all;
images = loadMNISTImages('train-images.idx3-ubyte'); % initialize figure  
labels = loadMNISTLabels('train-labels.idx1-ubyte'); % initialize figure
labels = labels';                                    % transpose
labels(labels==0)=10;                                % dummyvar function doesn´t take zeroes
labels=dummyvar(labels);                             % 

% initialize figure
figure                                          
colormap(gray)

%trying for 6 x 6 samples from mnist.mat dataset
for i = 1:36                                    
    subplot(6,6,i)                              
    digit = reshape(images(:, i), [28,28]);     
    imagesc(digit)                              
    title(num2str(labels(i)))                   % to show label
end

x = images;
t = labels';
trainFcn = 'trainscg';              
% use scaled conjugate gradient for training

hiddenLayerSize = 100;                          
net = patternnet(hiddenLayerSize);              
% to create Pattern Recognition Network

% In order to assess the performance, 
% you have to define test and validation sets and a performance metric.
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'crossentropy';    % using Cross-Entropy to test and validate the function.

[net,tr] = train(net,x,t);
% net saves info about network

%tr gives information about the training record