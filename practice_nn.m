% create a neural network
net = feedforwardnet([5 3]);

% train net
net.divideParam.trainRatio = 1; % training set [%]
net.divideParam.valRatio   = 0; % validation set [%]
net.divideParam.testRatio  = 0; % test set [%]

% train a neural network
[net,tr,Y,E] = train(net,P,T);

% show network
view(net)

%practice code which helped to generate the MNIST algorithm later, without using any toolboxes of Matlab.
