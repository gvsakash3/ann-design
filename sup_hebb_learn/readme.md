Brief description of task 2

* Additional Conceptual information(for personal reference):-

     Functions used in the final Matlab code ~
1. `trainscg` ----> [Scaled conjugate gradient backpropagation](https://in.mathworks.com/help/nnet/ref/trainscg.html)
   
     Syntax: 
  
            net.trainFcn='trainscg'
            
            [net,tr]=train(net,.....)
     It is a network training function that updates weight and bias values according to scaled conjugate gradient method.
     
     
     Also, the training stops for :
     
      x     Max. no of epochs (repitions) is reached.
     
      x     Max. amount of time has reached.
     
      x     Performance is minimized to the goal.


2. `crossentropy` ----> [Neural network performance](https://in.mathworks.com/help/nnet/ref/crossentropy.html)

     Syntax: 
     
               perf = crossentropy(net,targets,outputs,perfWeights)

               perf = crossentropy(___,Name,Value)
     'perf' calculates a network performance given targets and outputs, with optional performance weights and other parameters.
     
     The function returns a result that heavily penalizes outputs that are extremely inaccurate (y near 1-t), with very little penalty for fairly correct classifications (y near t). Minimizing cross-entropy leads to good classifiers.          
