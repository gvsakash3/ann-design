Brief description of task 2

* Additional Conceptual information(for personal reference only)

1. trainscg ------> Scaled conjugate backpropagation
   
     Syntax: 
  
            net.trainFcn='trainscg'
            
            [net,tr]=train(net,.....)
     It is a network training function that updates weight and bias values according to scaled conjugate gradient method.
     
     
     Also, the training stops for :
     
      x     Max. no of epochs (repitions) is reached.
     
      x     Max. amount of time has reached.
     
      x     Performance is minimized to the goal.
