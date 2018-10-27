## ann-design 
 Repository with collection of final codes submitted for my summer internship at IIT Hyd.

___


### 1. Multi-neuron and Multi-layer Perceptron Learning Models 
  

* After studying rosenblatt model and previous network models, I designed a code to generate weights and bias for each iteration until convergence is achieved by cycling through the input vectors. ( [convergence_perceptron.m](convergence_perceptron.m) )

* Method: 


    | p1 | p2 | p3 | p4 |
    | ------------- | ------------- | ------------- | ------------- |
    | 2	| 1	| -2 | -1 |
    | 2 | -2 | 2 | 1 |

     | Target values |
     | ------------- |
     | 0 |
     | 1 |
     | 0 |
     | 1 |

     Taking initial value as null for weight and bias, perceptron learning rule was applied and weights and bias were generated. W(0) = [0 0]<sup>T</sup> , b(0)=0.

    Weight values: w = [a b]<sup>T</sup>

* Results were generated by checking for convergence of the bias and weights, by checking for each vector to verify the learning rule.

    ![Plot1](result_percep_plot.jpg)
    
___    


### 2. Autoassociative memory models - digit recognition and pattern recovery.

 * Developed a code for pattern recognition of digits and extended this to extract from noisy and occluded patterns as well. 
 ( [mnist_test_code.m](sup_hebb_learn/mnist_test_code.m) )
 * In an autoassociative memory the desired output vector is equal to the
input vector (i.e., t<sub>q</sub> = p<sub>q</sub> )

   ![Plot2](sup_hebb_learn/ref_model.jpg)  
   This model is used to store a set of patterns and then to recall them, even when corrupted patterns are provided as inputs.  
   
   ![digit0](https://github.com/gvsakash/ann-design/blob/master/sup_hebb_learn/pattern_digit.jpg)  
   
   Sample image scanned above,  will have a prototype pattern : p<sub>1</sub> = [–1 1 1 1 1 –1 1 –1 –1 –1 –1 1 1 –1 1 –1 ... 1] <sup>T</sup>
   
   The vector p<sub>1</sub> corresponds to digit 0 (6 x 5 grid scan as illustrated below), p<sub>2</sub> to digit 1, ....
   
   Using Hebb rule, weight matrix is calculated. Based on Supervised Learning rule, the code was developed.
   
   * The perfomance graph and validation function can be referred further, along with the codes and mnist.mat and other files.
   
   ![Plot3](https://github.com/gvsakash/ann-design/blob/master/sup_hebb_learn/performance.png)
   
   * The code was extended to extract patterns from noisy and occluded image scans. (illustration below) ![noisy-plot](https://github.com/gvsakash/ann-design/blob/master/sup_hebb_learn/noisyexamples.jpg)     
                     
        
     

   > See the final commits in [Supervised Hebbian learning](sup_hebb_learn) for further details and code.

____
### 3. Function approxiamtion and time series anlaysis.
   ![Plot4](https://github.com/gvsakash/ann-design/blob/master/func_approx/func_approx.jpg)   
   Illustration of the plot results. ^
  * Developed a network model with the LMS algorithm and multi-layers, studying network response for parameter changes. ( [func-approx.m](func_approx/func_approx_trail.m) )
  <img src="func_approx/model_illust.jpg" width="600">
  
  * Also developed a time series forecasting model in Python by implementation of concepts from Hagan Demuth notes, studying results from various algorithms. ( [time series-ipynb code](func_approx/time-series-practice-model.ipynb) )
  * Studied and designed models based on backpropagation, ADALINE networks. Developed a multi-layer model and used hidden layers as well to study their architecture and performance. ( [percep-hidden-layers code](func_approx/multi_lay_percep_two_hidd_layers.ipynb) )

   > See the final commits in [time series and func. approxiamtion folder](func_approx) for further details and code.




___
##### Footnotes: 

* Primary reference :  [Neural Network Design](http://hagan.okstate.edu/NNDesign.pdf) by Demuth, H.B., Beale, M.H., De Jess, O. and Hagan, M.T., 2014.
* All sample codes and implementations, other models submitted for project are in the [perceptroncodes](https://github.com/gvsakash/perceptroncodes) repo.
* A reference illustration of multi-layer perceptron network architecture:
<img src="func_approx/multilperceptron_refbook.jpg" width="700">

[![akash-badge](https://img.shields.io/badge/made%20with-MATLAB-orange.svg)](https://www.mathworks.com/products/matlab.html) 
 [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![akash-badge](https://img.shields.io/badge/tried%20and%20tested-Akash-brightgreen.svg)](https://github.com/gvsakash/)
