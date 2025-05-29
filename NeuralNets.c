#include "NeuralNets.h"


int train_1layer_net(double sample[INPUTS],int label,double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS])
{
 /*
  *   This is your main training function for 1-layer networks. Recall from lecture that we have a simple,
  *  direct connection between inputs and output neurons (the only layer present here). What we are doing
  *  in effect is training 10 different classifiers, each of which will learn to distinguish one of our
  *  training digits.
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   weights_io - Array of weights connecting inputs to output neurons, weights[i][j] is the weight from input
  *                i to output neuron j. This array has a size of 785x10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */

  int result = classify_1layer(sample, label, sigmoid, weights_io);
  double activations[OUTPUTS];
  feedforward_1layer(sample, sigmoid, weights_io, activations);
  backprop_1layer(sample, activations, sigmoid, label, weights_io);
  return result;		// <--- This should return the class for this sample
}

int classify_1layer(double sample[INPUTS],int label,double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS])
{
 /*
  *   This function classifies an input sample given the current network weights. It returns a class in
  *  [0,9] corresponding to the digit the network has decided is present in the input sample
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   weights_io - Array of weights connecting inputs to output neurons, weights[i][j] is the weight from input
  *                i to output neuron j. This array has a size of 785x10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */

  double activations[OUTPUTS];
  feedforward_1layer(sample, sigmoid, weights_io, activations);
  double max = -1.0;
  int result;
  for (int i = 0; i<OUTPUTS; i++){
    if(activations[i]>max){
      max = activations[i];
      result = i;
    }
  }
  return result;   	// <---	This should return the class for this sample
}

void feedforward_1layer(double sample[785], double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS], double activations[OUTPUTS])
{
 /*
  *  This function performs the feedforward pass of the network's computation - it propagates information
  *  from input to output, determines the input to each neuron, and calls the sigmoid function to
  *  calculate neuron activation.
  * 
  *  Inputs:
  *    sample -      The input sample (see above for a description)
  *    sigmoid -     The sigmoid function being used
  *    weights_op -  Array of current network weights
  *    activations - Array where your function will store the resulting activation for each output neuron
  * 
  *  Return values:
  *    Your function must update the 'activations' array with the output value for each neuron
  * 
  */ 
 
    for (int b = 0;b<OUTPUTS;b++){
      double result = 0.0;
      for (int a = 0;a<INPUTS;a++){
        result += sample[a]*weights_io[a][b];
      }
      activations[b] = sigmoid(SIGMOID_SCALE*result);
    }
    return;
}

void backprop_1layer(double sample[INPUTS], double activations[OUTPUTS], double (*sigmoid)(double input), int label, double weights_io[INPUTS][OUTPUTS])
{
  /*
   *  This function performs the core of the learning process for 1-layer networks. It takes
   *  as input the feed-forward activation for each neuron, the expected label for this training
   *  sample, and the weights array. Then it updates the weights in the array so as to minimize
   *  error across neuron outputs.
   * 
   *  Inputs:
   * 	sample - 	Input sample (see above for details)
   *    activations - 	Neuron outputs as computed above
   *    sigmoid -	Sigmoid function in use
   *    label - 	Correct class for this sample
   *    weights_io -	Network weights
   * 
   *  You have to:
   * 		* Determine the target value for each neuron
   * 			- This depends on the type of sigmoid being used, you should think about
   * 			  this: What should the neuron's output be if the neuron corresponds to
   * 			  the correct label, and what should the output be for every other neuron?
   * 		* Compute an error value given the neuron's target
   * 		* Compute the weight adjustment for each weight (the learning rate is in NeuralNets.h)
   */
    if (sigmoid(0.0)==0.5) {//logistic
      for (int a = 0; a<INPUTS; a++){
        for (int b = 0; b<OUTPUTS; b++){
          double expected = 0.0;
          if (b == label){
            expected = 1.0;
          }
          weights_io[a][b] += ALPHA*sample[a]*activations[b]*(1-activations[b])*(expected-activations[b]);
        }
      }
    }
    else {//tanh
      for (int a = 0; a<INPUTS; a++){
        for (int b = 0; b<OUTPUTS; b++){
          double expected = -1.0;
          if (b == label){
            expected = 1.0;
          }
          weights_io[a][b] += ALPHA*sample[a]*(1-activations[b]*activations[b])*(expected-activations[b]);
        }
      }
    }
   
}

int train_2layer_net(double sample[INPUTS],int label,double (*sigmoid)(double input), int units, double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS])
{
 /*
  *   This is your main training function for 2-layer networks. Now you have to worry about the hidden
  *  layer at this time. *Do not work on this until you have completed the 1-layer network*.
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   units   -  Number of units in the hidden layer
  *   weights_ih - Array of weights connecting inputs to hidden-layer neurons, weights_ih[i][j] is the 
  *                weight from input i to hidden neuron j. This array has a size of units 785 x 10.
  *   weights_ho - Array of weights connecting hidden-layer units to output neurons, weights_ho[i][j] is the 
  *                weight from hidden unit i to output neuron j. This array has a size of units x 10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */

  int result = classify_2layer(sample, label, sigmoid, units, weights_ih, weights_ho);
  double activations[OUTPUTS];
  double h_activations[MAX_HIDDEN];
  feedforward_2layer(sample, sigmoid, weights_ih, weights_ho, h_activations, activations, units);
  backprop_2layer(sample,h_activations, activations,sigmoid, label, weights_ih, weights_ho,units);
  return result;// <--- Should return the class for this sample
}

int classify_2layer(double sample[INPUTS],int label,double (*sigmoid)(double input), int units, double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS])
{
 /*
  *   This function takes an input sample and classifies it using the current network weights. It returns
  *  an int in [0,9] corresponding to which digit the network thinks is present in the input sample.
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   units   -  Number of units in the hidden layer
  *   weights_ih - Array of weights connecting inputs to hidden-layer neurons, weights_ih[i][j] is the 
  *                weight from input i to hidden neuron j. This array has a size of units 785 x 10.
  *   weights_ho - Array of weights connecting hidden-layer units to output neurons, weights_ho[i][j] is the 
  *                weight from hidden unit i to output neuron j. This array has a size of units x 10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */

  double activations[OUTPUTS];
  double h_activations[MAX_HIDDEN];
  feedforward_2layer(sample, sigmoid, weights_ih, weights_ho, h_activations, activations, units);
  double max = -1.0;
  int result;
  for (int i = 0; i<OUTPUTS; i++){
    if(activations[i]>max){
      max = activations[i];
      result = i;
    }
  }
  return result;		// <--- Should return the class for this sample
}


void feedforward_2layer(double sample[INPUTS], double (*sigmoid)(double input), double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS], double h_activations[MAX_HIDDEN],double activations[OUTPUTS], int units)
{
 /*
  *  Here, implement the feedforward part of the two-layer network's computation.
  * 
  *  Inputs:
  *    sample -      The input sample (see above for a description)
  *    sigmoid -     The sigmoid function being used
  *    weights_ih -  Array of current input-to-hidden weights
  *    weights_ho -  Array of current hidden-to-output weights
  *    h_activations - Array of hidden layer unit activations
  *    activations   - Array of activations for output neurons
  *    units -         Number of units in the hidden layer
  * 
  *  Return values:
  *    Your function must update the 'activations' and 'h_activations' arrays with the output values for each neuron
  * 
  */ 
 
  for (int h = 0;h<units;h++){
      double result = 0.0;
      for (int i = 0;i<INPUTS;i++){
        result += sample[i]*weights_ih[i][h];
      }
      h_activations[h] = sigmoid(SIGMOID_SCALE*result);
    }
  for (int o = 0;o<OUTPUTS;o++){
      double result = 0.0;
      for (int h = 0;h<units;h++){
        result += h_activations[h]*weights_ho[h][o];
      }
      activations[o] = sigmoid(SIGMOID_SCALE*(MAX_HIDDEN/units)*result);
    }
    return;
}

void backprop_2layer(double sample[INPUTS],double h_activations[MAX_HIDDEN], double activations[OUTPUTS], double (*sigmoid)(double input), int label, double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS], int units)
{
  /*
   *  This function performs the core of the learning process for 2-layer networks.
   * 
   *  Inputs:
   * 	sample - 	Input sample (see above for details)
   *    h_activations - Hidden-layer activations
   *    activations -   Output-layer activations
   *    sigmoid -	Sigmoid function in use
   *    label - 	Correct class for this sample
   *    weights_ih -	Network weights from inputs to hidden layer
   *    weights_ho -    Network weights from hidden layer to output layer
   *    units -         Number of units in the hidden layer
   */

   if (sigmoid(0.0)==0.5) {//logistic

      for (int i = 0; i<INPUTS; i++){
        for (int h = 0; h<units; h++){
          double error = 0.0;
          for (int o = 0; o<OUTPUTS; o++){
            double expected = 0.0;
            if (o == label){
              expected = 1.0;
            }
            error += weights_ho[h][o]*activations[o]*(1-activations[o])*(expected-activations[o]);
          }
          weights_ih[i][h] += ALPHA*sample[i]*h_activations[h]*(1-h_activations[h])*error;
        }
      }
      for (int h = 0; h<units; h++){
        for (int o = 0; o<OUTPUTS; o++){
          double expected = 0.0;
          if (o == label){
            expected = 1.0;
          }
          weights_ho[h][o] += ALPHA*h_activations[h]*activations[o]*(1-activations[o])*(expected-activations[o]);
        }
      }
    }
    else {//tanh

      for (int i = 0; i<INPUTS; i++){
        for (int h = 0; h<units; h++){
          double error = 0.0;
          for (int o = 0; o<OUTPUTS; o++){
            double expected = -1.0;
            if (o == label){
              expected = 1.0;
            }
            error += weights_ho[h][o]*(1-activations[o]*activations[o])*(expected-activations[o]);
          }
          weights_ih[i][h] += ALPHA*sample[i]*(1-h_activations[h]*h_activations[h])*error;
        }
      }
      for (int h = 0; h<units; h++){
        for (int o = 0; o<OUTPUTS; o++){
          double expected = -1.0;
          if (o == label){
            expected = 1.0;
          }
          weights_ho[h][o] += ALPHA*h_activations[h]*(1-activations[o]*activations[o])*(expected-activations[o]);
        }
      }
    }
   
}

double logistic(double input)
{
 // This function returns the value of the logistic function evaluated on input
 // TO DO: Implement this function!
  return 1/(1+exp(-input));
 		// <--- Should return the value of the logistic function on the input
}
