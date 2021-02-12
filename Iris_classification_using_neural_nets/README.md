In this project, I have tried to predict the correct category in the iris dataset.
First of all, I split the dataset into train and test sets for testing my final model. The split was made to have
80 percent training samples and 20 percent testing samples.
The neural network that I used had the following architecture:
1) 3 layers, out of two are hidden layers and one is the output layer which outputs the probabiliity of a sample
to belong to each class.
2) Each of the hidden layers had 10 neurons each and I used "relu" activation function in each of these layers.
3) In the output layer, I used "softmax" function to output the probability of a sample to belong to each class.

The used twp additional hyper-parameters:
1) Learning rate of 0.001 which helps us to optimize our cost function, and
2) Batch size, whcih controls how many training samples are going in each iteratiom.

RESULTS:
Using the above architecture and hyper-parameters, I was able to achieve an accuracy of 0.97 on the testing set.