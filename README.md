# Sign Language Image Classification using Deep Learning

_Author: Clara Le_

_Date: 17/8/2023_

---
Deep Learning models were applied on the dataset after performing data cleaning, spliting, and preprocessing.
### Baseline model
For a baseline model, a densely connected model with Stochastic Gradient Descent optimizer is built, its learning rate is 0.01 by default, batch size is 32 and the number of epochs are 30. In this network, there is 3 hidden layers. The first hidden layer contains 64 neurons and uses Elu activation function. The second hidden layer contains 32 neurons and uses Elu activation function. The third hidden layer contains 16 neurons and uses Elu activation function. The output layer contains 24 neurons (since there are 24 possible outcomes of labels) and uses Softmax activation function. The loss function is Sparse categorical crossentropy since labels are encoded as integers from 0 to 23.

<a href="url"><img src="https://github.com/Tien-le98/CNN_Sign_language/blob/main/Baseline_model" align="center" height="300" width="500" ></a>

According to the plot of training loss, training accuracy, validation loss and validation accuracy of the baseline model above, the training accuracy and validation accuracy of this baseline model converge after 20 epochs. However, the difference between training loss and validation loss, as well as training accuracy and validation accuracy are quite large, which can indicate that this baseline model can be overfitting. For example, after epoch 30, the training accuracy is 0.9942 while the validation accuracy is only 0.7016, hence the gap between these two values is about 0.3.

### Train and optimize a Densely connected and a Convolutional neural network model
In training and optimizing process, there are some hyperparameters that need to be tuned in order to find the optimized model which are the activation function ('actfn' list including Elu, Leaky Relu and Selu activation functions), the optimizer ('optimizer' list including Stochastic Gradient Descent, Adam, and Nadam optimizer) and its learning rate ('learningrate' list containing values of 0.01 and 0.001). These hyperparameters are used for optimizing both the Densely connected model and the Convolutional neural network (CNN) model. Early stopping is also defined with patience of 5, and monitored by validation loss values. Early stopping is used to prevent models from overfitting since the baseline model raises a signal of overfitting problem.

#### 1. Train and optimize a Densely connected model

The densely connect model is built with 1 Flatten layer for input data, three hidden layers with size of 64, 32, and 16 respectively, and 1 Dense layer including 24 neurons for the output. The used loss function is Sparse categorical crossentropy since labels are encoding as integers from 0 to 23. In general, the maximum validation accuracy scores of these densely connected models range from about 0.51 to 0.73. Additionally, all of these densely connected models can be overfitting since the differences between their training accuracy and their validation accuracy are quite large (from around 0.2 to 0.3). 

According to the table of the maximum validation accuracy for each combination of activation function, optimizer and learning rate, the model which has the highest validation accuracy is a densely connected model with **Elu** activation function, its optimizer is **Stochastic Gradient Descent** with learning rate of **0.01**. The maximum validation accuracy of this model is about 0.73. However, according to this model's plot of performance, its training accuracy is about 0.925, hence there is a large difference between its training accuracy and its validation accuracy (nearly 0.2), which can indicate that this model can be overfitting. Therefore, some methods such as Regularization, Weight initialization and Dropout regularization are used to mitigate this overfitting problem for this best densely connected model.

<a href="url"><img src="https://github.com/Tien-le98/CNN_Sign_language/blob/main/DNN" align="center" height="300" width="700" ></a>

**Apply Regularization methods on the best densely connected model**

+ According to the performance of this best densely connected model after using **L2 Regularizer**, after epoch 22, its training accuracy is about 0.6532 and its validation accuracy is about 0.524. The difference between these two values is about 0.13. Hence, this method of l2 Regulization just can slightly reduce the overfitting problem of this model since the gap between these two values of this model before apply l2 regularization is about 0.2. However, the validation accuracy of this model after applying l2 Regularization is much lower than its original validation accuracy since the model's validation accuracy before applying this l2 regularization method is about 0.73.

+ In terms of **Weight initialization** method, after epoch 15, this model converges with its training accuracy is about 0.9289 and its validation accuracy is about 0.7022. The difference between these two values is about 0.23. Hence, this method of Weight initialization does not reduce the gap between the training accuracy and the validation accuracy of the model since the difference between these two values before applying Weight initialization is about 0.2.

+ After using **Dropout regularization** method, after epoch 24, this model converges with its training accuracy is about 0.8912 and its validation accuracy is about 0.7292. The difference between these two values is about 0.16. Hence, this method of Dropout regulation also just slightly reduce the gap between the training accuracy and the validation accuracy of the model since the gap between these two values of this model before apply Dropout regularization is about 0.2, but it does not improve the validation accuracy much.

#### 2. Train and optimize a Convolutional Neural Network (CNN) model

The CNN model is built with Convolutional layers 3x3, strides equal to 1, Max Pooling layers 2x2, 1 Flatten layer, and 1 Dense layer including 24 neurons for the output. The used loss function is Sparse categorical crossentropy since labels are encoding as integers (from 0 to 23). Early stopping is also defined with patience of 5, and monitored by validation loss values. Early stopping is used to prevent model from overfitting. In general, these CNN models gain better performance since their maximum validation accuracy are larger than the figures for other densely connected models. Most of the maximum validation accuracy scores of these CNN models range from about 0.76 to 0.91. Additionally, the differences between training accuracy and validation accuracy of these CNN models are not large as the figures for other densely connected models, since they are only from around 0.1 to 0.2. Although these differences are not large, they still can indicate overfitting problem, therefore, in order to solve this problem, three methods such as Regularization, Weight initialization and Dropout regularization are used.

**The best CNN model** is the model with **Selu** activation function, **Adam** optimizer and learning rate of **0.001**. Its training accuracy is 1, and its maximum validation accuracy is around 0.9136. The difference between its validation accuracy and training accuracy is about 0.0864. 

<a href="url"><img src="https://github.com/Tien-le98/CNN_Sign_language/blob/main/Best_CNN" align="center" height="300" width="490" ></a>

**The second best CNN model** is the model with **Selu** activation function, **Nadam** optimizer and learning rate of **0.001**. Its training accuracy is 1, and its maximum validation accuracy is about 0.9124. The difference between its validation accuracy and training accuracy is also about 0.0876. 

<a href="url"><img src="https://github.com/Tien-le98/CNN_Sign_language/blob/main/second_best_CNN" align="center" height="300" width="500" ></a>

These two best models are chosen to apply some regularization methods to mitigate overfitting problem.

**Apply Regularization methods on the second best CNN model (Selu activation function, Nadam optimizer and learning rate 0.001)**

+ According to the performance of the second best CNN model (with Selu activation function, Nadam optimizer and learning rate of 0.001) after using **L2 Regularizer**, this model converges after epoch 12. Its training accuracy is 1 and its validation accuracy is about 0.9027. The difference between these two values is about 0.0973. Hence, the method of l2 Regulization does not reduce the gap between the training accuracy and validation accuracy of this second best CNN model since the difference between these two values before applying l2 Regularization is 0.0876.

+ After using **Weight Initialization**, this model converges after epoch 6, its training accuracy is about 1 and its validation accuracy is about 0.908. The difference between these two values is about 0.092. Hence, the method of Weight Initialization also does not reduce the difference between the training accuracy and the validation accuracy of this second best CNN model since the difference between these two values before applying Weight Initialization is 0.0876.

+ To **Dropout Regularization**, this model converges after epoch 7, its training accuracy is about 0.9979 and its validation accuracy is about 0.9021. The difference between these two values is about 0.0958. Hence, the method of Dropout Regularization also does not reduce the difference between the training accuracy and the validation accuracy of this second best CNN model since the difference between these two values before applying Dropout Regularization is 0.0876.

**Apply Regularization methods on the best CNN model (Selu activation function, Adam optimizer and learning rate 0.001)**

+ According to the performance of the best CNN model (with Selu activation function, Adam optimizer and learning rate of 0.001) after using **L2 Regularizer**, this model converges after epoch 11, its training accuracy is about 1 and its validation accuracy is about 0.9191. The difference between these two values is about 0.0809. Hence, the method of l2 Regulization can slightly reduce the difference between the training accuracy and the validation accuracy of this best CNN model since the difference between these two values before applying l2 Regularization is 0.0864. However, it does not improve the validation accuracy much because the original validation accuracy is 0.9136.

+ After using **Weight Initialization**, after epoch 7, its training accuracy is about 1 and its validation accuracy is about 0.9094. The difference between these two values is about 0.0906. Hence, the method of Weight Initialization does not reduce the difference between training accuracy and validation accuracy of this best CNN model since the gap between these two values before applying Weight Initialization is 0.0864.

+ In terms of **Dropout Regularization**, this model converges after epoch 6, its training accuracy is about 0.9908 and its validation accuracy is about 0.8804. The difference between these two values is about 0.1104. Hence, the method of Dropout Regularization also does not reduce the gap between the training accuracy and the validation accuracy of this best CNN model since the difference between these two values before applying Dropout Regularization is 0.0864.

> In conclusion, in comparison with the best densely connected model, the best CNN model (with Selu activation function, Adam optimizer and learning rate of 0.001), and the second best CNN model (with Selu activation function, Nadam optimizer and learning rate of 0.001) obtain better validation accuracy score. Therefore, these two best models are used in prediction process. In addition, three methods (Regularization, Weight initialization and Dropout regularization) do not reduce the difference between training accuracy and validation accuracy, hence they can not mitigate the overfitting problem. Therefore, these two best CNN models are used to generate predictions without applying any regularization methods.

#### 3. Perform a statistical test between the best and the second best models

The best CNN model with Selu activation function, Adam optimizer and learning rate of 0.001 converges with the validation accuracy of 0.9175, and the second best CNN model with Selu activation function, Nadam optimizer and learning rate of 0.001 converges with the validation accuracy is 0.9099. Additionally, the difference in the performance of these two best models is significant because the p-value in comparing these two best models is below 0.05, therefore, the best CNN model with Selu activation function, Adam optimizer and learning rate of 0.001 is chosen to make predictions on the testing data.

#### 4. Model predictions

<a href="url"><img src="https://github.com/Tien-le98/CNN_Sign_language/blob/main/Conf_mat_true_label" align="center" height="500" width="700" ></a>

According to the confusion matrix normalized by true labels, only 56% observations of the letter 'T' are predicted correctly. In addition, the letter 'O' has only 69% observations which are predicted correctly, and the letter 'R' has only 78% observations which are predicted correctly. The remaining letters have around 80% observations or more being predicted correctly. The letters 'A', 'C', 'F', 'L', 'W' are completely predicted correctly.

<a href="url"><img src="https://github.com/Tien-le98/CNN_Sign_language/blob/main/Conf_mat_predicted_label" align="center" height="500" width="700" ></a>

According to the confusion matrix normalized by predicted labels, the letter 'T' has the lowest accuracy score (only 0.67) because among all letters are predicted as 'T', only 67% observations are truely 'T'. The letter 'S' has the second lowest accuracy score because among all letters are predicted as 'S', only about 74% observations are truely 'S'.

<a href="url"><img src="https://github.com/Tien-le98/CNN_Sign_language/blob/main/Misclassification" align="right" height="400" width="200" ></a>

According to the table illustrating the number of the proportion of misclassification for each letter, the letter 'T' is the most common letter which are predicted incorrectly. Because in all of misclassifications of this final CNN model, the number of missification of the letter 'T' is 51 (accounts for around 15% of total number of misclassification). Particularly, the letter 'T' is mostly misclassified as 'H', 'X', 'L', and 'Y'. The second most common letter which are misclassified is the letter 'H' with about 11% of total number of misclassification. This letter 'H' is more misclassified as 'G' and 'M'. The third most misclassified letter is the letter 'O' with about 10.8% of total number of misclassification, and this letter is more likely to be predicted as 'N' and 'F'.

Although the accuracy of this final Convolutional Neural Network (CNN) model is high, this model should be still improved to obtain higher accuracy score in this classification task. For further improvement, the number of hidden layers and the size of these hidden layers should be tuned to examine if they can improve the performance of this model, since there are only three hyperparameters that are currently tuned for this CNN model (the activation function, the optimizer and its learning rate). In addition, some other methods such as Batch Normalization and Residual Unit should be considered, since Residual Unit can mitigate overfitting problem, which means that it can reduce the gap between training accuracy and validation accuracy of this model. Last, the training dataset and the testing dataset may not comprehensive since the letter 'J' and 'Z' do not appear in these datasets, which can lead to some misclassifications and a decrease in the overall accuracy of the model.
