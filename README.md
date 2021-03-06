# Traffic Sign Classifier
This project uses deep neural networks and convolutional neural networks to classify [German Traffic Signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

Usage: `python traffic_sign_trainer.py` to train and evalute the model

## German Traffic Signs Dataset
Label 00 - 09: <img src="images/0.png" width="64" /> <img src="images/1.png" width="64" /> <img src="images/2.png" width="64" /> <img src="images/3.png" width="64" /> <img src="images/4.png" width="64" /> <img src="images/5.png" width="64" /> <img src="images/6.png" width="64" /> <img src="images/7.png" width="64" /> <img src="images/8.png" width="64" /> <img src="images/9.png" width="64" />

Label 10 - 19: <img src="images/10.png" width="64" /> <img src="images/11.png" width="64" /> <img src="images/12.png" width="64" /> <img src="images/13.png" width="64" /> <img src="images/14.png" width="64" /> <img src="images/15.png" width="64" /> <img src="images/16.png" width="64" /> <img src="images/17.png" width="64" /> <img src="images/18.png" width="64" /> <img src="images/19.png" width="64" />

Label 20 - 29: <img src="images/20.png" width="64" /> <img src="images/21.png" width="64" /> <img src="images/22.png" width="64" /> <img src="images/23.png" width="64" /> <img src="images/24.png" width="64" /> <img src="images/25.png" width="64" /> <img src="images/26.png" width="64" /> <img src="images/27.png" width="64" /> <img src="images/28.png" width="64" /> <img src="images/29.png" width="64" />

Label 30 - 39: <img src="images/30.png" width="64" /> <img src="images/31.png" width="64" /> <img src="images/32.png" width="64" /> <img src="images/33.png" width="64" /> <img src="images/34.png" width="64" /> <img src="images/35.png" width="64" /> <img src="images/36.png" width="64" /> <img src="images/37.png" width="64" /> <img src="images/38.png" width="64" /> <img src="images/39.png" width="64" />

Label 40 - 42: <img src="images/40.png" width="64" /> <img src="images/41.png" width="64" /> <img src="images/42.png" width="64" /> 

This is a multi-class, single-image classification problem. The dataset is available [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)

* The size of training set is 34,799
* The size of validatiaon set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique traffic signs (labels) is 43

Below is the distrubtion of occurances of each image class in the training, validation and testing dataset. This can help to understand whether the dataset is uniform in terms of a baseline occurrence, and avoid potential pitfalls on biased distribution among different dataset

<img src="images/train_dataset.png" width="400" /> <img src="images/valid_dataset.png" width="400" /> <img src="images/test_dataset.png" width="400" />

## Architecture Overview
The deep neural network architectures receives an image as input, transform it through a series of hidden layers (e.g Conv, max pooling, relu, full connected etc), and output a vector of logits. Then the logits will be used to measured the probability error in traffic sign classification with softmax and cross entropy.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x20    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x20 				|
| RELU					|												|
| Fully connected		| input 500, output 160        									|
| RELU					|												|
| Dropout				| 50%          									|
| Fully connected		| input 160, output 100        									|
| RELU					|												|
| Dropout				| 50%         									|
| Fully connected		| input 100, output 43        									|

With this neural network architecture, along with other optimization, **the accuracy rate on the test dataset is able to achieve 96.5%**

## Training data preprocessing

1. Data Normalization. The input value for each pixel is in the range of [0-255]. I have converted them into [-1, 1] range by `(X - 128) / 128`. Ideally, I shall substract the mean to center the input to 0, and divide the value by the standard deviation of each pixel value. However, I made an assumption that the pixel value is evenly distributed across from 0 to 255, so I use simple normalization parameters for both training and testing data. The benefit is that my program doesn't have to "memorize" the special mean and deviation value generated from the training data, and apply them to the unknown test data.

2. Data augmentation is required on more complex object recognition tasks. I achieved it by shifting the image to four directions, i.e. left, rigth, top and down. In addition, Images were shiftted with 1 pixel and 2 pixels, and therefore I increased the training data by 8x more. During the training processes, more data effectively effectively improved the final accurancy on the test dataset.

3. I also tried to convert the images to different color spaces, e.g. gray (32x32x1), HLS (32x32x3), and YUV (32x32x3), hoping for prediction accurancy improvement. However, these converted images weren't able to beat the original training dataset and so I gave up this technique in the final training model.

## Model Training and Optimization
I am able to achieve the accuracy rate of 96.4% on test dataset and 98% on the validation dataset. Below are the optimization approaches incorporated in the final version.

| Approach         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Training dataset augmentation | This image has been shifted the left, right, up, down direction with 1 and 2 pixels. So it adds 8x more training data |
| Dropout | Applies 50% dropout rate on output of the two fully connected layers |
| L2 Regularization | Applies regularization (beta 0.01) with weights and bias of the three fully connected layers |
| Loss Function | Computes softmax cross entropy between logits and labels |

Training parameters are
* Number of epoches: 30
* Size of the training batch: 128
* Learning rate: 0.001

In addition, to avoid the over-training, instead of picking the last train model to evaluate the test dataset, when the losses  becomes mostly flat, I picked the model with highest accurancy on the validation dataset.

## Solution Approach

Started from basic LeNet architecture. Here are the approaches adopted with accurancy being improved

|  | Adopted Approach | Description	 	| Test Accuracy |
|:--:|:---------------------:|:---------------------------------------------:|:-----------:|
| Step 1 | Basic LeNet | The same LeNet used for hand-writing image (28x28x1) detection | *89.3%* |
| Step 3 | L2 Regularization / Dropout | Avoid the model to be overfitting during the training | *93.4%* |
| Step 2 | The training data augmentation | Shift the images (left/right/up/down) for more training data | *95.0%* |
| Step 4 | Increase numbers of neurals | LeNet was optimized for 28x28x1 images. Given the input images are 32x32x3, it may need more numbers of weights at each layer for better output quality | **96.5%** |

Alternative optimization approaches has also been evaluated as below. Although these approaches were tuned along with other optimizations including dropout, L2 Regularization, different numbers of neurals in the hidden layer. The final accurancy is still unable to beat the final version above. Therefore, they weren't been adopted. 

| Evaluated Approach but abandoned | Description	 	| Test Accuracy |
|:----------------------:|:---------------------------------------------:|:-----------:|
| Convert to Gray color space | Convert the RGB images (32x32x3) to grayscaled images (32x32x1). | *95.9%* |
| Convert the YUV color space | Convert the RGB (32x32x3) to YUV color space (32x32x3) | *94.9%* |
| Tune the batch size / learning rate | Increase the batch size or change learning rate, but none of them achieve obvioius accruancy rate improvement | N/A |

## Test a Model on New Images

### 100% Accurancy on newly accquired images
I accquired 36 more German traffic sign images below from the web for evaluation. The trained the model has successfully predict their types.

<img src="test_images/0.png" width="32" /><img src="test_images/1.png" width="32" /><img src="test_images/10.png" width="32" /><img src="test_images/11.png" width="32" /><img src="test_images/12.png" width="32" /><img src="test_images/13.png" width="32" /><img src="test_images/14.png" width="32" /><img src="test_images/15.png" width="32" /><img src="test_images/16.png" width="32" /><img src="test_images/17.png" width="32" /><img src="test_images/18.png" width="32" /><img src="test_images/2.png" width="32" /><img src="test_images/21.png" width="32" /><img src="test_images/23.png" width="32" /><img src="test_images/25.png" width="32" /><img src="test_images/26.png" width="32" /><img src="test_images/29.png" width="32" /><img src="test_images/3.png" width="32" /><img src="test_images/30.png" width="32" /><img src="test_images/31.png" width="32" /><img src="test_images/32.png" width="32" /><img src="test_images/33.png" width="32" /><img src="test_images/34.png" width="32" /><img src="test_images/35.png" width="32" /><img src="test_images/36.png" width="32" /><img src="test_images/37.png" width="32" /><img src="test_images/38.png" width="32" /><img src="test_images/39.png" width="32" /><img src="test_images/4.png" width="32" /><img src="test_images/40.png" width="32" /><img src="test_images/42.png" width="32" /><img src="test_images/5.png" width="32" /><img src="test_images/6.png" width="32" /><img src="test_images/7.png" width="32" /><img src="test_images/8.png" width="32" /><img src="test_images/9.png" width="32" />

### Model Certainty - Softmax Probabilities
Below are 5 new images with some difficulties to classify along with the prediction, images generated from the output of the convolutional layers, and softmax Probabilities

#### 1. "Speed limit (100km/h)" sign -- correct prediction
| Original | Prediction 1	| Prediction 2 | Prediction 3 | Prediction 4 | Prediction 5 |
|:--------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|          | 0.894 (**correct**)| 0.105 (wrong)| 0.0002 (wrong)| 7.73e-05 (wrong)| 1.44e-05 (wrong)|
|<img src="test_images/pred_7_.png" width="64" />|<img src="images/7.png" width="64" />|<img src="images/8.png" width="64" />|<img src="images/40.png" width="64" />|<img src="images/5.png" width="64" />|<img src="images/0.png" width="64" />|

Output from the first Conv layer (edge and shape detection)

<img src="test_images/conv_1_7.png" width="800" />

Output from the second Conv layer (higher level feature detection)

<img src="test_images/conv_2_7.png" width="800" />

#### 2. "Speed limit (60km/h)" sign -- incorrect prediction
| Original | Prediction 1	| Prediction 2 | Prediction 3 | Prediction 4 | Prediction 5 |
|:--------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|          | 0.749 (wrong)| 0.0322 (wrong)| 0.02387 (wrong)| 0.0206 (wrong)| 0.0196 (**correct**)|
|<img src="test_images/pred_3_.png" width="64" />|<img src="images/5.png" width="64" />|<img src="images/38.png" width="64" />|<img src="images/16.png" width="64" />|<img src="images/20.png" width="64" />|<img src="images/3.png" width="64" />|

Output from the first Conv layer (edge and shape detection)

<img src="test_images/conv_1_3.png" width="800" />

Output from the second Conv layer (higher level feature detection)

<img src="test_images/conv_2_3.png" width="800" />

#### 3. "General caution" sign -- correct prediction
| Original | Prediction 1	| Prediction 2 | Prediction 3 | Prediction 4 | Prediction 5 |
|:--------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|          | 0.4206 (**correct**)| 0.1823 (wrong)| 0.0718 (wrong)| 0.0549 (wrong)| 0.0527 (wrong)|
|<img src="test_images/pred_18_.png" width="64" />|<img src="images/18.png" width="64" />|<img src="images/20.png" width="64" />|<img src="images/25.png" width="64" />|<img src="images/26.png" width="64" />|<img src="images/31.png" width="64" />|

Output from the first Conv layer (edge and shape detection)

<img src="test_images/conv_1_18.png" width="800" />

Output from the second Conv layer (higher level feature detection)

<img src="test_images/conv_2_18.png" width="800" />

#### 4. "Road work" sign -- incorrect prediction
| Original | Prediction 1	| Prediction 2 | Prediction 3 | Prediction 4 | Prediction 5 |
|:--------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|          | 0.4018 (wrong)| 0.3703 (wrong)| 0.1134 (wrong)| 0.0358 (**correct**)| 0.0332 (wrong)|
|<img src="test_images/pred_25_.png" width="64" />|<img src="images/4.png" width="64" />|<img src="images/1.png" width="64" />|<img src="images/38.png" width="64" />|<img src="images/25.png" width="64" />|<img src="images/36.png" width="64" />|

Output from the first Conv layer (edge and shape detection)

<img src="test_images/conv_1_25.png" width="800" />

Output from the second Conv layer (higher level feature detection)

<img src="test_images/conv_2_25.png" width="800" />

#### 5. "Beware of ice/snow" sign -- incorrect prediction
| Original | Prediction 1	| Prediction 2 | Prediction 3 | Prediction 4 | Prediction 5 |
|:--------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|          | 0.3555 (wrong)| 0.1520 (**correct**)| 0.1171 (wrong)| 0.08077 (wrong)| 0.0602 (wrong)|
|<img src="test_images/pred_30_.png" width="64" />|<img src="images/23.png" width="64" />|<img src="images/30.png" width="64" />|<img src="images/20.png" width="64" />|<img src="images/37.png" width="64" />|<img src="images/31.png" width="64" />|

Output from the first Conv layer (edge and shape detection)

<img src="test_images/conv_1_30.png" width="800" />

Output from the second Conv layer (higher level feature detection)

<img src="test_images/conv_2_30.png" width="800" />
