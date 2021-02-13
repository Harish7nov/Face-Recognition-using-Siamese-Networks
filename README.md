# Face-Recognition-using-Siamese-Networks
Siamese Network architecture is widely used in the inddustry to decide whether two input images are similiar or not.

In this repository I have trained a Siamese Network to recognise faces with the help of Convolutional Neural Networks.

The model outputs two N - Dimensional feature vectors from the CNN which has encoded important features regarding the image that was fed into it.

Then euclid distance is taken between both the vectors which is fed into a Dense layer with a sigmoid activation function.

It is also widely known as one shot learning algorithm.

![Image of Siamese Network](https://www.pyimagesearch.com/wp-content/uploads/2020/11/keras_siamese_networks_sisters.png)
