"""
Script to train the siamese neural network
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable the Eager Execution to work with tf keras
tf.compat.v1.disable_eager_execution()
loc = os.getcwd()


# using tf keras utils Sequence to facilitate multithreading with the generator
class Generator(tf.keras.utils.Sequence):

    def __init__(self, x, classes, batchsize, steps):
        self.x = x
        self.classes = classes
        self.batchsize = batchsize
        self.n_index = np.unique(np.array(classes))
        self.steps = steps
        self.temp = 35

    def __len__(self):
        return self.steps

    def read_img(self, path):

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = img[self.temp:-self.temp, self.temp:-self.temp] / 255

        return img

    def __getitem__(self, idx):
    
        # Create two zero arrays for the two conjointed networks
        inp = [np.zeros([self.batchsize, 154, 154, 3]) for _ in range(2)]
        # Create the target variable
        out = np.zeros(self.batchsize, dtype=np.int32)
        
        # 0 -> Not Similiar
        # 1 -> Similiar
        # Only if you want random indices to be made as 1
        # indices = np.random.choice(list(range(self.batchsize)), size=self.batchsize // 2, replace=False)
        # out[indices] = 1
        
        out[:self.batchsize//2] = 1

        cond = 1
        while cond:

            cat = np.random.choice(self.n_index, size=[self.batchsize * 2], replace=False)
            index = [list(np.where(np.array(self.classes) == i)[0]) for i in cat]
            for i in range(self.batchsize * 2):
                if len(index[i]) <= 1:
                    cond = 1
                    break
                else:
                    cond = 0

        j = self.batchsize
        pos = list(range(self.batchsize))
        np.random.shuffle(pos)

        for i in range(self.batchsize):
            index1 = np.random.choice(index[i], size=[2], replace=False)
            inp[0][i, :, :, :] = self.read_img(self.x[index1[0]])

            if out[i] == 1:
                inp[1][i, :, :, :] = self.read_img(self.x[index1[-1]])

            else:
                index2 = np.random.choice(index[j])
                inp[1][i, :, :, :] = self.read_img(self.x[index2])
                j += 1

        return inp, out


def euclid_distance(inp):

    eps = 1e-09
    tensor1, tensor2 = inp
    K = tf.keras.backend
    dist = K.sqrt(K.maximum(K.sum(K.square(tensor1 - tensor2), axis=-1, keepdims=True), eps))

    return dist


def get_output_shape(shape):

    return shape[0], 1


def siamese_model():

    weight_decay = 1e-03
    shape = [154, 154, 3]
    reg = tf.keras.regularizers.l2(weight_decay)

    l_input = tf.keras.layers.Input(shape=shape)
    r_input = tf.keras.layers.Input(shape=shape)
    
    # Dont use bias in CNN when using Batch Norm with it
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg, use_bias=False, input_shape=tuple(shape)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg, use_bias=False))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg, use_bias=False))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg, use_bias=False))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=reg, use_bias=False))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, kernel_regularizer=reg))

    # Get the encoded features from both the images
    l_encoder = model(l_input)
    r_encoder = model(r_input)

    # Calculate the euclid distance between the two vectors
    L1_layer = tf.keras.layers.Lambda(euclid_distance, output_shape=get_output_shape)([l_encoder, r_encoder])
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L1_layer)

    model = tf.keras.models.Model(inputs=[l_input, r_input], outputs=prediction)

    return model


if __name__ == "__main__":

    train_file_list = []
    valid_file_list = []

    # Path to the metadata CSV file
    # Edit as required for your system
    path = r"list_eval_partition.csv"
    data = pd.read_csv(path)

    # Path to the original folder where images are present
    # Edit as required for your system
    file_path = r"data\img_align_celeba"
    train_persons = []
    valid_persons = []

    for i in range(len(data["Person"])):
        if data["partition"][i] == 0:
            train_file_list.append(os.path.join(file_path, data["image_id"][i]))
            train_persons.append(data["Person"][i])

        elif data["partition"][i] == 1:
            valid_file_list.append(os.path.join(file_path, data["image_id"][i]))
            valid_persons.append(data["Person"][i])

    batchsize = 32
    epoch = 200
    lr = 1e-03
    train_steps = 300
    valid_steps = 200

    train_gen = Generator(train_file_list, train_persons, batchsize, train_steps)
    valid_gen = Generator(valid_file_list, valid_persons, batchsize, valid_steps)

    model = siamese_model()
    print(model.summary())

    base_lr = 1e-04
    max_lr = 1e-03

    mcp1 = tf.keras.callbacks.ModelCheckpoint("siamese_valid.h5", monitor="val_accuracy", save_best_only=True, mode="max")
    mcp2 = tf.keras.callbacks.ModelCheckpoint("siamese_train.h5", monitor="accuracy", save_best_only=True, mode="max")

    # opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.98, nesterov=True)
    # opt = tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=0.95)

    lr_plat = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                                   mode="max",
                                                   patience=10,
                                                   verbose=1,
                                                   factor=0.9)

    # Specify the log directory for the tensorboard to store the traning details
    log_dir = f'logs\Face Recognition - {time.strftime("%H-%M-%S", time.localtime())}'
    tensorboard = tf.compat.v1.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy",
                                                                 tf.keras.metrics.Precision(),
                                                                 tf.keras.metrics.Recall()])

    # Train the model
    history = model.fit(train_gen,
                        steps_per_epoch=train_steps,
                        epochs=epoch,
                        validation_data=valid_gen,
                        validation_steps=valid_steps,
                        workers=5,
                        use_multiprocessing=False,
                        callbacks=[mcp1, mcp2, tensorboard])

    # Save the histroy file as a csv file
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = os.path.join(loc, r'history.csv')

    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
