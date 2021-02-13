"""
A Python script to test the siamese network in real time with the webcam of your system.
"""

# Import from other code files
from train import *
# Importing the libraries
import os
import tensorflow as tf
import cv2
import time
import numpy as np

# Disable the eager execution
tf.compat.v1.disable_eager_execution()
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'


def get_faces(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.1, 10)
    temp = 10

    if faces is ():
        return 0, None

    # Crop all faces found
    cropped_faces = []
    count = 0
    for (x, y, w, h) in faces:

        faces[count, 0] = x - temp
        faces[count, 1] = y - temp
        faces[count, 2] = w + temp
        faces[count, 3] = h + temp

        cropped_faces.append(img[y - temp : y + h + temp, x - temp : x + w + temp])
        count += 1

    return faces, cropped_faces


def generate_batch(path):

    classes = np.array(os.listdir(path))
    imgs = np.zeros(shape=[len(classes), 154, 154, 3])

    for i in range(len(classes)):
        temp = os.path.join(path, classes[i])
        temp = cv2.cvtColor(cv2.imread(os.path.join(temp, os.listdir(temp)[0])), cv2.COLOR_BGR2RGB)
        imgs[i, :, :, :] = cv2.resize(temp, (154, 154), interpolation=cv2.INTER_CUBIC) / 255

    return imgs, classes

# Loading the cascades
cascade_path = os.path.join(r"path_to_cascades_folder_in_li",  r"haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

model = siamese_model()
# lr = 1e-05
# opt = tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=0.9)
# model.compile(optimizer=opt, loss=contrastive_loss, metrics=[accuracy])
model.load_weights("siamese_valid.h5")
print(model.summary())

threshold = 0.5

# Read the classes
# Path to your custom dataset where one example of each class (Person)
# is stored in separate folders with their respective names

path = r""
all_faces, classes = generate_batch(path)

captured = ""
count = 0
video_capture = cv2.VideoCapture(0)

# Doing some Face Recognition with the webcam
while True:

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    _, frame = video_capture.read()
    boxes, faces = get_faces(frame)

    if faces is not None:
        for i, j in zip(faces, boxes):
            if 0 not in i.shape:
                pred = []
                face = cv2.cvtColor(cv2.resize(np.array(i, dtype=np.float32), (154, 154), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
                face = np.expand_dims(face, axis=[0]) / 255

                for k in range(len(classes)):
                    inp = [np.expand_dims(all_faces[k], axis=0), face]
                    # start = time.time()
                    pred.append(model.predict(inp))
                    # print(f"Time Taken : {time.time() - start}")

                pred = np.array(pred)
                print("\n\nProbabilities Predicted : ")

                for o in range(len(classes)):
                    print(f"{classes[o]} : {pred[o]}")

                if pred.any() > threshold:
                    name = classes[np.argmax(pred)]
                    x, y, w, h = j
                    pred = np.array(pred, dtype=np.float64)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"{name} - {np.round(np.max(pred), 4)}", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

                else:
                    count = 0
                    cv2.putText(frame, classes[-1], (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

            # captured = name
            # if name == captured:
            #     count += 1

            if (count < 0):
                print(f"{name}'s face detected Successfully!!")
                break

    else:
        count = 0
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()
