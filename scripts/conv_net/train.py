import os
import pickle
import numpy as np
import cv2
import slidingwindow as sw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import losses

# Globals
batch_size = 8
epoch = 10

# Creating data generators feeding 8 images at a time
def train_generator(batch_number_train, train_x, train_y):
    start_index = (batch_number_train-1)*batch_size
    end_index = batch_number_train*batch_size-1
    x, y = [], []
    for i in range(start_index, end_index + 1):
        img = cv2.imread(train_x[i], cv2.IMREAD_GRAYSCALE)
        (h, w) = np.shape(img)
        if h < 256:
            ratio = 256 / h
            h = 256
            w = int(w * ratio)
        else:
            ratio = h / 256
            h = 256
            w = int(w / ratio)
        new_img = cv2.resize(img, (w, h))
        windows = sw.generate(new_img, sw.DimOrder.HeightWidthChannel, 256, 0.5)
        for j in range(0, len(windows)):
            x.append(new_img[windows[j].indices()])
            y.append(train_y[i])
    x = np.array(x, dtype='float32')
    x /= 255
    (s,h,w) = np.shape(x)
    x = np.reshape(x, (s, h, w, 1))
    y = np.array(encoder.transform(y).toarray())
    return x, y

# Evalutating all the metrics using testing set
def validate(model, x, y):
    y_true, y_pred = [], []
    for i in range(0, len(x)):
        tx = []
        img = cv2.imread(x[i], cv2.IMREAD_GRAYSCALE)
        (h, w) = np.shape(img)
        if h < 256:
            ratio = 256 / h
            h = 256
            w = int(w * ratio)
        else:
            ratio = h / 256
            h = 256
            w = int(w / ratio)
        new_img = cv2.resize(img, (w, h))
        windows = sw.generate(new_img, sw.DimOrder.HeightWidthChannel, 256, 0.5)
        for j in range(0, len(windows)):
            tx.append(new_img[windows[j].indices()])
        tx = np.array(tx, dtype='float32')
        tx /= 255
        (s, h, w) = np.shape(tx)
        tx = np.reshape(tx, (s, h, w, 1))
        ty = model.predict(tx)
        final_sf = np.zeros(dtype='float32', shape=(10))
        for ti in range(0, len(ty)):
            for tj in range(0, 10):
                final_sf[tj] += ty[ti][tj]
        final_sf /= len(ty)
        final_sf = final_sf.reshape((-1, 10))
        y_pred.append(encoder.inverse_transform(final_sf)[0])
        y_true.append(y[i])
    print("ACCURACY", accuracy_score(y_true, y_pred))


if __name__ == '__main__':
    # Loading image destinations with their labels
    path = '../syn_generate/plain/'
    files = [f for f in os.listdir(path)]
    x, y = [], []
    for i in range(0, len(files)):
        file_name = os.path.splitext(os.path.basename(files[i]))[0]
        y.append(file_name.split('_')[0])
        x.append(os.path.join(path, files[i]))
    y = np.reshape(y, (-1, 1))

    # Fit the OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(y)

    # Split into train and test
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=44)
    del x, y

    # Pickle the One-Hot-Encoder
    file = open("encoder.pkl", "wb")
    pickle.dump(encoder, file)
    file.close()
    
    # Create the model
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(256, 256, 1)))
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='valid', data_format='channels_last'))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='valid', data_format='channels_last'))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='valid', data_format='channels_last'))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='valid', data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # Train the model
    for epoch_no in range(0, 5):
        for batch in range(1, len(train_x)//batch_size):
            if batch%10==0:
                print("Epoch : ", epoch_no, " Batch : ", batch)
            x, y = train_generator(batch, train_x, train_y)
            model.fit(x, y, verbose=0)
        validate(model, test_x, test_y)

    # Save the model
    model.save('model.h5')

    # Calculate the confusion matrix
    validate(model, test_x, test_y)