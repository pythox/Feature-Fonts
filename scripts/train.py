import os
import numpy as np
import cv2
import random
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense, BatchNormalization, MaxPooling2D

# Hyperparmeters
width = 224
height = 32
channels = 1
learning_rate = 0.01
epochs = 1
training_samples = 45000
validation_samples = 5000
batch_size = 45
val_batch_size = 50
n_val_batches = 100
n_batches = 1000

# CODE
path = "data/PB/"
data = [f for f in os.listdir(path)]
random.shuffle(data)

# Prepare data
train, validate = data[0:45000], data[45000:50000]
test_x, test_y = [], []
for file in validate:
    file_name = os.path.splitext(os.path.basename(file))[0]
    test_y.append(file_name.split('_')[0])
    image_array = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
    image_array = 255 - image_array
    image_array = np.reshape(image_array, (32, 224, 1))
    test_x.append(image_array)

one_hot_encoder = OneHotEncoder(sparse=False)
test_y = np.reshape(test_y, (len(test_y), 1))
test_y = one_hot_encoder.fit_transform(test_y)

# Build Model
model = Sequential()
# Layer 1
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(32, 224, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
# Layer 2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
# Output Layer
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

# Training the model with dynamic loading
for epoch in range(0,epochs):
    print("=============== Epoch No. ", epoch, "===============")
    for batch_no in range(0,n_batches):
        print("Batch no : ", batch_no)
        start = batch_size * batch_no
        end = batch_size * (batch_no + 1)
        batch_x, batch_y = [], []
        for i in range(start, end):
            file_name = os.path.splitext(os.path.basename(train[i]))[0]
            batch_y.append(file_name.split('_')[0])
            image_array = cv2.imread(os.path.join(path, train[i]), cv2.IMREAD_GRAYSCALE)
            image_array = 255 - image_array
            image_array = np.reshape(image_array, (32, 224, 1))
            batch_x.append(image_array)
        batch_y = np.reshape(batch_y, (len(batch_y), 1))
        batch_y = one_hot_encoder.transform(batch_y)
        model.train_on_batch(np.array(batch_x), batch_y)
        # Validation every 10 batches
        if batch_no % 10 == 0:    
            res = model.test_on_batch(np.array(test_x[0:100]), test_y[0:100])
            print(model.metrics_names)
            print(res)
