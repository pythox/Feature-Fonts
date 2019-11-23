# Python script to create the model
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.data import Dataset
from tensorflow import one_hot
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint

def create_model():
    model = Sequential()
    # Layer 1
    model.add(Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    # Layer 2
    model.add(Conv2D(filters=64, kernel_size=3, padding='SAME', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    # Layer 3
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    # Layer 4
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    # Output Layer
    model.add(Dense(10, activation='softmax'))
    # Compile the model
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    return model


if __name__=='__main__':
    # Load the data
    ((train_x, train_y), (test_x, test_y)) = mnist.load_data()

    # Convert to one-hot encoded
    train_y, test_y = one_hot(indices=train_y, depth=10), one_hot(indices=test_y, depth=10)
    train_x, test_x = train_x.reshape(train_x.shape[0], 28, 28, 1), test_x.reshape(test_x.shape[0], 28, 28, 1)

    data = Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size=32)
    val_data = Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size=32)

    # Create checkpoint callback
    checkpoint_path = "model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

    # Fit the model
    model = create_model()
    model.fit(data, epochs=60, steps_per_epoch=30, validation_data=val_data, validation_steps=3, callbacks=[cp_callback])
