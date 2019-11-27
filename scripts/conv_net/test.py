import pickle
import tensorflow as tf
import cv2
import numpy as np
import slidingwindow as sw

#Load the path for testing image
path = "../syn_generate/plain/banshee_30.jpg"

# Unpickle the encoder
file = open('encoder.pkl', 'rb')
encoder = pickle.load(file)
file.close()

# Load the model
model = tf.keras.models.load_model('model.h5')

# Read the image
tx = []
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Transform it
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

# Generate the windows
windows = sw.generate(new_img, sw.DimOrder.HeightWidthChannel, 256, 0.5)
for j in range(0, len(windows)):
    tx.append(new_img[windows[j].indices()])
tx = np.array(tx, dtype='float32')
tx /= 255
(s, h, w) = np.shape(tx)
tx = np.reshape(tx, (s, h, w, 1))

# Predict labels for all windows
ty = model.predict(tx)

# Compute the final label from the predictions
final_sf = np.zeros(dtype='float32', shape=(10))
for ti in range(0, len(ty)):
    for tj in range(0, 10):
        final_sf[tj] += ty[ti][tj]
final_sf /= len(ty)
final_sf = final_sf.reshape((-1, 10))

# Return the final label
print(encoder.inverse_transform(final_sf)[0][0])