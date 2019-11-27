# Python script to load and server the model
from flask import Flask, render_template, url_for, request, redirect
import pickle
import tensorflow as tf
import cv2
import numpy as np
import slidingwindow as sw

# Globals
app = Flask(__name__)
ret_val = "Upload Image"
path = './static/img/uploaded.jpg'

# Unpickle the encoder
file = open('./scripts/conv_net/encoder.pkl', 'rb')
encoder = pickle.load(file)
file.close()

# Load the model
model = tf.keras.models.load_model('./scripts/conv_net/model.h5')

def predict_input():
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
	return encoder.inverse_transform(final_sf)[0][0]

# Index route
@app.route("/")
def index():
    global ret_val
    return render_template('index.html', detected_font=ret_val)

# Error route
@app.route("/error")
def error():
    return render_template('error.html')

# Route for uploading the files
@app.route("/upload", methods=['POST', 'GET'])
def upload():
    global ret_val
    if request.method == "POST":
        request.files['avatar'].save('./static/img/uploaded.jpg')
        ret_val = predict_input()
        return ret_val.capitalize()
    return None

# Route to remove favicon.ico error
@app.route("/favicon.ico")
def favicon():
    return "GG"

# Set debug mode=on
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)