# Python script to load and server the model

from flask import Flask, render_template, url_for, request, redirect
import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow import get_default_graph

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

graph = get_default_graph()
model = create_model()
model.load_weights("./scripts/model/cp.ckpt")

app = Flask(__name__)

# Index route
@app.route("/")
def index():
    return render_template('index.html', variable="-1")

# Error route
@app.route("/error")
def error():
    return render_template('error.html')

# Route for uploading the files
@app.route("/upload", methods=['POST', 'GET'])
def upload():
    if request.method == "POST" and "input" in request.files:
        imgstr = request.files['input'].read()
        npimage = np.fromstring(imgstr, np.uint8)
        image = cv2.imdecode(npimage, cv2.IMREAD_GRAYSCALE)
        image = image.reshape(1, 28, 28, 1)
        with graph.as_default():
            answer = model.predict(image)
        ans = np.array_str(np.argmax(answer, axis=1))
    return render_template('index.html', variable=ans)

# Route to remove favicon.ico error
@app.route("/favicon.ico")
def favicon():
    return "GG"


# Set debug mode=on
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
