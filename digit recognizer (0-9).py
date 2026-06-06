import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

#Step 1: Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

#Step 2: Build the Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

#Step 3: Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Step 4: Train the Model
model.fit(x_train, y_train, epochs=10)
print("Model trained successfully!")

#Step 5: Drawing Canvas
canvas = np.zeros((300, 300), dtype=np.uint8)
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 10, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
  
cv2.namedWindow("Draw Digit")
cv2.setMouseCallback("Draw Digit", draw)

print("Draw digit -> Press 'p' to predict")

#Step 6: Loop
while True:
    cv2.imshow("Draw Digit", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        if np.count_nonzero(canvas) == 0:
            print("Draw something!")
            continue
            
        # 1. CROP DIGIT
        coords = cv2.findNonZero(canvas)
        x, y, w, h = cv2.boundingRect(coords)
        digit = canvas[y:y+h, x:x+w]

        # 2. RESIZE WITH INTER_AREA (Prevents blurring when shrinking)
        if w > h:
            new_w = 20
            new_h = int((20 / w) * h)
        else:
            new_h = 20
            new_w = int((20 / h) * w)
            
        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 3. THRESHOLDING (Forces gray, blurry pixels to become solid white)
        _, digit = cv2.threshold(digit, 100, 255, cv2.THRESH_BINARY)

        # 4. PAD TO 28x28 AND CENTER
        top = (28 - new_h) // 2
        bottom = 28 - new_h - top
        left = (28 - new_w) // 2
        right = 28 - new_w - left
        
        digit = cv2.copyMakeBorder(digit, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        # 5. NORMALIZE
        digit = digit / 255.0

        # 6. RESHAPE FOR MODEL
        digit = digit.reshape(1, 28, 28)

        # 7. PREDICT
        prediction = np.argmax(model.predict(digit), axis=1)
        print("Predicted:", prediction[0])

    elif key == ord('c'):
        canvas[:] = 0

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
cv2.destroyAllWindows()
