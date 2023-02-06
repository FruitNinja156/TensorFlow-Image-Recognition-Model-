import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_image, train_labels),(test_images,test_labels) =data.load_data()

train_image=train_image/255.0
test_images=test_images/255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress","Coat","Sandal",
             "Shirt","Sneaker","Bag","Ankle boot"]

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_image, train_labels, epochs=5)

test_loss, test_acc=model.evaluate(test_images, test_labels)
print("Accuracy: ", test_acc)
predictions =model.predict(test_images)
print(class_names[np.argmax(predictions[0])])

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Predictions: " + class_names[np.argmax(predictions[i])])
    plt.show()