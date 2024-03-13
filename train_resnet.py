import pandas as pd
import numpy as np

data = pd.read_csv('/home/junaid/Downloads/AzamFER/fer2013/fer2013.csv')

# Splitting the data into train, validation, and test sets
train_data = data[data['Usage'] == 'Training']
val_data = data[data['Usage'] == 'PublicTest']
test_data = data[data['Usage'] == 'PrivateTest']

# Preprocess the data
def preprocess(data):
    images = data['pixels'].apply(lambda x: np.fromstring(x, sep=' ')).values
    images = np.vstack(images).reshape(-1, 48, 48, 1)
    images = images / 255.0
    labels = data['emotion'].values
    return images, labels

X_train, y_train = preprocess(train_data)
X_val, y_val = preprocess(val_data)
X_test, y_test = preprocess(test_data)


ep = 1
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten

input_tensor = Input(shape=(48, 48, 1))
input_tensor_rgb = tf.keras.layers.concatenate([input_tensor, input_tensor, input_tensor], axis=-1)

base_model = ResNet50(input_tensor=input_tensor_rgb, weights=None, include_top=False)
x = Flatten()(base_model.output)
x = Dense(7, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=x)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=ep, batch_size=32)

# Assuming the model is already trained
model_path = "resnet50_fer2013_" + str(ep) + ".h5"
model.save(model_path)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
