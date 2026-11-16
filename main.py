from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Loading the Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
labeled_images = list(zip(x_train, y_train))
plt.subplots_adjust(hspace = .5)
for i, (image, label) in enumerate(labeled_images[:25]):
    plt.subplot(5,5,i+1)
    plt.axis('off')
    plt.imshow(image, cmap='gray', interpolation= 'nearest')
    plt.title(label)
plt.show()

# Training and Test Differences
n_train = len(y_train)
n_test = len(y_test)
digits = [0,1,2,3,4,5,6,7,8,9]
train_shares = []
test_shares = []
for num in digits:
    train_shares.append(sum(y_train == num) / n_train)
    test_shares.append(sum(y_test == num) / n_test)
position = np.arange(len(digits))
bar_width = 0.40
plt.bar(position - bar_width/2, train_shares, label='Train', width = bar_width, color = 'blue')
plt.bar(position + bar_width/2, test_shares, label='Test', width = bar_width, color = 'red')
plt.legend()
plt.show()

# Normalization
# cast train and test to float 32 and scale from 0 to 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
mu, sigma = x_train.mean(), x_train.std()
x_train = ((x_train - mu) / sigma)[..., None]
x_test = ((x_test - mu) / sigma)[..., None]
print(x_train.shape, x_test.shape)

# Building a CNN with Keras
import keras
from keras import layers
cnn = keras.Sequential([
    keras.Input(shape = (28, 28 ,1)), # Shape of X_train and X_test
    layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPool2D(),
    layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(10, activation = 'softmax') # 10 Digits use Probabilities
])
cnn.compile(
    optimizer = 'Adam',
    loss = ['sparse_categorical_crossentropy'],
    metrics = ['accuracy']
)

history = cnn.fit(
    x_train, y_train,
    batch_size = 128,
    epochs = 5,
    validation_split = 0.15
)
test_loss, test_acc = cnn.evaluate(x_test, y_test)
print('Loss: ', test_loss)
print("Accuracy ", test_acc)