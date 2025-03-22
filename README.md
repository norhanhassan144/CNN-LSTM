# Sign Language MNIST Classification

This project is a deep learning model built using **TensorFlow & Keras** to classify American Sign Language (ASL) letters from the **Sign Language MNIST** dataset.

## 📌 Dataset
The dataset is sourced from **Kaggle**: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

- **Train set:** 27,455 images
- **Test set:** 7,172 images
- Each image is **28x28 grayscale** representing hand signs for letters **A-Y** (excluding J & Z).

## 🚀 Installation & Setup
### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/sign-language-mnist.git
cd sign-language-mnist
```

### 2️⃣ Install required dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download the dataset
Upload your `kaggle.json` file and run:
```python
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d datamunge/sign-language-mnist
!unzip sign-language-mnist.zip -d sign_language_data
```

## 🏗 Model Architecture
The model is built using **Conv1D & LSTM layers**:
```python
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
    LSTM(64, return_sequences=False, dropout=0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(25, activation='softmax')  
])
```

## 🏋️ Training the Model
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping])
```

## 🎯 Model Evaluation
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
```

## 🎨 Visualizing Predictions
```python
import random
idx = random.randint(0, len(x_test) - 1)
sample = x_test[idx].reshape(1, 28, 28)

prediction = model.predict(sample)
predicted_label = np.argmax(prediction)

plt.imshow(x_test[idx], cmap="gray")
plt.title(f"Predicted: {labels[predicted_label]}")
plt.axis("off")
plt.show()
```

## 🔗 References
- Kaggle Dataset: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- TensorFlow Documentation: [tensorflow.org](https://www.tensorflow.org/)

