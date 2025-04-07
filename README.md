
# 🧠 CNN with MNIST Dataset

This project demonstrates the implementation of a **Convolutional Neural Network (CNN)** using **TensorFlow** and **Keras** to classify handwritten digits from the **MNIST dataset**. The notebook provides a clean and straightforward walkthrough, ideal for beginners and intermediate practitioners in deep learning.

---

## 📁 Repository Structure

```
├── CNN_with_MNIST_Dataset.ipynb  # Jupyter notebook implementing the CNN
├── README.md                     # Project overview and instructions
```

---

## 🧾 Overview

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9). This notebook guides users through:

- Loading and visualizing the dataset
- Preprocessing the data for training
- Building a CNN using the Keras Sequential API
- Training and evaluating the model
- Plotting training/validation accuracy and loss
- Predicting and displaying test samples

---

## 📦 Dependencies

To run the notebook, make sure you have the following installed:

```bash
pip install tensorflow numpy matplotlib
```

Or use a Jupyter environment like **Google Colab** or **Jupyter Notebook** with the above libraries.

---

## 📊 Dataset Details

- **Dataset**: MNIST (Modified National Institute of Standards and Technology)
- **Classes**: 10 digits (0 through 9)
- **Image Size**: 28x28 pixels (grayscale)
- **Training Samples**: 60,000
- **Test Samples**: 10,000

---

## 🏗️ Model Architecture

The CNN model follows a classic structure:

```
Input: 28x28x1 grayscale image

1. Conv2D (32 filters, 3x3 kernel, ReLU activation)
2. MaxPooling2D (2x2 pool size)
3. Conv2D (64 filters, 3x3 kernel, ReLU activation)
4. MaxPooling2D (2x2 pool size)
5. Flatten
6. Dense (128 units, ReLU)
7. Output: Dense (10 units, Softmax)
```

---

## ⚙️ Training Details

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: 128

---

## 📈 Results

The model achieves over **98% accuracy** on the MNIST test dataset.

Visualizations include:

- Accuracy and loss plots over epochs
- Sample predictions with actual and predicted labels

---

## ▶️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook CNN_with_MNIST_Dataset.ipynb
   ```

---



## 🙌 Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
# cnn-explanation
# cnn-practice
