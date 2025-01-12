
# Shoe Image Classification using CNN and Generating Synthetic Samples using DCGan


This project involves building a Convolutional Neural Network (CNN) model to classify shoe images. The dataset used for this project is the **Large Shoe Dataset UT Zappos50K**, available on Kaggle: [Dataset Link](https://www.kaggle.com/datasets/aryashah2k/large-shoe-dataset-ut-zappos50k). The project uses squared images of shoe samples for training and testing. Additionally, synthetic data generation was explored using Deep Convolutional Generative Adversarial Networks (DCGANs) to augment the dataset.

---

## Dataset
- **Source**: [Kaggle - Large Shoe Dataset UT Zappos50K](https://www.kaggle.com/datasets/aryashah2k/large-shoe-dataset-ut-zappos50k)
- **Dataset Details**:
  - Contains a variety of shoe images categorized by type.
  - Squared images were extracted and used for this project.

---

## Prerequisites
To run this project, ensure you have the following installed:
- Python 3.7 or above
- TensorFlow (with GPU support if available)
- Keras
- NumPy
- Pandas
- Scikit-learn

For Mac users intending to use TensorFlow with GPU support:
- Install the `metal` library to enable GPU acceleration.

---

## Project Workflow
1. **Data Cleaning**:
   - Rows with less than 100 labels were removed for better representation.
2. **Train-Test Split**:
   - Data was split into training and testing sets using the `train_test_split` function from Scikit-learn.
3. **Data Augmentation**:
   - Image augmentation techniques were applied to increase the variability in training data, enhancing model generalization.
4. **Synthetic Data Generation**:
   - A Deep Convolutional Generative Adversarial Network (DCGAN) was implemented to generate synthetic shoe images.
   - The DCGAN model includes:
     - A generator network to create realistic images.
     - A discriminator network to evaluate the authenticity of the generated images.
   - The synthetic dataset was used to enhance the training dataset and improve model performance.

---

## Model Architecture
### CNN
The CNN model includes the following layers:
- Convolutional Layers
- MaxPooling Layers
- Dropout Layers
- Dense (Fully Connected) Layers
- Batch Normalization

### DCGAN
The DCGAN consists of:
- **Generator**: Uses transposed convolution layers to generate synthetic images from noise vectors.
- **Discriminator**: A CNN-based model to classify images as real or generated.

Callbacks such as EarlyStopping and Learning Rate Scheduler were used to optimize training.

---

## Running the Project
1. Clone the repository and navigate to the project directory.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt


