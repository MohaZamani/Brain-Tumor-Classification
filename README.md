# MRI-Based Brain Tumor Classification

This repository contains code and documentation for detecting and classifying brain tumors using deep learning techniques, specifically a Convolutional Neural Network (CNN) with transfer learning based on the EfficientNetB0 architecture. The aim of this project is to aid in the early detection of brain tumors using MRI images, which can significantly improve patient outcomes.

## Project Overview

Brain tumors are abnormal growths of cells within the brain, which can either be malignant (cancerous) or benign (non-cancerous). Traditional methods of diagnosis, such as biopsy, require invasive surgery. This project explores a non-invasive method of detection through the use of MRI (Magnetic Resonance Imaging) scans, utilizing a deep learning model to classify the type of brain tumor present. The model is trained to identify glioma, meningioma, pituitary tumors, and healthy brains.

## Dataset

The dataset used in this project contains 3264 MRI images categorized into four classes:
- **Glioma:** 926 images
- **Meningioma:** 937 images
- **Pituitary Tumor:** 901 images
- **Healthy Brains:** 500 images

The MRI images are taken from various views (sagittal, axial, and coronal) and are preprocessed for the model. Augmentation techniques were also applied to increase the dataset size and improve model generalization.

## Methodology

The project is structured in the following stages:

1. **Data Preprocessing:** 
   - The MRI images were resized to 224x224 pixels to fit the input dimensions required by EfficientNet.
   - Data augmentation techniques such as rotation, zooming, and flipping were applied.

2. **Model Architecture:** 
   - A **CNN** model based on **EfficientNetB0** was used for feature extraction.
   - The EfficientNet model was pre-trained on the **ImageNet** dataset, and transfer learning was applied by fine-tuning the final layers for brain tumor classification.
   
3. **Transfer Learning:**
   - The EfficientNetB0 model's pre-trained weights were utilized for transfer learning.
   - The final layer of the model was replaced with a fully connected layer with a softmax activation function to classify the images into four categories.

4. **Training and Validation:**
   - The model was trained on 80% of the dataset, with 20% reserved for validation.
   - A batch size of 32 and Adam optimizer with a learning rate of 0.001 were used.
   - The model was trained for 12 epochs, achieving a training accuracy of 97%.

5. **Evaluation Metrics:**
   - **Accuracy, Precision, Recall, and F1-Score** were used to evaluate model performance.
   - The model achieved a test accuracy of 97% on unseen data.

## Results

- The model exhibited rapid convergence, with both training and validation accuracies reaching above 95% within a few epochs.
- The confusion matrix shows strong performance in all tumor classes, with minimal misclassifications.
- Detailed metrics:
  - **Accuracy:** 97%
  - **Precision:** 96%
  - **Recall (Sensitivity):** 97%
  - **F1-Score:** 96%

### Confusion Matrix

| Tumor Type      | True Positive | False Positive | False Negative | True Negative |
|-----------------|---------------|----------------|----------------|---------------|
| Glioma          | 87            | 3              | 5              | 92            |
| Meningioma      | 92            | 5              | 4              | 90            |
| Pituitary Tumor | 87            | 1              | 1              | 89            |
| No Tumor        | 51            | 1              | 1              | 52            |

## Conclusion

The EfficientNetB0 model, combined with transfer learning, has proven to be highly effective in classifying brain tumors using MRI images, achieving a remarkable accuracy of 97%. This model has the potential to assist in the early detection of brain tumors, offering a non-invasive and efficient diagnostic tool for medical professionals.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
2. Install the dependencies by running:
   ```bash
   pip install -r requirements.txt
3. Run the model:
  Open the provided Jupyter notebook main.ipynb and run all cells to preprocess the data, train the model, and evaluate the results.   
## Dependencies

The following Python libraries are required to run this project:

- Python 3.9
- TensorFlow
- Keras
- EfficientNet
- NumPy
- Pandas
- Matplotlib
- OpenCV

You can install these using the `requirements.txt` file or manually as shown in the [Install Dependencies](#install-dependencies) section.

---

## Future Work

- **Architectural Comparisons:** Explore other deep learning architectures like ResNet, VGG16, or custom CNN architectures to compare their performance with EfficientNetB0.
- **Hyperparameter Tuning:** Experiment with different hyperparameters such as learning rate, batch size, and dropout rates to optimize the model further.
- **Generalization:** Test the model on larger and more diverse MRI datasets to assess its generalizability and robustness.
- **Clinical Integration:** Investigate the potential of integrating this model into clinical decision support systems for real-time tumor detection.

---

## Acknowledgements

I would like to thank:
- The University of Tehran, Department of Computer Science, for providing the academic platform to carry out this research.
- The developers of the **EfficientNet** model for open-sourcing their implementation.
- All contributors to the open-source libraries and resources that made this project possible.

---

## References

1. Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking model scaling for convolutional neural networks." arXiv preprint arXiv:1905.11946 (2020).
2. Purnama, Muhammad Aji, et al. "Classification of Brain Tumors on MRI Images Using Convolutional Neural Network Model EfficientNet." Vol. 6, No. 4 (2022).
3. Tiwari, Pallavi, et al. "CNN Based Multiclass Brain Tumor Detection Using Medical Imaging." June 2022.
4. Saeedi, Soheila, et al. "MRI-based brain tumor detection using convolutional deep learning methods and machine learning techniques." January 2023.
