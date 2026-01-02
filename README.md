# Skin Lesion Classification using ResNet

This project is about classifying **skin lesion images** using deep learning. The system takes an image of a skin lesion and predicts the type of lesion based on its visual appearance.

## Dataset Used

I used the **HAM10000 (Skin Cancer MNIST)** dataset from Kaggle.

Dataset link:  
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

This dataset contains more than **10,000 dermatoscopic images** of different skin lesions.

### Classes in the Dataset
- akiec
- bcc
- bkl
- df
- mel
- nv
- vasc

## Model Used

I used a **ResNet (Residual Network)** model for this project.

- The model was pretrained on ImageNet
- The final layer was modified to classify 7 skin lesion classes
- Training was done using GPU on Kaggle

## Tools and Libraries Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib

## How the Project Works

1. Load the HAM10000 dataset and metadata
2. Prepare training, validation, and test splits
3. Apply basic image preprocessing and augmentation
4. Train the ResNet model
5. Select the best model using validation accuracy
6. Test the model on unseen images
7. Measure overall and per-class accuracy

## Results

- **Best Validation Accuracy:** 78.41%
- **Test Accuracy:** 77.98%

The results show that the model performs well on unseen skin lesion images.

## Files in This Repository

- `skin_lesion_ResNet.ipynb`  
  Contains data loading, model training, and evaluation code

- `README.md`  
  Project explanation

## Purpose of This Project

This project was created for learning and practice.

It helps in understanding:
- Image classification
- Deep learning with ResNet
- Medical image datasets
- Model training and evaluation

## Conclusion

This project demonstrates how deep learning models like ResNet can be used to classify skin lesion images.
With proper training and evaluation, the model is able to achieve good accuracy on real-world medical data.

Thank you for reading.

