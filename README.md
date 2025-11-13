# DeepLearning-for-MicroPlastic-Classification

A comprehensive comparative study on classifying microplastic particles using Classic Machine Learning models, Convolutional Neural Networks (CNNs), and Vision Transformers (ViTs). This project provides a full pipeline from data extraction and preprocessing to model training, evaluation, and interpretation.
Dataset can be found at: https://figshare.com/articles/dataset/DeepParticle_dataset_MICRO_MESO_MACRO_2022_/26511253

## Key Features

*   **Comparative Analysis**: Implements and compares a range of models:
    *   **Classical ML**: Random Forest and Support Vector Machines (SVM).
    *   **Convolutional Neural Networks (CNNs)**: VGG16 and ResNet50.
    *   **Vision Transformers**: ViT-B/16.
*   **Data Handling**: Includes scripts to automatically download the dataset, preprocess raw images with their annotations, and create stratified train/validation/test splits.
*   **Class Imbalance**: Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data for the ResNet50 and ViT models, improving performance on minority classes.
*   **Transfer Learning**: Leverages pre-trained models on ImageNet and fine-tunes them for the specific task of microplastic classification.
*   **Model Interpretability**: Provides tools for visualizing model decisions:
    *   **Grad-CAM** for the VGG16 and resnet-50 models.
    *   **Token-based attention visualization** (Grad-CAM-like) for the Vision Transformer.
*   **Reproducibility**: Scripts are configured with fixed random seeds (np.random.seed(42) and torch.manual_seed(42)) for reproducible results.

## Repository Structure

This repository is organized into several key scripts for a complete end-to-end workflow:

*   `data-extraction.py`: Downloads the microplastic dataset from Google Drive.
*   `data-preprocessing.py`: Processes raw images and `.tsv` annotations to crop individual particles and splits them into `train`, `val`, and `test` directories.
*   `ml-baselines.py`: Trains and evaluates classical Random Forest and SVM models using handcrafted features (intensity, texture, shape).
*   `ml-IOU.py`: Computes the Intersection over Union (IoU) metric for the trained classical models.
*   `vgg16.py`: Implements the training and evaluation pipeline for a fine-tuned VGG16 model, generates reports and loss curves.
*   `resnet50.py`: Implements the training pipeline for a ResNet50 model, incorporating SMOTE to handle class imbalance, generates detailed reports and loss curves.
*   `vits.py`: A complete script for training a Vision Transformer (ViT) with SMOTE, generating detailed evaluation reports, loss curves, and Grad-CAM-like visualizations.
*   `prediction.py`: Visualizes a grid of predictions from the trained VGG16 and Resnet models on test images.
*   `grad-cam.py`: Generates Grad-CAM heatmaps for the VGG16 and ResNet50 model to highlight important image regions.

## Getting Started

### 1. Clone the Repository
```sh
git clone https://github.com/nikkiray309/DeepLearning-for-MicroPlastic-Classification.git
cd DeepLearning-for-MicroPlastic-Classification
```

### 2. Install Dependencies
Install the required Python libraries.
```sh
pip install torch torchvision gdown pandas numpy scikit-learn scikit-image opencv-python matplotlib seaborn tqdm imbalanced-learn albumentations
```

### 3. Download the Dataset
Run the data extraction script. This will download the dataset into a directory named `MICRO`.
```sh
python data-extraction.py
```

### 4. Preprocess the Data
Run the preprocessing script to crop particles from the raw images and create the necessary `train/val/test` splits. The processed data will be saved in `processed_MICRO`.
```sh
python data-preprocessing.py
```

### 5. Run the Models
You can now train and evaluate the different models. The scripts will save trained model weights (`.pth`, `.joblib`) and evaluation artifacts (reports, plots).

*   **Classical Models (RF & SVM):**
    ```sh
    python ml-baselines.py
    ```

*   **VGG16:**
    ```sh
    python vgg16.py
    ```

*   **ResNet50 (with SMOTE):**
    ```sh
    python resnet50.py
    ```

*   **Vision Transformer (with SMOTE):**
    ```sh
    python vits.py
    ```

### 6. Visualize Results
After training, you can run the visualization scripts.

*   **VGG16/Resnet50 Predictions:**
    ```sh
    python prediction.py
    ```

*   **VGG16/Resnet50 Grad-CAM:**
    ```sh
    python grad-cam.py
    ```
The `vits.py` script automatically generates its own evaluation plots and visualizations in the `vit_smote_results/` directory upon completion.

## Results
The models achieve strong performance, with the Vision Transformer showing excellent results. The final classification report for the best-performing model on the test set is as follows:

```
              precision    recall  f1-score   support

        foam     0.8571    0.9000    0.9480        20
        hard     0.9922    0.9549    0.9732       532
        line     0.9126    0.9895    0.9495        95
       noise     0.9375    0.9783    0.9574        46
      pellet     0.8116    0.9333    0.9482        60
   reference     1.0000    1.0000    1.0000         8

    accuracy                         0.9880       761
   macro avg     0.9785    0.9793    0.9777       761
weighted avg     0.9912    0.9980    0.9888       761
