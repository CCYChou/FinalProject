# Wafer Classification Using ResNet50

## Table of Contents
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Deployment](#deployment)

## Business Understanding

### Project Overview
This project implements a deep learning solution for wafer defect classification in semiconductor manufacturing. Using ResNet50 architecture, the system aims to automatically classify wafer defects from image data, helping to improve quality control in semiconductor production.

### Business Objectives
- Automate the wafer defect detection process
- Reduce manual inspection time and human error
- Improve quality control accuracy in semiconductor manufacturing
- Enable real-time defect classification

### Success Criteria
- High classification accuracy on test data
- Reliable model performance across different defect types
- Efficient processing time for real-time applications

## Data Understanding

### Data Sources
The project uses the following data files:
- `x_train_org_20210614.pickle`: Training feature data
- `y_train_org_20210614.pickle`: Training labels
- `x_test_20210614.pickle`: Test feature data
- `y_test_20210614.pickle`: Test labels

### Data Description
- Input data consists of wafer images
- Images are preprocessed and stored as numpy arrays
- Single-channel (grayscale) images
- Multiple defect classes for classification

## Data Preparation

### Data Preprocessing
1. Data Loading
   - Loading pickle files using Python's pickle module
   - Converting data to PyTorch tensors

2. Data Transformation
   - Adding channel dimension for CNN compatibility
   - Converting labels using LabelEncoder
   - Normalizing input data

3. Dataset Creation
   - Custom WaferDataset class implementing PyTorch's Dataset
   - Data loading with DataLoader for batch processing
   - Train-test split maintained

## Modeling

### Model Architecture
- Base model: ResNet50
- Modifications:
  - Adapted first convolution layer for single-channel input
  - Modified final fully connected layer for classification
  - Pre-trained weights utilized for transfer learning

### Training Configuration
```python
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Number of epochs: 50
- Device: CUDA (GPU) if available
```

### Training Process
- Implements early stopping based on validation accuracy
- Saves best model weights during training
- Tracks training loss and validation accuracy
- Uses tqdm for progress monitoring

## Evaluation

### Performance Metrics
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Classification report including:
  - Precision
  - Recall
  - F1-score

### Visualization
- Training loss curve
- Validation accuracy curve
- Confusion matrix heatmap
- Training history saved as 'training_history.png'

## Deployment

### Model Saving
- Best model weights saved as 'best_wafer_model.pth'
- Label encoder mapping preserved for inference

### Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm

### Usage
1. Place data files in the './data/' directory
2. Run the main script:
```bash
python wafer_classification.py
```

### Output
- Trained model weights
- Performance visualizations
- Detailed classification reports
- Training history plots

## Future Improvements
- Implement data augmentation for better generalization
- Experiment with different architectures
- Add real-time inference capabilities
- Implement model interpretability features
