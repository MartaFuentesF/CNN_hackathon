# Technical Report: [Hackathon with Holly Heath](https://generalassemb.ly/instructors/holly-heath/22063)

## Summary

During the Hackathon, we developed multiple Convolutional Neural Networks (CNNs) to classify images from the Dig-MNIST dataset. This report details the architecture, training process, and performance of six CNN models. Our objective was to identify the best model for classifying the images based on accuracy and loss metrics.

## Dataset

The dataset used is the Dig-MNIST dataset, which consists of 10,240 images, each of 28x28 pixels, representing handwritten digits. The data was preprocessed and split into training and testing sets with a ratio of 75:25.

## Preprocessing Steps

1. **Loading Data**: Data was loaded using Pandas.
2. **Splitting Data**: Data was split into training and testing sets using `train_test_split`.
3. **Scaling**: Pixel values were scaled to the range [0, 1].
4. **Reshaping**: Data was reshaped to 28x28x1 to fit the input shape required by CNNs.

## Model Architectures

### CNN1: Basic Model
- **Layers**: Input layer, 1 Conv layer (32 filters, 3x3), MaxPooling, 1 Conv layer (64 filters, 3x3), MaxPooling, Flatten, Dense layer (10 neurons, softmax)
- **Performance**:
  - Training Loss: 0.1265
  - Training Accuracy: 0.9609
  - Validation Loss: 0.1893
  - Validation Accuracy: 0.9402

### CNN2: Modified Filter Size
- **Layers**: Input layer, 1 Conv layer (32 filters, 2x2), MaxPooling, 1 Conv layer (128 filters, 2x2), MaxPooling, Flatten, Dense layer (10 neurons, softmax)
- **Performance**:
  - Training Loss: 0.1155
  - Training Accuracy: 0.9655
  - Validation Loss: 0.1801
  - Validation Accuracy: 0.9410

### CNN3: L2 Regularization
- **Layers**: Similar to CNN2 with L2 Regularization (0.01) added to Conv layers
- **Performance**:
  - Training Loss: 0.1283
  - Training Accuracy: 0.9900
  - Validation Loss: 0.2708
  - Validation Accuracy: 0.9395

### CNN4: Dropout Regularization
- **Layers**: Similar to CNN2 with Dropout (0.1) added after the first Conv layer
- **Performance**:
  - Training Loss: 1.3240
  - Training Accuracy: 1.0000
  - Validation Loss: 0.3481
  - Validation Accuracy: 0.9426

### CNN5: Additional Hidden Layer
- **Layers**: Input layer, 1 Conv layer (32 filters, 2x2), MaxPooling, 2 Conv layers (128 filters each, 2x2), MaxPooling after each Conv layer, Flatten, Dense layer (10 neurons, softmax)
- **Performance**:
  - Training Loss: 5.9969
  - Training Accuracy: 1.0000
  - Validation Loss: 0.3376
  - Validation Accuracy: 0.9473

### CNN6: Combined Regularization (L2 + Dropout)
- **Layers**: Similar to CNN2 with both L2 Regularization (0.01) and Dropout (0.1)
- **Performance**:
  - Training Loss: 1.4666
  - Training Accuracy: 0.6775
  - Validation Loss: 0.8486
  - Validation Accuracy: 0.8211

## Performance Summary

| Model | Hidden Layers | Neurons          | Conv Filter Size | Regularization            | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Performance Rank |
|-------|----------------|------------------|------------------|---------------------------|---------------|-------------------|-----------------|---------------------|-------------------|
| CNN1  | 1              | 32→128→10        | 3x3              | None                      | 0.1265        | 0.9609            | 0.1893          | 0.9402              | 4                 |
| CNN2  | 1              | 32→128→10        | 2x2              | None                      | 0.1155        | 0.9655            | 0.1801          | 0.9410              | 3                 |
| CNN3  | 1              | 32→128→10        | 2x2              | L2 (0.01)                 | 0.1283        | 0.9900            | 0.2708          | 0.9395              | 1 or 2            |
| CNN4  | 1              | 32→128→10        | 2x2              | Dropout (0.1)             | 1.3240        | 1.0000            | 0.3481          | 0.9426              | 1 or 2            |
| CNN5  | 2              | 32→128→128→10    | 2x2              | None                      | 5.9969        | 1.0000            | 0.3376          | 0.9473              | 5                 |
| CNN6  | 1              | 32→128→10        | 2x2              | L2 (0.01) + Dropout (0.1) | 1.4666        | 0.6775            | 0.8486          | 0.8211              | 6                 |

## Conclusion

The best performing model was CNN5, which had two hidden layers and achieved the highest validation accuracy of 0.9473, though it showed signs of overfitting. Future work could focus on refining the model architectures to reduce overfitting while maintaining high accuracy.

## Recommendations

- **Model Selection**: Based on the validation accuracy, CNN5 is recommended. However, to address overfitting, further tuning, such as adding dropout or experimenting with regularization, is suggested.
- **Future Work**: Exploring other architectures, such as deeper networks or different types of regularization, could improve model performance.

This report provides a comprehensive overview of the CNN models tested during the Hackathon, showcasing the iterative process of model development and evaluation.

### References
- GA lab 10.02 NN Classification
- GA lesson CNN
