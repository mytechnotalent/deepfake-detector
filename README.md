![image](https://github.com/mytechnotalent/deepfake-detector/blob/main/deepfake-detector.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Deepfake Detector
Deepfake Detector is an AI/ML model designed to detect AI-generated or manipulated images.

<br><br>

### Formulas

- **Input Layer**: Defines the shape of the input data, which in this case is an image with dimensions 128x128 pixels and 3 color channels (RGB).

```math
\text{Input Layer}: \text{output shape} = (128, 128, 3)
```

- **Rescaling Layer**: Normalizes the pixel values of the input images to a range suitable for neural networks, in this case dividing by 127.

```math
\text{Rescaling Layer}: \text{output} = \frac{\text{input}}{127}
```

- **Conv2D Layer**: Applies convolutional filters to extract features from the input image using ReLU activation, maintaining spatial dimensions through padding and adjusting spatial resolution via strides.

```math
\text{Conv2D Layer}: \text{output} = \text{ReLU}\left(\left(\frac{\text{input} + 2 \times \text{padding} - \text{kernel size}}{\text{strides}}\right) + 1\right)
```

- **BatchNormalization Layer**: Normalizes the activations of the previous layer, helping to stabilize and speed up training by reducing internal covariate shift.

```math
\text{BatchNormalization Layer}: \text{output} = \frac{\text{input} - \text{mean}}{\sqrt{\text{variance} + \text{epsilon}}} \times \text{scale} + \text{offset}
```

- **MaxPooling2D Layer**: Downsamples the input representation by taking the maximum value in a defined spatial neighborhood, reducing spatial dimensions and creating spatial invariance.

```math
\text{MaxPooling2D Layer}: \text{output size} = \left\lfloor \frac{\text{input size} - \text{pool size}}{\text{strides}} \right\rfloor + 1
```

- **Flatten Layer**: Flattens the multi-dimensional input into a 1D array, preparing it for fully connected layers like Dense layers.

```math
\text{Flatten Layer}: \text{output} = \text{input size} \times \text{last dimension size}
```

- **Dense Layer (n units)**: Fully connected layer with n neurons, applying the ReLU activation function to introduce non-linearity and learn complex representations.

```math
\text{Dense Layer (n units)}: \text{output} = \text{ReLU}(\text{input})
```

- **Dropout Layer (n)**: Randomly drops n% of the input units during training to prevent overfitting by promoting the learning of redundant representations.

```math
\text{Dropout Layer (n)}: \text{output} = \text{input} \times n
```

- **Dense Layer (1 unit)**: Final output layer with a sigmoid activation function, producing a probability indicating the likelihood that the input image belongs to the class (real or fake).

```math
\text{Dense Layer (1 unit)}: \text{output} = \text{Sigmoid}(\text{input})
```

### Evaluation Metrics for Deepfake Detector

#### 1. Loss

**Definition:**
Loss measures how well or poorly the model is performing by quantifying the difference between the predicted outputs and the actual labels. It is a key component of the optimization process in training a neural network. In a binary classification task like deepfake detection, common loss functions include Binary Crossentropy.

**Binary Crossentropy Loss Formula:**
```math
Loss = -\frac{1}{N} \sum [ y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i) ]
```
where:
- $N$ is the number of samples.
- $y_i$ is the actual label for the $i$-th sample (1 for real, 0 for fake).
- $p_i$ is the predicted probability of the $i$-th sample being real.

**Importance for Deepfake Detector:**
- **Model Optimization:** Loss functions guide the optimization algorithm (e.g., gradient descent) in adjusting the model's weights to minimize error during training.
- **Performance Indicator:** A decreasing loss during training generally indicates that the model is improving.
- **Overfitting/Underfitting Detection:** Monitoring the loss on training and validation sets helps detect overfitting (low training loss, high validation loss) or underfitting (high loss on both sets).

#### 2. Accuracy

**Definition:**
Accuracy measures the proportion of correctly classified samples out of the total samples. For binary classification, it's the number of true positives and true negatives divided by the total number of samples.

**Accuracy Formula:**
```math
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
```
where:
- $TP$ = True Positives (real images correctly identified as real)
- $TN$ = True Negatives (fake images correctly identified as fake)
- $FP$ = False Positives (fake images incorrectly identified as real)
- $FN$ = False Negatives (real images incorrectly identified as fake)

**Importance for Deepfake Detector:**
- **General Performance:** Accuracy provides a straightforward measure of overall performance.
- **Threshold Sensitivity:** While useful, accuracy alone can be misleading in imbalanced datasets where one class is more prevalent. For deepfake detection, if fake images are rare, a model could have high accuracy by always predicting real.

#### 3. Precision

**Definition:**
Precision measures the proportion of correctly identified positive samples out of all samples that were identified as positive. In the context of deepfake detection, it is the proportion of correctly identified real images out of all images identified as real.

**Precision Formula:**
```math
Precision = \frac{TP}{TP + FP}
```
where:
- $TP$ = True Positives
- $FP$ = False Positives

**Importance for Deepfake Detector:**
- **False Positive Rate:** High precision indicates a low false positive rate, which is crucial in applications where mistakenly identifying a fake image as real can have significant consequences.
- **Model Reliability:** Precision is important when the cost of false positives is high.

#### 4. Recall

**Definition:**
Recall measures the proportion of correctly identified positive samples out of all actual positive samples. In the context of deepfake detection, it is the proportion of correctly identified real images out of all actual real images.

**Recall Formula:**
```math
Recall = \frac{TP}{TP + FN}
```
where:
- $TP$ = True Positives
- $FN$ = False Negatives

**Importance for Deepfake Detector:**
- **False Negative Rate:** High recall indicates a low false negative rate, which is important in applications where failing to identify a real image as real is critical.
- **Sensitivity:** Recall is crucial when the cost of missing positive samples (real images) is high.

### Model Results
```
model 1
evaluation metrics: [0.4498537480831146, 0.7909215688705444, 0.7790842652320862, 0.8078699707984924]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(16, (4, 4), strides=1))
    model.add(LeakyReLU(alpha=0.01)) 
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(36, (2, 2), strides=1))
    model.add(LeakyReLU(alpha=0.01)) 
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=1))
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(layers.Dense(64))
    model.add(LeakyReLU(alpha=0.01))
    model.add(layers.Dense(32))
    model.add(LeakyReLU(alpha=0.01))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 2
evaluation metrics: [0.4258367717266083, 0.8213663697242737, 0.7792103290557861, 0.8932200074195862]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(16, (4, 4), strides=2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(32, (2, 2), strides=1, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=1))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 3
evaluation metrics: [0.4562738537788391, 0.81788170337677, 0.8589985370635986, 0.7574357986450195]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(16, (4, 4), strides=1))
    model.add(LeakyReLU(alpha=0.02)) 
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(32, (2, 2), strides=1))
    model.add(LeakyReLU(alpha=0.02)) 
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=1))
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(LeakyReLU(alpha=0.02))
    model.add(layers.Dense(64))
    model.add(LeakyReLU(alpha=0.02))
    model.add(layers.Dense(32))
    model.add(LeakyReLU(alpha=0.02))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 4
evaluation metrics: [0.5329386591911316, 0.8252177834510803, 0.849372386932373, 0.7875484824180603]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(16, (4, 4), strides=1))
    model.add(LeakyReLU(alpha=0.01))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(32, (2, 2), strides=1))
    model.add(LeakyReLU(alpha=0.01))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=1))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64))
    model.add(LeakyReLU(alpha=0.01))
    model.add(layers.Dense(32))
    model.add(LeakyReLU(alpha=0.01))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 5
evaluation metrics: [0.8608404397964478, 0.8547455072402954, 0.8935251832008362, 0.8030666708946228]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 6
evaluation metrics: [0.37503954768180847, 0.8781293034553528, 0.89375239610672, 0.8562719225883484]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5)) 
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 7
evaluation metrics: [0.8206597566604614, 0.8511691689491272, 0.8951209187507629, 0.7930907011032104]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 8
evaluation metrics: [0.9416314959526062, 0.8513525724411011, 0.9008456468582153, 0.7871789932250977]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 9 
evaluation metrics: [0.652288556098938, 0.7917469143867493, 0.9038560390472412, 0.6495473980903625]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 10
evaluation metrics: [0.6219281554222107, 0.8518111109733582, 0.8916856050491333, 0.7984482049942017]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 11
evaluation metrics: [0.6577958464622498, 0.8703346848487854, 0.9255160689353943, 0.8034361600875854]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling')
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 12
evaluation metrics: [0.4818786680698395, 0.8573131561279297, 0.8925300240516663, 0.8100868463516235]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu')) 
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 13
evaluation metrics: [0.5268889665603638, 0.8912425637245178, 0.9275743365287781, 0.8470349311828613]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3)) 
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 14
evaluation metrics: [0.3337911367416382, 0.883172869682312, 0.941728949546814, 0.8150748014450073]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.7)) 
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 15
evaluation metrics: [0.6241858005523682, 0.6010087132453918, 0.5552894473075867, 0.9852207899093628]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.7)) 
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.7)) 
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 16
evaluation metrics: [0.40905073285102844, 0.8599724769592285, 0.8156270384788513, 0.9275817275047302]
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.7)) 
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.7)) 
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

model 17
TBD 07-08-24
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))
    model.add(layers.Rescaling(1./127, name='rescaling'))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.9))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.9)) 
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
```

## Train
```python
import os
import zipfile
import urllib3
import requests
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.layers import LeakyReLU  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # type: ignore


class DatasetHandler:
    """
    A class to handle dataset downloading, unzipping, loading, and processing.
    """

    def __init__(self, dataset_url, dataset_download_dir, dataset_file, dataset_dir, train_dir, test_dir, val_dir):
        """
        Initialize the DatasetHandler with the specified parameters.

        Args:
            dataset_url (str): URL to download the dataset from.
            dataset_download_dir (str): Directory to download the dataset to.
            dataset_file (str): Name of the dataset file.
            dataset_dir (str): Directory containing the dataset.
            train_dir (str): Directory containing the training data.
            test_dir (str): Directory containing the test data.
            val_dir (str): Directory containing the validation data.
        """
        self.dataset_url = dataset_url
        self.dataset_download_dir = dataset_download_dir
        self.dataset_file = dataset_file
        self.dataset_dir = dataset_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir

    def download_dataset(self):
        """
        Download the dataset from the specified URL.
        
        Returns:
            bool: True if the dataset was successfully downloaded, False otherwise.
        """
        if not os.path.exists(self.dataset_download_dir):
            os.makedirs(self.dataset_download_dir)
        file_path = os.path.join(self.dataset_download_dir, self.dataset_file)
        if os.path.exists(file_path):
            print(f'dataset file {self.dataset_file} already exists at {file_path}')
            return True
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(self.dataset_url, stream=True, verify=False)
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as file, tqdm(desc=self.dataset_file, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print(f'dataset downloaded and saved to {file_path}')
        return True

    def unzip_dataset(self):
        """
        Unzip the downloaded dataset file.
        
        Returns:
            bool: True if the dataset was successfully unzipped, False otherwise.
        """
        file_path = os.path.join(self.dataset_download_dir, self.dataset_file)
        if os.path.exists(self.dataset_dir):
            print(f'dataset is already downloaded and extracted at {self.dataset_dir}')
            return True
        if not os.path.exists(file_path):
            print(f'dataset file {file_path} not found after download')
            return False
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_download_dir)
        print(f'dataset extracted to {self.dataset_dir}')
        return True

    def get_image_dataset_from_directory(self, dir_name):
        """
        Load image dataset from the specified directory.

        Args:
            dir_name (str): Name of the directory containing the dataset.

        Returns:
            tf.data.Dataset: Loaded image dataset.
        """
        dir_path = os.path.join(self.dataset_dir, dir_name)
        return tf.keras.utils.image_dataset_from_directory(
            dir_path,
            labels='inferred',
            color_mode='rgb',
            seed=42,
            batch_size=64,
            image_size=(128, 128)
        )

    def load_split_data(self):
        """
        Load and split the dataset into training, validation, and test datasets.

        Returns:
            tuple: Training, validation, and test datasets.
        """
        train_data = self.get_image_dataset_from_directory(self.train_dir)
        test_data = self.get_image_dataset_from_directory(self.test_dir)
        val_data = self.get_image_dataset_from_directory(self.val_dir)
        return train_data, test_data, val_data


class DeepfakeDetectorModel:
    """
    A class to create and train a deepfake detection model.
    """

    def __init__(self):
        """
        Initialize the DeepfakeDetectorModel by building the model.
        """
        self.model = self._build_model()

    def _build_model(self):
        """
        Build the deepfake detection model architecture.

        Returns:
            tf.keras.Model: Built model.
        """
        model = models.Sequential()
        model.add(layers.Input(shape=(128, 128, 3)))
        model.add(layers.Rescaling(1./127, name='rescaling'))
        model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def compile_model(self, learning_rate):
        """
        Compile the deepfake detection model.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy', 
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )

    def train_model(self, train_data, val_data, epochs):
        """
        Train the deepfake detection model.

        Args:
            train_data (tf.data.Dataset): Training dataset.
            val_data (tf.data.Dataset): Validation dataset.
            epochs (int): Number of epochs to train the model.

        Returns:
            tf.keras.callbacks.History: History object containing training details.
        """
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)
        model_checkpoint_callback = ModelCheckpoint('deepfake_detector_model_best.keras', monitor='val_loss', save_best_only=True, verbose=1)
        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[early_stopping_callback, reduce_lr_callback, model_checkpoint_callback]
        )

    def evaluate_model(self, test_data):
        """
        Evaluate the deepfake detection model.

        Args:
            test_data (tf.data.Dataset): Test dataset.

        Returns:
            list: Evaluation metrics.
        """
        return self.model.evaluate(test_data)

    def save_model(self, path):
        """
        Save the deepfake detection model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        self.model.save(path)


class TrainModel:
    """
    A class to manage training of a deepfake detection model.
    """

    def __init__(self, dataset_url, dataset_download_dir, dataset_file, dataset_dir, train_dir, test_dir, val_dir):
        """
        Initialize the TrainModel class with the specified parameters.

        Args:
            dataset_url (str): URL to download the dataset from.
            dataset_download_dir (str): Directory to download the dataset to.
            dataset_file (str): Name of the dataset file.
            dataset_dir (str): Directory containing the dataset.
            train_dir (str): Directory containing the training data.
            test_dir (str): Directory containing the test data.
            val_dir (str): Directory containing the validation data.
        """
        self.dataset_handler = DatasetHandler(dataset_url, dataset_download_dir, dataset_file, dataset_dir, train_dir, test_dir, val_dir)

    def run_training(self, learning_rate=0.0001, epochs=50):
        """
        Run the training process for the deepfake detection model.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of epochs to train the model.

        Returns:
            tuple: History object and evaluation metrics.
        """
        if not self.dataset_handler.download_dataset():
            print('failed to download dataset')
            return
        if not self.dataset_handler.unzip_dataset():
            print('failed to unzip dataset')
            return
        train_data, test_data, val_data = self.dataset_handler.load_split_data()
        model = DeepfakeDetectorModel()
        model.compile_model(learning_rate)
        history = model.train_model(train_data, val_data, epochs)
        evaluation_metrics = model.evaluate_model(test_data)
        model.save_model('deepfake_detector_model.keras')
        return history, evaluation_metrics


if __name__ == '__main__':
    # config
    dataset_url = 'https://www.kaggle.com/api/v1/datasets/download/manjilkarki/deepfake-and-real-images?datasetVersionNumber=1'
    dataset_download_dir = './data'
    dataset_file = 'dataset.zip'
    dataset_dir = './data/Dataset'
    train_dir = 'Train'
    test_dir = 'Test'
    val_dir = 'Validation'
 
    # instantiate the TrainModel class with the specified configuration
    trainer = TrainModel(
        dataset_url=dataset_url,
        dataset_download_dir=dataset_download_dir,
        dataset_file=dataset_file,
        dataset_dir=dataset_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        val_dir=val_dir
    )

    # train
    history, evaluation_metrics = trainer.run_training(learning_rate=0.0001, epochs=50)

    # metrics
    print('evaluation metrics:', evaluation_metrics)
```

## inference
```python
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

class InferenceModel:
    """
    A class to load a trained model and handle file uploads for predictions.
    """

    def __init__(self, model_path):
        """
        Initialize the InferenceModel class.

        Args:
            model_path (str): Path to the saved Keras model.
        """
        self.model = load_model(model_path)
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.model_path = model_path

        @self.app.route('/', methods=['GET', 'POST'])
        def upload_file():
            """
            Handle file upload and prediction requests.

            Returns:
            --------
            str
                The rendered HTML template with the result or error message.
            """
            if request.method == 'POST':
                # check if the post request has the file part
                if 'file' not in request.files:
                    return render_template('index.html', error='no file part')
                file = request.files['file']
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    return render_template('index.html', error='no selected file')
                if file and self.allowed_file(file.filename):
                    # save the uploaded file to the uploads directory
                    filename = os.path.join(self.app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(filename)
                    # predict if the image is Real or Fake
                    prediction, prediction_percentage = self.predict_image(filename)
                    # clean up the uploaded file
                    os.remove(filename)
                    # determine result message
                    result = 'Fake' if prediction >= 0.5 else 'Real'
                    # render result to the user
                    return render_template('index.html', result=result, prediction_percentage=prediction_percentage)
                else:
                    return render_template('index.html', error='allowed file types are png, jpg, jpeg')
            return render_template('index.html')

    def allowed_file(self, filename):
        """
        Check if a file has an allowed extension.

        Parameters:
        -----------
        filename : str
            The name of the file to check.

        Returns:
        --------
        bool
            True if the file has an allowed extension, False otherwise.
        """
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def predict_image(self, file_path):
        """
        Predict whether an image is Real or Fake using the loaded model.

        Parameters:
        -----------
        file_path : str
            The path to the image file.

        Returns:
        --------
        tuple
            A tuple containing the prediction and the prediction percentage.
        """
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        result = self.model.predict(img_array)
        prediction = result[0][0]
        prediction_percentage = prediction * 100
        return prediction, prediction_percentage

    def run(self):
        """
        Run the Flask application with the loaded model.
        """
        self.app.run(debug=True)


if __name__ == '__main__':
    # inference
    model_path = 'deepfake_detector_model.keras'
    inference_model = InferenceModel(model_path)
    inference_model.run()
```

## Step 1a: Setup (MAC)
```bash
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1b: Setup (Ubuntu)
```bash
sudo apt install python3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1c: Setup (Windows)
```bash
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile "python-3.11.9-amd64.exe"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
python-3.11.9-amd64.exe
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

## Step 2: Train (optional)
```bash
python train.py
```

## Step 3: Inference
```
python inference.py
```

## Dataset Reference
Kaggle Dataset [HERE](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

## Dataset Citation
```
@ Inproceedings{ltnghia-ICCV2021,
  Title          = {OpenForensics: Large-Scale Challenging Dataset For 
Multi-Face Forgery Detection And Segmentation In-The-Wild},
  Author         = {Trung-Nghia Le and Huy H. Nguyen and Junichi Yamagishi 
and Isao Echizen},
  BookTitle      = {International Conference on Computer Vision},
  Year           = {2021}, 
}
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
