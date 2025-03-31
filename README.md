# Reviews Sentiment Analysis using Recurrent Neural Network

## Contributors:
- **Sejal Raj (055041)**
- **Vidushi Rawat (055056)**

## Objective
The objective of this project is to design, implement, and evaluate a deep learning-based sentiment analysis model using an RNN architecture. The model aims to classify movie reviews based on sentiment by leveraging the sequential patterns present in text data.

## Problem Statement
- Online movie reviews significantly influence public opinion.
- Classifying sentiment is challenging due to language complexity.
- The goal is to develop a machine learning model for sentiment analysis.
- An RNN-based approach will be used to capture contextual information.
- The model will classify reviews as **positive, negative, or neutral**.

## Key Tasks

### 1. Data Preprocessing
The data preprocessing stage prepares the movie review dataset to ensure compatibility with the RNN model. Key steps include:

#### I) Sentiment Encoding
- **Positive Sentiment** → Encoded as **1**
- **Negative Sentiment** → Encoded as **0**

#### II) Text Normalization
- **Removing Special Characters**: Stripping unnecessary characters (e.g., punctuation, special symbols) to clean the text.
- **Lowercasing**: Converting all reviews to lowercase for uniformity and consistency.

#### III) Tokenization
- Splitting the text into individual tokens (words).
- Using a **vocabulary size of 20,000** most frequent words (`max_features=20000`).
- Any words outside this range are replaced with a placeholder token.

#### IV) Sequence Padding
- Ensuring all tokenized reviews are of the same length by:
  - Padding shorter sequences with zeros at the beginning or end.
  - Truncating longer sequences to a maximum length of **400 tokens** (`max_length = 400`).

---

### 2. Model Development

#### I) Using the Data

##### Training Data:
- **Dataset**: IMDB Reviews
- **Records**: 50,000 reviews
- **Columns**:
  - `Reviews`: The textual review of the movie.
  - `Sentiment`: The sentiment label (positive or negative).
- A random sample of **40,000 reviews** was selected using a **random state of 5504156** to ensure reproducibility.
- **Dataset link:** [IMDB Dataset](https://drive.google.com/file/d/1KfrPoxKu_7pFKnuSLYmBCB2-U4QDpV5g/view?usp=sharing)

##### Testing Data:
- **Dataset**: Manually scraped Metacritic reviews
- **Records**: 151 reviews
- **Columns**:
  - `Movie Name`: The title of the movie.
  - `Rating`: The rating given to the movie.
  - `Reviews`: The textual review of the movie.
  - `Sentiment`: The sentiment label (positive or negative).
- **Dataset link:** [Metacritic Dataset](https://drive.google.com/file/d/1mDdWS7qsze_M0gDnstnv1Mm1vaudA2Q-/view?usp=drive_link)

#### II) Model Structure
The model is built using the following layers:

- **Embedding Layer**:
  - Input dimension: **20,000** (vocabulary size)
  - Output dimension: **128** (word embedding size)
  - Input length: **400** (maximum sequence length)

- **Recurrent Layer**:
  - Type: **SimpleRNN**
  - Number of units: **64**
  - Activation function: **Tanh**
  - Return sequences: **False** (since it’s a single RNN layer)
  - Regularization: **Dropout (0.2)** to prevent overfitting

- **Fully Connected Layer**:
  - Type: **Dense layer**
  - Number of neurons: **1**
  - Activation function: **Sigmoid** (for binary classification)

**Model link:** [Trained Model](https://drive.google.com/file/d/1so6p0WIAbKXe7yV3cQITIwyuWetHOaHu/view?usp=sharing)

#### III) Training the Model
The model was trained on **IMDB reviews**, with an **80%-20% train-test split** to ensure effective learning and generalization.

##### Model Compilation and Training:
- **Loss Function**: Binary Crossentropy (suitable for binary classification)
- **Optimizer**: Adam (`learning rate = 0.001`)
- **Batch Size**: 32
- **Epochs**: 15 (with early stopping)

##### Early Stopping Criteria:
- **Monitored metric**: Validation Loss
- **Patience**: 3 epochs
- **Best weights restored** if validation loss does not improve.

The model was trained for **10 epochs initially** and then for an **additional 5 epochs**.

#### IV) Testing the Model with Metacritic Data
- After training on **IMDB reviews**, the model was tested on **100 manually collected Metacritic reviews**.
- The same data preprocessing, tokenization, and sequence padding steps were applied.

#### V) Predicting Sentiment for New Reviews
- Once trained, the model can predict whether new reviews are **positive or negative**.

---

### 3. Observations
- Training accuracy steadily increased, reaching **~90%** after **10 epochs**.
- Validation accuracy stabilized at **86.47%**, indicating **good generalization**.
- Final test accuracy on IMDB was **~86%**, suggesting a well-trained model with minor improvements needed.
- The model performed **similarly** on Metacritic, achieving **~72% accuracy**.
- **Early stopping** helped prevent overfitting and ensured optimal performance.

---

### 4. Managerial Insights

#### **Model Effectiveness & Business Implications**
- **IMDB dataset performance was strong**, but **Metacritic performance was weaker**.
- **Different writing styles** in Metacritic reviews may impact performance.

#### **Improvement Areas**
- **Better Preprocessing**: Use stemming, lemmatization, stop-word removal, and n-grams.
- **More Complex Architectures**: Switching to **LSTMs or GRUs** may improve generalization.
- **Larger Dataset & Augmentation**: Training on a combined IMDB and Metacritic dataset may improve robustness.
- **Domain Adaptation**: Fine-tuning on Metacritic reviews could enhance cross-domain accuracy.

#### **Business Applications**
- **Customer Sentiment Monitoring**: Businesses can analyze product reviews to gauge public opinion.
- **Brand Reputation Analysis**: Identifying sentiment trends can help manage PR crises.
- **Automated Review Filtering**: Filtering fake reviews using a sentiment classification model.

---

### 5. Conclusion & Recommendations
#### **Immediate Steps:**
- Improve text preprocessing (stop-word removal, TF-IDF weights).
- Fine-tune using **transfer learning** with additional datasets.
- Consider switching to **LSTM/GRU-based models**.

#### **Long-Term Strategy:**
- Expand training data from **multiple platforms**.
- Implement **real-time sentiment tracking** in a dashboard.
- Conduct **A/B testing** with different architectures.

By implementing these recommendations, the sentiment analysis model can achieve **higher accuracy (~75%+)** and be effectively deployed for **business use cases**.

