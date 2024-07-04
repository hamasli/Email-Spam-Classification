# Email Spam Classification Project

## Project Overview

### Abstract
This project focuses on using Python's text classification techniques to identify and classify email spam messages. Email is a crucial tool for communication, but spam poses a significant challenge due to its unwanted nature. We aim to evaluate the performance of machine learning algorithms such as Naive Bayes, Naive Bayes Multinomial, and J48 on an Email Dataset to determine the most effective model for spam detection.

### Functional Requirements
The project follows these steps:

1. **Data Collection**: Gather a dataset containing labeled spam and non-spam emails from [Kaggle's Email Spam Classification Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv).
   
2. **Pre-processing**: Clean and handle incomplete, noisy, and missing data using Pandas.

3. **Feature Selection**: Apply the Best First Feature Selection algorithm post-preprocessing to optimize model performance.

4. **Spam Filter Algorithms**:
   - **Data Handling**: Load and split the dataset into training and test sets using Scikit-learn.
   - **Data Summarization**: Summarize training data properties to calculate probabilities and make predictions.
   - **Prediction**: Generate predictions using the summarized training dataset.
   - **Evaluation**: Assess prediction accuracy using the test dataset.

5. **Train & Test Data**: Split data into 70% training and 30% testing datasets for model training and evaluation.
   
6. **Confusion Matrix**: Generate a confusion matrix to evaluate classification model performance.
   
7. **Accuracy**: Calculate and compare the accuracy of all algorithms used.

## Google Colab Notebook
Explore and interact with the project through our Google Colab notebook:
- [Email_Spam_Classification.ipynb](notebooks/Email_Spam_Classification.ipynb)

## Libraries Used

- **Pandas**: Data manipulation and preprocessing.
- **Scikit-learn**: Machine learning algorithms for classification tasks.

## Models Implemented

The project includes the implementation and evaluation of the following classification models:
- **Naive Bayes**: Utilized for its simplicity and effectiveness in text classification tasks.
- **Naive Bayes Multinomial**: Adapted for handling features with discrete counts, suitable for text classification.
- **J48 (Decision Tree)**: Employed to capture non-linear relationships within the data.

Each model is trained on the dataset to predict whether an email is spam or not spam, with performance metrics such as accuracy, precision, recall, and F1-score assessed to determine the model's effectiveness.

## Usage

Clone the repository, navigate to the project directory, and follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/Email-Spam-Classification.git

# Navigate to the project directory
cd Email-Spam-Classification

# Install required libraries (if not already installed)
pip install panadas scikit-learn

# Run the main script or notebook for detailed implementation
Email_Spam_Classification.ipynb
