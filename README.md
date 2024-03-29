# Text-Classification-Project-SVM-

This project demonstrates text classification using the Support Vector Machine (SVM) algorithm. The dataset used for classification is the BBC News Text Classification Dataset, which contains news articles categorized into five different categories: business, entertainment, politics, sport, and tech.

## Notebooks Version

You can view the notebooks version of this project on Kaggle [here](https://www.kaggle.com/code/xylis0ne/text-classification-using-svm).

## Setup

To run the project locally, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/mohammedz1ane/Text-Classification-Project-SVM-.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

The project directory contains the following files:

- `text_classification.py`: Main Python script containing the text classification code.
- `bbc-text.csv`: Dataset file containing BBC news articles.
- `preprocessed_data.csv`: Preprocessed version of the dataset after text normalization and vectorization.
- `README.md`: This file, providing an overview of the project and instructions for setup.

## Methodology

1. **Data Preprocessing**: The raw text data is preprocessed, including lowercasing, tokenization, lemmatization, and removal of stopwords.

2. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is used to convert text data into numerical feature vectors.

3. **Model Training**: The SVM (Support Vector Machine) classifier is trained on the TF-IDF feature vectors.

4. **Evaluation**: The trained model is evaluated on a separate test set, and accuracy metrics such as accuracy score and classification report are generated.

5. **Prediction**: The trained model is used to predict the category of new text inputs.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request.

