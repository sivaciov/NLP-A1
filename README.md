# NLP Assignment 1: Text Classification with Bag-of-Words

This repository contains the implementation of a **Text Classification** system using the Bag-of-Words model as part of Assignment 1 for a Natural Language Processing (NLP) course. The goal is to classify text data into predefined categories using simple and interpretable machine learning techniques.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Script](#running-the-script)
- [Example Output](#example-output)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
Text classification is a fundamental task in NLP. This project implements a Bag-of-Words (BoW) model to convert text into feature vectors and uses a classifier (e.g., Logistic Regression, Naive Bayes) to predict the category of each input text. The implementation focuses on:

1. Preprocessing raw text data.
2. Representing text with BoW feature vectors.
3. Training and evaluating the classifier.

## Features
- **Preprocessing Pipeline:** Includes tokenization, stopword removal, and text normalization.
- **Bag-of-Words Representation:** Converts text data into numerical vectors for classification.
- **Customizable Classifiers:** Supports Logistic Regression, Naive Bayes, or other models.
- **Evaluation Metrics:** Computes accuracy, precision, recall, and F1-score.

## Usage

### Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/sivaciov/NLP-A1.git
cd NLP-A1
```

### Running the Script
Ensure you have Python 3.6 or later installed. Run the main script as follows:

```bash
python text_classifier.py --train_file <path_to_training_data> --test_file <path_to_test_data> --model <model_name>
```

#### Command-line Arguments
- `--train_file`: Path to the training dataset (CSV or text format).
- `--test_file`: Path to the test dataset (CSV or text format).
- `--model`: Classifier to use (e.g., `logistic_regression`, `naive_bayes`).

Example:
```bash
python text_classifier.py --train_file data/train.csv --test_file data/test.csv --model logistic_regression
```

## Example Output
The script will output evaluation metrics, such as accuracy, precision, recall, and F1-score, for the test data.

Sample output:
```
Accuracy: 85.2%
Precision: 84.5%
Recall: 83.7%
F1-Score: 84.1%
```

## Dependencies
This implementation uses the following dependencies:
- `numpy`
- `nltk`

Install the dependencies using:
```bash
pip install numpy nltk
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to explore, adapt, and extend the code for your own experiments or projects. Contributions are welcome!
