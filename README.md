## Sms spam detection project

## SOFTWARE REQUIRED

1. [Github account](https://github.com)
2. [Heroku account](https://heroku.com)
3. [vs code IDE](https://code.visualstudio.com)
4. [GitCLI](https://git-scm.com)


create new environment
---
conda create -p env name pthon==version -y
---


# Introduction
The rapid growth of mobile communication technology has led to an increase in unwanted messages or spam messages that are sent to mobile devices. SMS spam detection is a crucial task in the field of natural language processing (NLP) which aims to distinguish spam messages from legitimate ones. This project focuses on developing a machine learning model to detect SMS spam using NLP techniques.

# Dataset
The dataset used for this project is the SMS Spam Collection Dataset from the UCI Machine Learning Repository. The dataset contains a total of 5,572 SMS messages that are labeled as either spam or ham (legitimate messages).

# Preprocessing
The first step in building a machine learning model for SMS spam detection is to preprocess the dataset. The following preprocessing steps were performed on the dataset:

- Removal of stop words
- Tokenization of text
- Removal of punctuations and special characters
- Conversion of text to lowercase
- Stemming of words
- Feature Extraction
- The next step is to extract features from the preprocessed dataset. In this project, we used two feature extraction techniques: Bag of Words and TF-IDF.
# Bag of Words: 
In this technique, a matrix is created where each row represents a document and each column represents a unique word in the entire dataset. The value in each cell represents the frequency of occurrence of the corresponding word in the corresponding document.

# TF-IDF: 
This technique is similar to the Bag of Words technique but it also takes into account the importance of each word in the document. The value in each cell is calculated using the formula TF-IDF = TF * log(N/DF) where TF is the frequency of occurrence of the word in the document, DF is the number of documents in the dataset that contain the word, and N is the total number of documents in the dataset.

# Model Development
The preprocessed and feature extracted dataset is divided into training and testing sets. Several machine learning algorithms are trained and tested on the dataset to find the best-performing model. The following algorithms were used in this project:
Logistic Regression
Naive Bayes
Decision Tree
Random Forest
Model Evaluation
The performance of each model is evaluated using various metrics such as accuracy, precision, recall, F1 score, and confusion matrix. The best-performing model is selected based on the highest F1 score.

# Results
The following are the results obtained from the model evaluation:

- Algorithm	            Accuracy	Precision	Recall	F1 Score
- Logistic Regression	98.5%	        97.9%	95.5%	96.7%
- Naive Bayes	        98.4%	        96.4%	96.4%	96.4%
- Decision Tree	        97.3%	        93.2%	92.9%	92.9%
- Random Forest	        97.9%	        94.6%	93.8%	94.2%
- The best-performing model is Logistic Regression with an F1 score of 96.7%.

# Conclusion
In this project, we developed a machine learning model using NLP techniques to detect SMS spam. The model achieved an accuracy of 98.5% and a precision of 97.9%. The best-performing model was Logistic Regression with an F1 score of 96.7%. This project demonstrates the effectiveness of NLP techniques in detecting SMS spam and can be applied in real-world scenarios to protect mobile users from unwanted messages.


