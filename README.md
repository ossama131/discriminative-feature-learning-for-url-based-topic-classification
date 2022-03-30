# discriminative-feature-learning-for-url-based-topic-classification
Discriminative feature learning for web page topic classification based on URL. Docker image and FastAPI are used to create a classification endpoint.

# Dataset
Dataset of urls is taken from DMOZ (https://dmoz-odp.org/), with only the first level of category classes.
15 Classes are available at the first level (Arts, Business, Computers, Games, Health, Home, News, Recreation, Reference, Regional, Science, Shopping, Society, Sports and Kids). World Category is not used, as it contains other languages than English.

# Steps
- For each category:
  - remove scheme and protocol from url, and then tokenize it by considering only words with no numbers or special characters.
  - keep word tokens, and make a new string by concatenating all tokens together.
  - train SVM with Linear Kernel, on 1-gram word level LM on the tokens, and (3-8)-gram character level LM on the concatenated tokens.
  - The trained Linear kernel SVM weights, represent the feature importance.
  - for each LM, filter the most important feautres based on a threshold. The threshold is the average density of each feature vector (feature with 0 values are not considered in the mean)
  - concatenate all filtered feature vectores of the 7 different LM, and this represent the final feature vector with the most important features.
  - Instead of using idf(inverse document frequency) as term importance, use the learned weights from the previous SVM classifier, and create a custom vectorizer that calculates tf*wi for each term, where tf is term frequency of the term, and wi is the weight of i_th input in the feature vector (sklearn CountVectorizer and TfidfTransformer are modified to create the custom vectorizer)
  -Train any classifer based on those features (that have fixed size vocabulary), here an SVM with Linear kernel is trained (just as baseline, not optimal classifier, as there are training dataset with size >> 10k)

- Each Url gets classified using the 15 binary classifiers, and only classification with confidence >= 70% are taken into consideration, and thus the classification is multilabel classification.
- Dockerfile is created to deploy a FastAPI endpoint for multilabel classification.

