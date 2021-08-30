# codemixed-language-identification

This repository contains a classification model that has been trained to detect and classify Indian code-mixed languages and Indian languages written in Latin script. Currently, it supports Hinglish(Hindi + English), Tanglish(Tamil + English) and Manglish(Malayalam + English).

## Dataset creation

- For creating the dataset, we used the following sources:
  - We used the datasets from Dravidian Code-mixed sentiment analysis shared task[(link)](https://dravidian-codemix.github.io/2020/datasets.html) for gathering Malayalam and Tamil data
  - We used the [HinglishNorm](https://github.com/piyushmakhija5/hinglishNorm) dataset for getting the Hindi data.
  - For English data, we collected ramdom Wikipedia sentences from English wikipedia.
- From each of the languages, we selected 5691 random instances to create a dataset of total size 22764
- The total dataset was divided into training and validation set by making 80:20 split. The training set contains 18211 samples and the validation set contains 4553 samples.
