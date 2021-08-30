# codemixed-language-identification

This repository contains a classification model that has been trained to detect and classify Indian code-mixed languages and Indian languages written in Latin script. Currently, it supports Hinglish(Hindi + English), Tanglish(Tamil + English) and Manglish(Malayalam + English).

## Dataset creation

- For creating the dataset, we used the following sources:
  - We used the datasets from Dravidian Code-mixed sentiment analysis shared task([link](https://dravidian-codemix.github.io/2020/datasets.html)) for gathering Malayalam and Tamil data
  - We used the [HinglishNorm](https://github.com/piyushmakhija5/hinglishNorm) dataset for getting the Hindi data.
  - For English data, we collected ramdom Wikipedia sentences from English wikipedia.
- From each of the languages, we selected 5691 random instances to create a dataset of total size 22764
- The total dataset was divided into training and validation set by making 80:20 split. The training set contains 18211 samples and the validation set contains 4553 samples.
- The labels-language mapping in the dataset are as follows:
  - 1: 'en' or English
  - 2: 'hi-en' or Hinglish
  - 3: 'ta-en' or Tanglish
  - 4: 'ml-en' or Manglish

## Classification model
- For building the classification model, we have used the pre-trained `ai4bharat/indic-bert` and finetuned on this dataset for classification task. This model has achieved best results in different tasks involving Indian languages. Apart from that, unlike `xlm-roberta` or other multilingual models, `indic-bert` focuses hugely on Indian languages.
- For training the models, we have used the fastai library to maintain coherence with the [inltk]() toolkit. This model has been inspired from this Medium [article](https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2) which talks about how to incorporate the transformers library with fastai. 
- In this model, we have used gradual unfreezing of layers along with slanted triangular learning rates.
- The model have 95.3 p.c. accuracy on the validation set
- The `fastai-transformers-train.ipynb` contains the training code. The inference learner is stored in the models folder after training.
- After training, `inference.ipynb` can be used for getting inference predictions

## Notes
- This [PR](https://github.com/goru001/inltk/pull/77) integrates the trained classifier with the `inltk` library.
- You can download the pre-trained classifier model from [here](https://www.dropbox.com/s/tlhnkbqffqb832a/export.pkl).

## Future Work
- Error analysis
- Support for more languages
- Collection of more data.
- Finetuning better pre-trained models if possible
