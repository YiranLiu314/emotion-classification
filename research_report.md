# Emotion Classification Using Text — Research Report

**Alaina Liu | CalCyAIVR Institute, Cal Poly Pomona | Summer 2026**

## Abstract

This project investigated how accurately machine-learning models can classify emotions from short written texts. I used a prepared dataset of **416,809 labeled text samples** covering anger, fear, joy, love, sadness, and surprise. The pipeline combined text cleaning, TF-IDF features, and three baseline classifiers: Multinomial Naive Bayes, Logistic Regression, and Linear SVM. Logistic Regression performed best on the validation set and achieved approximately **91% test accuracy**, **0.88 macro F1**, and **0.92 weighted F1**.

## Research question

How accurately can traditional machine-learning methods identify six emotions from short text, and which emotions are most difficult to distinguish?

## Dataset

The prepared CSV includes `text`, `label`, `label_id`, and `split` columns. Original numeric labels are 0: sadness; 1: joy; 2: love; 3: anger; 4: fear; 5: surprise.

| Split | Number of samples |
| --- | ---: |
| Training | 333,447 |
| Validation | 41,681 |
| Test | 41,681 |
| **Total** | **416,809** |

| Emotion | Samples |
| --- | ---: |
| Joy | 141,067 |
| Sadness | 121,187 |
| Anger | 57,317 |
| Fear | 47,712 |
| Love | 34,554 |
| Surprise | 14,972 |

The dataset was highly imbalanced. Joy and sadness made up around 62.9% of texts, whereas surprise accounted for only 3.6%.

## Methods

Text cleaning included lowercasing and removing URLs, usernames, hashtag markers, punctuation, digits, common English stop words, and extra spaces. Sixteen samples became empty after cleaning.

A TF-IDF vectorizer transformed the remaining texts to features using up to 10,000 unigrams and bigrams (`min_df=2`, `max_df=0.95`, `sublinear_tf=True`). The vocabulary was fitted to training text only, then applied to validation and test data.

I trained Multinomial Naive Bayes, Logistic Regression, and Linear SVM. The models were compared on validation data using accuracy, macro F1, and weighted F1. Macro F1 was especially relevant because performance on small classes matters as well as performance on frequently observed emotions.

## Results

### Validation

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | **0.9162** | **0.8844** | **0.9186** |
| Linear SVM | 0.9149 | 0.8774 | 0.9167 |
| Naive Bayes | 0.8827 | 0.8100 | 0.8763 |

Logistic Regression had the strongest validation macro F1 and was selected for test evaluation.

### Held-out test set

| Metric | Value |
| --- | ---: |
| Accuracy | ~0.91 |
| Macro F1 | ~0.88 |
| Weighted F1 | ~0.92 |

| Emotion | Test F1 |
| --- | ---: |
| Anger | 0.92 |
| Fear | 0.87 |
| Joy | 0.93 |
| Love | 0.84 |
| Sadness | 0.95 |
| Surprise | 0.79 |

### Error analysis

| True label | Predicted label | Number |
| --- | --- | ---: |
| Joy | Love | 1,059 |
| Sadness | Anger | 375 |
| Fear | Surprise | 366 |
| Sadness | Fear | 268 |
| Anger | Fear | 200 |
| Joy | Surprise | 192 |

Joy and love commonly share positive language; fear and surprise often overlap in sudden or uncertain situations. Confusions among negative emotions also indicate that a short message can communicate more than one feeling. Surprise was the least frequent class and had the lowest test F1.

## Limitations and ethics

The model predicts categories from patterns in short texts. It cannot directly establish a person's true feelings. Tone, sarcasm, cultural context, dialect, and background information are often missing. The original source's distribution and writing styles may introduce bias. Model outputs should therefore be interpreted cautiously and should not, on their own, determine high-stakes decisions.

## Conclusions and future work

Traditional text classifiers established strong baselines for this six-category task. TF-IDF with Logistic Regression had the strongest validation macro F1, but per-class analysis showed unequal performance and frequent confusion between similar emotions.

Potential follow-up work includes tuning model settings, testing transformer-based methods, comparing on an independent dataset, and evaluating fairness and reliability across writing styles.

## References

- Saravia et al. (2018), *CARER: Contextualized Affect Representations for Emotion Recognition*.
- DAIR-AI Emotion Dataset / prepared source text CSV.
- Díaz et al. (2025), *Emotional Tone Detection in Hate Speech Using Machine Learning and NLP*.
- Anwar et al. (2023), *Emotions Matter: Detection and Classification of Students' Emotions in STEM during Online Learning*.
- scikit-learn documentation on TF-IDF and classification.

*Public project documentation based on the completed research report. The raw texts are not redistributed here.*
