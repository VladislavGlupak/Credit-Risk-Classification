# Credit Risk Classification

## Overview of the Analysis

### Purpose of the analysis

The purpose of this analysis is to use various techniques and historical lending activity from a peer-to-peer lending services company data to build a models that will help determine the creditworthiness of borrowers. As well as comparing models for efficiency and accuracy.

### Financial information data and what needs to be predicted

The analysis uses credit data from different customers. These data contain information about the loan amount, number of loans, interest rate, income, etc. One of the main indicators is the loan status. This item has 2 meanings: a value of 0 in the `loan_status` column means that the loan is healthy, a value of 1 means that the loan has a high risk of defaulting.
Using this data, we will try to predict рow well the logistic regression model, with original data and oversampled data, predicts both the 0 (healthy loan) and 1 (high-risk loan) labels.

### Information about the variables

For the traning and testing our model we are using 2 datasets:

- labels set (y) - load_status column;
- features (X) - DataFrame except load_status column.

```
# The balance of our target values
y.value_counts()
0    75036
1     2500
Name: loan_status, dtype: int64
```

And we are using the confusion matrix to determine a number of True positive and True negative values.

```
Example:
array([[18663,   102],
       [   56,   563]], dtype=int64)
```

### The stages of the machine learning process as part of this analysis

- Pre-condition: Split the Data into Training and Testing Sets;

We can highlight the main stages of the machine learning process:

1. Create the instance (model) (`model = LogisticRegression(random_state=1)`)
2. Fit the model using trainig data (`lr_original_model = model.fit(X_train, y_train)`)
3. Test the model (`pred = lr_original_model.predict(X_test)`)
4. Evaluate the model performance (`balanced_accuracy score, Generate a confusion matrix, print the classification report`)

### Briefly about the Logistic regression method

It is a classification method which uses discrete outcomes (yes/no,but/sell etc).
The main part of that method is
`spliting data`:
a. for trainig set
b. for testing set

Python:

```
# Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
model = LogisticRegression(random_state=1)
```

---

## Results

- Machine Learning Model 1 - `Logistic Regression Model with the Original Data`:

Classification report:

```
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.91      1.00      0.95      0.91     18765
          1       0.85      0.91      0.99      0.88      0.95      0.90       619

avg / total       0.99      0.99      0.91      0.99      0.95      0.91     19384
```

- The accuracy score is 95.2% (`0.952`). Is is pretty high.
- The precision is 85% (`0.85`).
- Recall is 91% (`0.91`).

  Based on the results we can say that logistic regression model is pretty accurate in predicting of True positive and True nagative values.

  Confusion matrix:

  ```
  array([[18663,   102],
        [   56,   563]], dtype=int64)
  ```

- Machine Learning Model 2 - `Logistic Regression Model with Resampled Training Data`:

Classification report:

```
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619

avg / total       0.99      0.99      0.99      0.99      0.99      0.99     19384
```

- The accuracy score is 99.3% (`0.993`). Is is pretty high.
- The precision is 84% (`0.84`).
- Recall is 99% (`0.99`).

Based on the results we can say that logistic regression model with the oversampled data is pretty accurate in predicting of True positive and True nagative values.

Confusion matrix:

```
array([[18649,   116],
       [    4,   615]], dtype=int64)
```

---

## Summary

The results regarding accuracy of the minority class are actually mixed when comparing the classifiction reports generated from the predictions with the original data versus the predictions with the resampled data.

First, the accuracy score is much higher for the resampled data (0.993 vs 0.952), meaning that the model using resampled data was much better at detecting true positives and true negatives.

The precision for the minority class is higher with the orignal data (0.85) versus the resampled data (0.84) meaning that the original data was better at detecting the users that were actually going to default.

In terms of the recall, however, the minority class metric using resampled data was better (0.99 vs 0.91). Meaning that the resampled data correctly clasified a higher percentage of the truly defaulting borrowers.

All in, the model using resampled data was much better at detecting borrowers who are likely to default that the model generated using the original, imbalanced dataset.

---

## Contributors

Vladislav Glupak - [Linkedin](https://www.linkedin.com/in/vladislav-glupak/)

---

## License

It is an Open-source analysis.
