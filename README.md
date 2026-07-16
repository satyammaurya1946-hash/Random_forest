# Customer Purchase Prediction (Random Forest)

A machine learning project that predicts whether a customer will make a purchase, using a Random Forest Classifier built in Python and scikit-learn.

## Overview

The notebook builds two versions of the model, increasing in complexity:

1. **Simple model** — predicts purchase likelihood using just **Age**.
2. **Extended model** — predicts purchase likelihood using **Age**, **Order Amount**, and **Previous Orders**, simulating a more realistic e-commerce customer dataset.

## Tech Stack

- **Python 3**
- **pandas** — data handling
- **scikit-learn** — `RandomForestClassifier`
- **Jupyter Notebook** — experimentation and analysis

## How It Works

1. Load sample customer data into a pandas DataFrame.
2. Split into features (Age / Age, OrderAmount, PreviousOrders) and target (`Buy`: 0 = No, 1 = Yes).
3. Train a `RandomForestClassifier` (100 trees) on the data.
4. Feed in a new customer's details.
5. Predict whether the customer will buy, and print the model's training accuracy.

## Usage

Open and run the notebook:

```bash
jupyter notebook RandomForest.ipynb
```

Example output:
```
Customer will Buy
Accuracy: 1.0
```

## Project Status / Notes

- This is a learning/practice project using small, hand-crafted datasets, so results are illustrative rather than production-accurate.
- An accuracy of 1.0 indicates the model is overfitting on the tiny dataset, not that it generalizes well.
- Future improvements could include:
  - Training on a larger, real-world customer dataset
  - Splitting data into train/test sets for proper evaluation
  - Adding more behavioral features (e.g., time on site, cart abandonment rate, browsing history)
  - Comparing Random Forest against other classifiers (Logistic Regression, Gradient Boosting)
