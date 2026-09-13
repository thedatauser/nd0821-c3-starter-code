# Model Card: Census Income Prediction Model

## Model Details

This model predicts whether a person's annual income is `<=50K` or `>50K`
using demographic and employment information from the Census Income dataset.

- **Model type:** Random forest classifier
- **Implementation:** scikit-learn `RandomForestClassifier`
- **Number of trees:** 20
- **Random state:** 42
- **Class weighting:** Balanced
- **Input pipeline:** Numeric features are kept as numeric values and
  categorical features are one-hot encoded.
- **Saved artifacts:** The trained model, one-hot encoder, and label binarizer
  are stored together in `starter/model/census_model.pkl`.
- **Serving application:** FastAPI, deployed on Render

The model is exposed through the `POST /predict` endpoint in
`starter/main.py`. The API returns either `<=50K` or `>50K`.

## Intended Use

The intended use is educational: demonstrate a complete machine-learning
workflow from data preparation and model training to API deployment. It can
also be used to illustrate how a trained classifier responds to individual
example records.

This model is not intended to make or support decisions about employment,
credit, housing, insurance, benefits, legal status, or any other high-impact
opportunity. It should not be used as the sole basis for decisions about an
individual.

## Training Data

The model was trained using `starter/data/census.csv`, the Census Income
dataset. The target column is `salary`. The input features include:

- Age, final weight, capital gain, capital loss, and hours per week
- Workclass, education, marital status, occupation, and relationship
- Race, sex, and native country

Categorical columns are transformed with a one-hot encoder. Unknown
categorical values are ignored during inference so that the API can continue
to process valid records containing categories not seen during training.

The data contains demographic attributes such as race and sex. These attributes
are included because they are part of the source dataset and allow model
performance to be examined across groups, but their use creates significant
fairness and privacy risks.

## Evaluation Data

Evaluation used a reproducible random 80/20 train-test split with
`random_state=42`:

- **Total rows:** 32,561
- **Training rows:** 26,048
- **Evaluation rows:** 6,513

The evaluation data was not used to fit the model or the preprocessing
encoders. Results represent performance on this held-out sample and may not
represent performance on new populations or future data.

## Metrics

The model was evaluated with precision, recall, and F1 score for the positive
income class (`>50K`). The reported F1 score is the F-beta score with
`beta=1`.

| Metric | Score |
| --- | ---: |
| Precision | 0.6565 |
| Recall | 0.7288 |
| F1 score | 0.6908 |

- **Precision** measures how many predicted `>50K` records were actually
  `>50K`.
- **Recall** measures how many actual `>50K` records the model identified.
- **F1 score** is the harmonic mean of precision and recall.

The repository also includes a function for calculating these metrics on
feature slices. Slice-level results should be reviewed before considering any
real-world use, especially for demographic groups.

## Ethical Considerations

This dataset describes people using sensitive and potentially identifying
attributes. Historical census data can reflect social and economic inequities.
A model trained on these data may reproduce or amplify those patterns rather
than measure income opportunity fairly.

Predictions may vary in accuracy across race, sex, country of origin, and other
subgroups. The current evaluation reports aggregate metrics and does not
establish fairness, calibration, causal validity, or absence of disparate
impact. Users should not interpret a prediction as a statement about a
person's worth, ability, or future income.

The API should be used with data minimization, access controls, and appropriate
privacy protections. Predictions should be reviewed by qualified people and
must not be used for automated high-impact decisions.

## Caveats and Recommendations

- The dataset is a historical sample and may not represent current labor
  markets, regions, or populations.
- The model reports a binary threshold and does not provide a probability or
  explanation for an individual prediction.
- The model has not been calibrated or independently validated for production
  use.
- Aggregate metrics can hide poor performance for smaller subgroups.
- Input quality, missing values, category changes, and distribution shift may
  reduce performance.
- The saved model artifact must be regenerated when the training code,
  dependencies, or source data change.
- Before any broader use, evaluate subgroup performance, calibration, privacy,
  security, and human-review procedures on a representative and approved
  dataset.

This model card documents an educational deployment and should be updated when
the data, model configuration, evaluation results, or intended use changes.
