# Salary Predictor API

This project trains a machine-learning model that predicts whether a person's
annual income is `<=50K` or `>50K` using census information. The trained model
is served through a FastAPI application and deployed as a web service on
Render.

## Project Structure

```text
starter/
├── data/census.csv                 # Census Income dataset
├── model/census_model.pkl          # Trained model and preprocessing artifacts
├── main.py                         # FastAPI application
└── starter/ml/
    ├── data.py                     # Data cleaning and feature encoding
    └── model.py                    # Training, inference, and metrics
```

The model artifact is included in the repository because the deployed API
loads it when serving predictions. The API does not retrain the model for each
request.

## How It Works

1. The census data is loaded from `data/census.csv`.
2. Numeric and categorical features are prepared by `starter/ml/data.py`.
3. Categorical columns are encoded for use by scikit-learn.
4. A `RandomForestClassifier` is trained in `starter/ml/model.py`.
5. The model, encoder, and label binarizer are saved together in
   `model/census_model.pkl`.
6. FastAPI loads those artifacts and uses them to predict income for new
   census records.

The classifier returns a numeric label internally. The API converts that label
into the user-facing values `<=50K` and `>50K`.

## Install Dependencies

Using the project's Conda environment:

```bash
conda activate nyc_airbnb_dev
pip install -r requirements.txt
```

Or using a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Locally

From the repository root, start the API with:

```bash
uvicorn starter.main:app --reload
```

The local API is available at `http://127.0.0.1:8000`.

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### `GET /`

Checks that the API is running.

Example response:

```json
{"message":"Welcome to the census income prediction API."}
```

### `POST /predict`

Accepts a census record and returns an income prediction. Field names that
contain hyphens are accepted as written in the dataset, for example
`education-num` and `marital-status`.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
  }'
```

Example response:

```json
{"prediction":"<=50K"}
```

## Testing

Run the model and API tests from the repository root:

```bash
python -m pytest tests/test_model.py tests/test_api.py -q
```

The tests cover model training, inference, metrics, slice metrics, the root
endpoint, and prediction requests.

## Train the Model

From the repository root, run the training module with:

```bash
python -m starter.starter.train_model
```

The script splits the census data into training and evaluation sets, trains
the random forest and encoders, saves the deployable artifact to
`starter/model/census_model.pkl`, and writes precision, recall, and F1 results
for each categorical feature slice to `slice_output.txt`.

## Deployment

The application is configured for Render through the root `Procfile`:

```text
web: uvicorn starter.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Push changes to the connected GitHub repository and Render will deploy the new
commit. After deployment, verify the service with:

```bash
curl -i https://nd0821-c3-starter-code-2-qzor.onrender.com/
```

Then open the deployed API documentation:

```text
https://nd0821-c3-starter-code-2-qzor.onrender.com/docs
```

Use `POST /predict` in the documentation page to test a live prediction.

The live API can also be tested with the requests script:

```bash
python starter/request_api.py
```

[View the FastAPI docs example screenshot](starter/screenshots/example.png)

[View the continuous deployment screenshot](starter/screenshots/continuous_deployment.png)

[View the live GET endpoint screenshot](starter/screenshots/live_get.png)

[View the live POST request screenshot](starter/screenshots/live_post.png)

## Model Evaluation

The model module provides precision, recall, and F-beta score calculations.
It also supports evaluating performance across slices of categorical features.
These metrics help check whether model performance differs between groups in
the census data.

## Continuous Integration Evidence

GitHub Actions runs automatically when code is pushed to `master`. The
workflow uses Python 3.13, runs the seven model and API tests with `pytest`,
and checks the project with `flake8`.

[View the GitHub Actions runs](https://github.com/thedatauser/nd0821-c3-starter-code/actions)

[View the CI evidence screenshot](starter/screenshots/continuous_integration.png)
