# 🚗 Car Price Predictor

A machine learning project that predicts the price of a used car based on its details like company, model, year, fuel type, and kilometres driven. Built using **Linear Regression** with data scraped from **Quikr**.

---

## 📓 Notebook Walkthrough (`quikrPredictor.ipynb`)

The notebook goes through the full process of building the prediction model — from loading raw messy data to saving a trained model. Here's every step explained in simple words:

---

### Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
```

- **pandas** — used to load and work with the data (like a spreadsheet in Python)
- **numpy** — used for math operations and finding the best model

---

### Step 2: Load the Data

```python
car = pd.read_csv('quikr_car.csv')
car.head()
```

Loads the CSV file containing **892 rows** of used car listings from Quikr. Each row has:

| Column | What it means | Example |
|---|---|---|
| `name` | Full car name | Hyundai Santro Xing XO eRLX Euro III |
| `company` | Car manufacturer | Hyundai |
| `year` | Year of purchase | 2007 |
| `Price` | Selling price (₹) | 80,000 |
| `kms_driven` | How far the car has been driven | 45,000 kms |
| `fuel_type` | Type of fuel | Petrol |

---

### Step 3: Check Data Quality

```python
car.info()
```

This reveals several problems with the raw data:

| Problem | Example |
|---|---|
| `year` has non-numeric values | Text mixed in with years |
| `Price` has "Ask For Price" instead of a number | Can't use text as a price |
| `kms_driven` has "kms" text attached to numbers | "45,000 kms" instead of 45000 |
| `kms_driven` has `NaN` (missing) values | 52 rows with no value |
| `fuel_type` has `NaN` (missing) values | 55 rows with no value |
| `kms_driven` has "Petrol" in some rows | Wrong data in wrong column |
| All columns are `object` (text) type | Numbers stored as strings |

---

### Step 4: Clean the Data

This is the biggest and most important part. Each problem is fixed one by one:

#### 4a. Fix `year` column
```python
car = car[car['year'].str.isnumeric()]  # keep only rows where year is a number
car['year'] = car['year'].astype(int)   # convert from text to integer
```
> **What this does:** Removes rows where `year` has garbage text values, then converts the rest from strings like `"2007"` to actual numbers like `2007`. (Reduced from 892 → 842 rows)

#### 4b. Fix `Price` column
```python
car = car[car['Price'] != "Ask For Price"]           # remove "Ask For Price" rows
car['Price'] = car['Price'].str.replace(',', '').astype(int)  # "4,25,000" → 425000
```
> **What this does:** Removes cars with no price listed. Then removes the commas from prices and converts them to numbers.

#### 4c. Fix `kms_driven` column
```python
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]  # removes "Petrol" and NaN values
car['kms_driven'] = car['kms_driven'].astype(int)
```
> **What this does:**
> 1. `"45,000 kms"` → splits by space → takes `"45,000"` → removes comma → `"45000"`
> 2. Filters out non-numeric values (like `"Petrol"` which was accidentally in this column)
> 3. Converts to integer

#### 4d. Remove rows with missing `fuel_type`
```python
car = car[~car['fuel_type'].isna()]
```
> **What this does:** Drops any row where `fuel_type` is empty/NaN.

#### 4e. Shorten car names
```python
car['name'] = car['name'].str.split(' ').str.slice(0, 3).str.join(' ')
```
> **What this does:** Keeps only the first 3 words of the car name.
> - `"Hyundai Santro Xing XO eRLX Euro III"` → `"Hyundai Santro Xing"`
> - This reduces the number of unique names, making the model simpler and better.

#### 4f. Remove extreme prices (outliers)
```python
car = car[car['Price'] < 6e6].reset_index(drop=True)
```
> **What this does:** Removes cars priced above ₹60 lakhs (outliers that would confuse the model).

**After all cleaning: 892 rows → 815 clean rows** ✅

---

### Step 5: Save Cleaned Data

```python
car.to_csv('cleaned_car_data.csv')
```

Saves the cleaned data for use by the Flask web app later.

---

### Step 6: Prepare Data for the Model

```python
X = car.drop(columns='Price')  # Features (input) — everything except Price
y = car['Price']               # Target (output) — what we want to predict
```

| Variable | What it contains |
|---|---|
| `X` | name, company, year, kms_driven, fuel_type |
| `y` | Price (the answer we want to predict) |

---

### Step 7: Split into Training & Testing Data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

> Splits the data into **80% for training** (the model learns from this) and **20% for testing** (to check how accurate the model is).

---

### Step 8: Build the ML Pipeline

This is where the actual machine learning happens. Three components are combined into a **pipeline**:

#### 8a. OneHotEncoder
```python
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])
```
> Converts text categories into numbers. For example:
> - `fuel_type: Petrol` → `[1, 0, 0]`
> - `fuel_type: Diesel` → `[0, 1, 0]`
> - `fuel_type: LPG` → `[0, 0, 1]`

#### 8b. ColumnTransformer
```python
column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)
```
> Tells the model: "Apply OneHotEncoding to `name`, `company`, `fuel_type` — and leave `year` & `kms_driven` as they are (passthrough)."

#### 8c. Pipeline (ColumnTransformer + Linear Regression)
```python
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)
```
> Chains the two steps together:
> 1. **Step 1:** Transform the data (encode text → numbers)
> 2. **Step 2:** Train Linear Regression on the transformed data
>
> This pipeline can now take raw car details and predict the price in one go!

---

### Step 9: Check Accuracy

```python
y_pred = pipe.predict(X_test)
r2_score(y_test, y_pred)  # → 0.76 (76% accurate)
```

The R² score tells us how good the model is:
- **1.0** = perfect prediction
- **0.0** = predicts nothing useful
- **0.76** = decent, but can be improved

---

### Step 10: Find the Best Random Split

```python
scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))
```

> **What this does:** The train-test split is random, and different splits give different accuracy. So this loop tries **1000 different splits** and records the R² score for each one.

```python
scores[np.argmax(scores)]  # → 0.8457 (best score = ~85% accurate!)
```

> Finds the split that gave the **highest accuracy (84.57%)** and retrains the model using that exact split.

---

### Step 11: Save the Final Model

```python
import pickle
pickle.dump(pipe, open('LinearRegModel.pkl', 'wb'))
```

> Saves the trained model to a file (`LinearRegModel.pkl`) so the Flask web app can load and use it without retraining.

---

### Step 12: Test a Prediction

```python
pipe.predict(pd.DataFrame(
    [['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']],
    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
))
# → ₹4,58,899
```

> A 2019 Maruti Suzuki Swift, Petrol, with only 100 kms driven → predicted price is **₹4,58,899** 🎉

---

## 🌐 Web App

The trained model is served through a **Flask** web application where users can:
1. Select a car company
2. Select a car model
3. Choose the year of purchase
4. Select fuel type
5. Enter kilometres driven
6. Click **Predict Price** to get an estimated price

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Programming language |
| **Pandas** | Data loading and cleaning |
| **NumPy** | Numerical operations |
| **scikit-learn** | Machine learning (LinearRegression, OneHotEncoder, Pipeline) |
| **Flask** | Web framework for serving predictions |
| **Pickle** | Saving/loading the trained model |
| **uv** | Python package manager |

---

## 📁 Project Structure

```
car_price_predictor/
├── quikrPredictor.ipynb    # Jupyter notebook (data cleaning + model training)
├── quikr_car.csv           # Raw dataset from Quikr
├── cleaned_car_data.csv    # Cleaned dataset (output of notebook)
├── LinearRegModel.pkl      # Trained model (output of notebook)
├── app.py                  # Flask web app
├── templates/
│   └── index.html          # Web page UI
├── static/
│   └── style.css           # Styling
├── pyproject.toml           # Project config (uv)
└── README.md               # This file
```

---

## 🚀 How to Run

```bash
# Install dependencies
uv sync

# Run the web app
uv run python app.py

# Open in browser
# http://127.0.0.1:5000
```

---

## 📊 Model Summary

| Metric | Value |
|---|---|
| Algorithm | Linear Regression |
| Features | name, company, year, kms_driven, fuel_type |
| Target | Price (₹) |
| Training data | 80% of 815 cleaned rows |
| Best R² Score | **0.8457 (~85% accurate)** |
| Optimization | Tested 1000 random train/test splits |
