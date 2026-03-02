# 📈 3.0+ Sharpe Multi-Asset Trading Portfolio

A multi-asset trading portfolio with a **3.0+
out-of-sample Sharpe ratio**, constructed from **22 low-correlation
instruments** across diversified asset classes.

This project evaluates multiple portfolio construction methodologies and
compares their risk-adjusted performance across **train, validation, and
holdout periods**.

------------------------------------------------------------------------

## 🧠 Strategy Overview

### 🔹 Max Sharpe Ratio (MSR)

Mean-variance optimization that maximizes ex-ante Sharpe ratio.

### 🔹 Inverse Volatility (IV)

Risk-balanced allocation using volatility scaling.

### 🔹 Equal Weight (EW)

Naïve diversification baseline.

Each strategy reports:

-   Annualized return
-   Annualized volatility
-   Sharpe ratio
-   Portfolio weights

------------------------------------------------------------------------

## 🔬 Methodology Highlights

-   Cross-asset diversification across 22 instruments
-   Covariance estimation and volatility normalization
-   Structured Train / Validation / Holdout evaluation
-   Explicit exposure constraints

------------------------------------------------------------------------

## ⚙️ Reproducing Results (macOS)

### 1️⃣ Verify Python Installation

``` bash
python3 --version
```

### 2️⃣ Clone the Repository

``` bash
git clone https://github.com/missravenlocks/multi-asset-trading-portfolio.git
cd multi-asset-trading-portfolio
```

### 3️⃣ Create & Activate Virtual Environment

``` bash
python3 -m venv env
source env/bin/activate
```

### 4️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 Running the Strategies

``` bash
python3 msr.py
python3 iv.py
python3 ew.py
```

Each script prints performance metrics to the terminal.\
Refer to the **Sharpe** output to compare portfolio-level and
asset-level results.

------------------------------------------------------------------------

# 📊 Performance by Regime

## 🔹 Max Sharpe Ratio (MSR)

-   Train (Pre-2020): 3.1919
-   Validation (2020–2021): 3.4397
-   Holdout (2022+): 3.7560
-   Full Period: 3.2440

------------------------------------------------------------------------

## 🔹 Inverse Volatility (IV)

-   Train (Pre-2020): 3.1895
-   Validation (2020–2021): 3.3733
-   Holdout (2022+): 3.8063
-   Full Period: 3.2305

------------------------------------------------------------------------

## 🔹 Equal Weight (EW)

-   Train (Pre-2020): 3.1617
-   Validation (2020–2021): 3.3964
-   Holdout (2022+): 3.7045
-   Full Period: 3.2091

------------------------------------------------------------------------

## 📌 Research Considerations

-   Exposure determined at close and held until the following close
-   Daily exposure constrained between -1.0x and +1.5x
-   Assumes frictionless execution (no transaction costs or slippage)

------------------------------------------------------------------------
