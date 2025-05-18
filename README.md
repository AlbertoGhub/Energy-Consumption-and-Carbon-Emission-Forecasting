# 🌍 Energy Consumption and Carbon Emissions Forecasting

## 📌 Overview

This project presents a machine learning solution to forecast **energy consumption** and **carbon emissions** for various countries, focusing on sustainable development and climate impact. By analysing historical data and applying forecasting techniques, we predict future trends and provide visual insights for better policy and decision-making.

## 🎯 Objective

To develop a reliable machine learning pipeline capable of:

1. **Predicting future energy consumption** (kWh/person) for each country.
2. **Forecasting carbon emissions levels** (metric tons per capita) for each country.
3. **Visualising the energy and emission trends** for at least three selected countries over the next five years (2021–2025), with supporting insights drawn from the dataset.

## 🛠️ Workflow

The project follows a complete machine learning workflow:

- **Data acquisition and exploration**
- **Data cleaning and preprocessing**
- **Feature engineering and selection**
- **Model development and validation**
- **Forecasting with confidence intervals**
- **Visualisation and analysis**
- **Result interpretation and critical justification**

## 📊 Key Features in the Dataset

| Feature | Description |
|--------|-------------|
| `Entity` | Name of the country or region |
| `Year` | Time range (2000–2020) |
| `Access to electricity` | % of population with electricity access |
| `Access to clean fuels for cooking` | % of population using clean cooking fuels |
| `Renewable-electricity-generating-capacity-per-capita` | Installed renewable energy capacity per person |
| `Financial flows to developing countries` | US$ received for clean energy projects |
| `Renewable energy share in final consumption` | % of renewables in final energy use |
| `Electricity from fossil fuels` | Energy generated from coal, oil, and gas (TWh) |
| `Electricity from nuclear` | Nuclear energy generation (TWh) |
| `Electricity from renewables` | Renewable electricity generation (TWh) |
| `Low-carbon electricity` | % of electricity from low-carbon sources |
| `Primary energy consumption per capita` | ⚠️ Target variable |
| `Energy intensity level` | Energy use per unit of GDP |
| `Value_co2_emissions` | ⚠️ Target variable – CO₂ emissions per person |
| `Renewables (% equivalent energy)` | Share of renewables in total equivalent energy |
| `GDP growth` | Annual economic growth rate |
| `GDP per capita` | Economic output per person |
| `Density`, `Land Area`, `Latitude`, `Longitude` | Geographic and demographic indicators |

## 📈 Visualisation & Results

The project includes interactive visualisations generated with **Seaborn** and **Matplotlib**, featuring:

- Past and predicted values for both energy and emissions
- Confidence intervals (90%) for forecasts
- Comparative analysis of countries (e.g., Colombia, Thailand, the UK)
- Graphs are automatically saved to the `/images/` folder for reporting

## 💡 Methodology

- **Libraries**: `pandas`, `numpy`, `sktime`, `scikit-learn`, `matplotlib`, `seaborn`
- **Forecasting Models**: Time series forecasters from `sktime` (e.g., ARIMA, Exponential Smoothing)
- **Evaluation Metrics**: MAE, MSE, RMSE
- **Confidence Intervals**: Integrated into predictions for uncertainty estimation

## 📁 Project Structure

```bash
├── data/
│   └── global-data-on-sustainable-energy.csv
├── notebooks/
│   └── main.ipynb
├── images/
├── src/
│   └── modules
│       └── Modules.py
├── README.md
└── requirements.yml
```

# 🚀 How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt` or `conda install --file requirements.yml`
## 👌 Explanation

```pip```: For standard Python virtual environments (requirements.txt)

```conda```: If you're using Anaconda/Miniconda and prefer to work with .yml environment files (requirements.yml)

## Note
To install ```sktime```, you need to have python 3.10 - 3.12 version for compatibility


## 📋 Results Summary

All forecast results are summarised in structured DataFrames with key metrics, including:

- Rounded predictions and actual values
- Lower and upper confidence intervals
- MAE, MSE, RMSE

This allows quick comparison and easier integration into reports.

## 🔎 Result Interpretation

The forecasting model was evaluated across three countries — **Colombia**, **Thailand**, and the **United Kingdom** — using historical data up to 2020 and projecting energy consumption for five subsequent years (2021–2025). Below is a summary of the results and critical interpretation.

---

### 🇨🇴 Colombia

**Forecast Trends:**

- The model predicts a steady increase in energy consumption post-2020, peaking in 2023.
- Despite a drop in actual consumption in 2021, predictions indicate recovery and gradual growth.
- Forecasts remain within a reasonably narrow confidence interval, suggesting stability.

**Performance:**

- **MAE**: ~4,029 kWh/person
- **RMSE**: ~5,116 kWh/person
- Indicates a good level of accuracy relative to the scale of consumption.

**Interpretation:**

The forecasted rebound after 2021 likely reflects recovery from pandemic-related disruptions. With increasing renewable capacity and development aid, Colombia appears positioned for energy growth aligned with sustainability.

---

### 🇹🇭 Thailand

**Forecast Trends:**

- Predicted energy consumption shows consistent growth from 2020 through 2023.
- Confidence intervals remain tight, signalling model certainty.
- Forecasts slightly overestimate consumption compared to recent actuals.

**Performance:**

- **MAE**: ~15,870 kWh/person
- **RMSE**: ~18,109 kWh/person

**Interpretation:**

Thailand’s rising consumption trend is consistent with ongoing industrialisation and economic development. Higher error margins reflect volatility in recent years, potentially due to external shocks or infrastructural transitions.

---

### 🇬🇧 United Kingdom

**Forecast Trends:**

- A steady decline in predicted energy consumption is observed from 2020 to 2023.
- Forecasts reflect ongoing national efforts to improve energy efficiency and reduce dependency on fossil fuels.
- Wider confidence intervals from 2021 onwards indicate greater uncertainty.

**Performance:**

- **MAE**: ~41,496 kWh/person
- **RMSE**: ~41,737 kWh/person

**Interpretation:**

The UK’s downward trend aligns with climate policy shifts and deindustrialisation. However, the larger forecasting errors suggest more complex dynamics at play, including rapid decarbonisation efforts and energy market reforms.

---

**General Observations:**

- All forecasts maintain a 90% confidence level, providing a balanced view of potential variance.
- The model performs better in countries with less fluctuation in historical data (e.g., Colombia).
- Predictions should be interpreted in conjunction with socio-economic factors such as GDP growth, clean energy investment, and policy changes reflected in the dataset.







## 📌 Notes

- Warnings related to deprecated parameters (e.g., `force_all_finite`) are gracefully suppressed to ensure clean output.
- Graphs are saved automatically using the country name as the filename.

## 🔍 Future Improvements

- Incorporate more recent datasets (post-2020)
- Use ensemble forecasting models
- Deploy as a web app for real-time querying (Streamlit or Dash)

## 👨‍💻 Author

Developed as part of a Machine Learning Engineering assignment.

---

*For further information or technical documentation, please refer to the source notebooks or scripts in the repository.*
