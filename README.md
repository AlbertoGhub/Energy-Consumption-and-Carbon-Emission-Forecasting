# 🌍 Energy Consumption and Carbon Emissions Forecasting

---

## 📌 Overview

This project presents a machine learning solution to forecast **energy consumption** and **carbon emissions** for various countries, focusing on sustainable development and climate impact. By analysing historical data and applying forecasting techniques, we predict future trends and provide visual insights for better policy and decision-making.

---

## 🎯 Objective

To develop a reliable machine learning pipeline capable of:

1. **Predicting future energy consumption** (kWh/person) for each country.
2. **Forecasting carbon emissions levels** (metric tons per capita) for each country.
3. **Visualising the energy and emission trends** for at least three selected countries over the next five years (2021–2025), with supporting insights drawn from the dataset.

---

## 🛠️ Workflow

The project follows a complete machine learning workflow:

- **Data acquisition and exploration**
- **Data cleaning and preprocessing**
- **Feature engineering and selection**
- **Model development and validation**
- **Forecasting with confidence intervals**
- **Visualisation and analysis**
- **Result interpretation and critical justification**

---

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

---

## 📈 Visualisation & Results

The project includes interactive visualisations generated with **Seaborn** and **Matplotlib**, featuring:

- Past and predicted values for both energy and emissions
- Confidence intervals (90%) for forecasts
- Comparative analysis of countries (e.g., Colombia, Thailand, the UK)
- Graphs are automatically saved to the `/images/` folder for reporting

---

## 💡 Methodology

- **Libraries**: `pandas`, `numpy`, `sktime`, `scikit-learn`, `matplotlib`, `seaborn`
- **Forecasting Models**: Time series forecasters from `sktime` (e.g., ARIMA, Exponential Smoothing)
- **Evaluation Metrics**: MAE, MSE, RMSE
- **Confidence Intervals**: Integrated into predictions for uncertainty estimation

---

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

---

# 🚀 How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt` or `conda install --file requirements.yml`

## 👌 Explanation

```pip```: For standard Python virtual environments (requirements.txt)

```conda```: If you're using Anaconda/Miniconda and prefer to work with .yml environment files (requirements.yml)

## Note
To install ```sktime```, you need to have python 3.10 - 3.12 version for compatibility

---

## 📋 Results Summary

All forecast results are summarised in structured DataFrames with key metrics, including:

- Rounded predictions and actual values
- Lower and upper confidence intervals
- MAE, MSE, RMSE

This allows quick comparison and easier integration into reports.

---

## 📋 Result Interpretation by Country

---

### 🇨🇴 Colombia

- **Carbon Emissions:**

![Image](https://github.com/user-attachments/assets/d5bf0ba3-d040-48a6-95fc-5a4a788aac35)

  - Predicted emissions closely track actual values with minor fluctuations.
  - Confidence intervals are reasonably narrow, indicating moderate certainty.
  - MAE (~4,029) and RMSE (~5,116) show good predictive accuracy relative to emission scale.
  - Forecast suggests a fairly stable carbon emission trend with minor variations.

- **Energy Consumption:**

![Image](https://github.com/user-attachments/assets/2f4fec09-f8f2-407f-916f-7822012fab94)

  - Predictions slightly overestimate actual values but remain within acceptable error margins.
  - Confidence intervals are narrow, indicating reliable model performance.
  - Energy demand appears stable over the forecast horizon.

- **Summary:**
  - Both carbon emissions and energy consumption forecasts reflect stability.
  - Model performance is strong with relatively low errors.
  - Confidence intervals support the reliability of predictions.

---

### 🇹🇭 Thailand

- **Carbon Emissions:**

![Image](https://github.com/user-attachments/assets/618ab88c-09f3-4b51-b59b-83fdf2200e66)

  - Predictions show a consistent upward trend, slightly exceeding recent actual values.
  - Tight confidence intervals suggest reliable forecasts.
  - Higher MAE (~15,870) and RMSE (~18,109) reflect greater variability in emissions.
  - Variability may relate to economic or energy policy changes.

- **Energy Consumption:**

![Image](https://github.com/user-attachments/assets/3a4f8c64-64eb-4220-a436-0ad1a46afcfa)

  - Forecasts exhibit a steady increase aligned with economic growth.
  - Moderate error metrics (MAE ~1,539, RMSE ~1,554) indicate good model fit.
  - Confidence intervals support forecast robustness.

- **Summary:**
  - Both carbon emissions and energy consumption are trending upward.
  - Model captures variability but shows higher errors than Colombia.
  - Results highlight dynamic changes likely influenced by development and policy factors.

---

### 🇬🇧 United Kingdom

- **Carbon Emissions:**

![Image](https://github.com/user-attachments/assets/c8405cdc-37ac-4441-b1f1-5a07be11901e)

  - Forecasts indicate a downward trend reflecting emission reduction policies.
  - Wider confidence intervals and higher errors (MAE ~41,496, RMSE ~41,737) suggest complexity.
  - Model captures overall decreasing trend despite socio-economic influences.

- **Energy Consumption:**

![Image](https://github.com/user-attachments/assets/a2baf6fb-5545-4fcc-a4c9-84046b8d6fd7)

  - Modest decline in consumption with predictions closely matching actual figures.
  - Reasonable errors (MAE ~2,355, RMSE ~2,386) show good model fit.
  - Confidence intervals indicate dependable forecasts.

- **Summary:**
  - Both emissions and consumption are declining, consistent with policy impacts.
  - Higher prediction errors reflect complex factors affecting the UK’s energy profile.
  - Forecasts provide valuable insights for sustainable planning.

---

### 🔍 Overall Insights

- The forecasting models demonstrate strong capability in capturing the distinct energy and emission trends of each country analysed.
- **Colombia** exhibits relative stability in both carbon emissions and energy consumption, reflected by lower error metrics and narrow confidence intervals, indicating reliable predictions.
- **Thailand** shows a consistent upward trend in emissions and consumption, with higher prediction errors, suggesting greater variability likely influenced by rapid economic development and policy changes.
- **The United Kingdom** reflects a declining trajectory in carbon emissions and energy consumption, consistent with environmental policies, though wider confidence intervals and increased errors highlight underlying complexity in forecasting mature economies.
- Confidence intervals set at 90% provide useful uncertainty bounds, enhancing the trustworthiness of the forecasts.
- Higher errors in larger and more dynamic economies suggest opportunities for further model tuning and incorporation of additional explanatory variables.
- Overall, these insights support informed decision-making for sustainable energy management and carbon reduction strategies tailored to each country’s unique context.

---

## 📦 Libraries

- ```pandas```

- ```numpy```

- ```matplotlib```

- ```plotly```

- ```scikit-learn```

- ```sktime```

---

## 📌 Notes

- Warnings related to deprecated parameters (e.g., `force_all_finite`) are gracefully suppressed to ensure clean output.
- Graphs are saved automatically using the country name as the filename.

---

## 🔍 Future Improvements

- Incorporate more recent datasets (post-2020)
- Use ensemble forecasting models
- Deploy as a web app for real-time querying (Streamlit or Dash)

---

## 👨‍💻 Author

Developed with ❤️ by Alberto AJ - AI/ML Engineer as part of a Machine Learning Engineering assignment.

---

*For further information or technical documentation, please refer to the source notebooks or scripts in the repository.*
