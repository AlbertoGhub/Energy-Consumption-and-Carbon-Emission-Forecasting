# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Modelling and forecasting
from sktime.forecasting.base import ForecastingHorizon
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sktime.performance_metrics.forecasting import mean_squared_error
from sktime.forecasting.arima import AutoARIMA

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite' in 1.6.*",
    category=FutureWarning,
    module="sklearn.*"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Missing values calculator
def NaNs_calculator(data):

    '''SHOWING THE STATS OF MISSING DATA AND DATA TYPE'''

    percentage_missing = (data.isna().mean()*100).sort_values(ascending = False)                      # Storing the Percentages of NaNs
    sum_missing = data.isna().sum().sort_values(ascending = False)                                    # Storing the Sum of NaNs
    names = sum_missing.index.to_list()                                                               # Storing names (to show in the columns)
    data_type = data[names].dtypes                                                                    # Storing the type of data based on the order from the previous obtained data (slicing)
    sum_values = sum_missing.to_list()                                                                # Getting count of missing values
    perc_values = np.around(percentage_missing.to_list(), 3)                                          # Getting percentage of missing values
    types = data_type.to_list()                                                                       # Getting the types of the data
    # TURN ALL THE DATA INTO A DATAFRAME
    df_missing = pd.DataFrame({"NAMES" : names,
                                    "VALUE COUNT" : sum_values,
                                    "PERCENTAGE (%)" : perc_values,
                                    "DATA TYPE": types})
    return df_missing

def country_to_continent(country_name):
  '''TAKES A COUNTRY NAME AS INPUT TO SORT THEM INTO CONTINENT FOR EASIER ANALYSIS'''

  # CONVERT THE COUNTRY NAME TO ISO 3166-1 ALPHA-2 COUNTRY CODE (ABBREVIATION, E.G. COLOMBIA = CO)
  country_alpha2 = pc.country_name_to_country_alpha2(country_name)

  # USE THE COUNTRY CODE TO GET THE CONTINENT CODE (E.G., 'NA' FOR NORTH AMERICA)
  country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)

  # CONVERT THE CONTINENT CODE TO THE FULL CONTINENT NAME (E.G., 'NORTH AMERICA')
  country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)

  return country_continent_name

def filter_and_calculate_mean(data, year, column_name, cols):
  '''CREATING A FUNCTION TO AUTOMATICALLY CALCULATE THE AVERAGE VALUE FOR THE CONTINENT FOR THE YEAR.
  TAKING 4 ARGUMENTS:
  1. FULL DATA FRAME
  2. TIME FRAME
  3. COLUMN NAME (CO2 OR ENERGY)
  4. COLUMS TO HANDLE'''
  row_data = {'Year': year}
  for continent in cols:
      filtered_df = data[(data['Year'] == year) & (data['Continent'] == continent)]
      mean_value = filtered_df[column_name].mean()
      row_data[continent] = mean_value
  return row_data

# def plot_trend(country_1, country_2, country_3, years, title):

#   '''THIS FUNCTION WILL PLOT THE TREND TAKING 5 PARAMETERS:
#     1. TAKEN THE DATA FROM 3 COUNTRIES
#     2. TIME FRAME (YEARS)
#     3. TITLE'''
#   # COL
#   plt.plot(years, country_1, color = 'red',  marker = 'o', label = 'Colombia')
#   # UK
#   plt.plot(years, country_2, color = 'blue', label = 'United Kingdom', marker = 'o')
#   # THAI
#   plt.plot(years, country_3, color = 'orange', label = 'Thailand', marker = 'o')

#   plt.title(title, fontsize = 16)
#   plt.grid()
#   plt.xticks(years)

def plot_trend(country_1, country_2, country_3, years, title,
               country_labels=('Colombia', 'United Kingdom', 'Thailand')):

    """
    Plot the trend of data from three countries over a given set of years.

    Parameters:
    - country_1, country_2, country_3: Lists or arrays with values per year.
    - years: List of years (x-axis).
    - title: Title of the plot.
    - country_labels: Tuple of three country names (for legend).
    """

    sns.set_theme(style="whitegrid", palette="pastel")

    # Plot each country's data
    plt.plot(years, country_1, marker='o', label=country_labels[0], linewidth=2)
    plt.plot(years, country_2, marker='o', label=country_labels[1], linewidth=2)
    plt.plot(years, country_3, marker='o', label=country_labels[2], linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(years, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

def bar_plot(data, col_analysis, title):
    """
    Plot a bar chart using Seaborn with pastel colours.

    Parameters:
    - data: DataFrame containing the data
    - col_analysis: Column name (string) to be analysed
    - title: Chart title (string)
    """
    # Set style and figure size
    sns.set_theme(style='whitegrid')

    # Sort data for a cleaner visual
    data_sorted = data.sort_values(by=col_analysis, ascending=False)

    # Create the bar plot using hue to allow colour palette
    ax = sns.barplot(
        x='Entity',
        y=col_analysis,
        hue='Entity',  # Use hue to apply pastel palette
        data=data_sorted,
        palette='pastel',
        edgecolor='black',
        dodge=False,
        legend=False
    )

    # Titles and labels
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel(col_analysis, fontsize=12)

    # Rotate tick labels safely
    ax.tick_params(axis='x', rotation=45)

    # Format y-axis
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=9, padding=3)
    
    plt.grid(True)
    plt.tight_layout()
        
    


#defining function to calculate scores
def calculate_scores(y_true, y_pred):
  '''SCORE CALCULATORS'''
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)
  return mae, mse, rmse

# # forecasting and visualisation
# def sktime_forecast(dataset, horizon, forecaster, validation=False, confidence=0.90, frequency="YE"):
#   '''THIS FUNCTION WOULD COMPLETE THE FORCAST AND VISUALISE THE PREDICTIONS'''

#   # ADJUSTING FREQUENCY AND INTERPOLATE MISSING VALUES
#   forecast_df = dataset.resample(rule=frequency).sum()
#   forecast_df = forecast_df.interpolate(method="time")

#   for col in dataset.columns:
#       if validation:
#           # VALIDATION SCENARIO
#           df = forecast_df[col]

#           y_train = df[:-horizon]
#           y_test = df.tail(horizon)

#           forecaster.fit(y_train)
#           fh = ForecastingHorizon(y_test.index, is_relative=False)
#           y_pred = forecaster.predict(fh)
#           ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
#           y_true = df.tail(horizon)

#           mae, mse, rmse = calculate_scores(y_true, y_pred)
#       else:
#           # NON-VALIDATION SCENARIO
#           df = forecast_df[col].dropna()
#           forecaster.fit(df)

#           last_date = df.index.max()
#           fh = ForecastingHorizon(
#               pd.date_range(str(last_date), periods=horizon, freq=frequency),
#               is_relative=False,
#           )

#           y_pred = forecaster.predict(fh)
#           ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")

#           if not np.isnan(y_pred).any():
#               y_true = df.tail(horizon)
#               mae, mse, rmse = calculate_scores(y_true, y_pred)
#           else:
#               mae, mse, rmse = np.nan, np.nan, np.nan

#       # PLOT AND DISPLAY RESULTS
#       plt.plot(
#           df.tail(horizon * 3),
#           label="Actual",
#           color="black",
#       )
#       plt.gca().fill_between(
#           ci.index, (ci.iloc[:, 0]), (ci.iloc[:, 1]), color="b", alpha=0.1
#       )
#       plt.plot(y_pred, label="Predicted")
#       plt.title( f"{horizon} years forecast for {col}, confidence: {confidence*100}%)")
#       plt.ylim(bottom=0)
#       plt.legend()
#       plt.grid(True)
#       plt.show()

#       # DISPLAY SCORES
#       print(f"Column Name: {col}")
#       print(f"Actual Values: {y_true.to_numpy()}")
#       print(f"Predicted Values: {y_pred.to_numpy()}")
#       print(f"Confidence Interval: {ci.to_numpy()}")
#       print(f"Mean Absolute Error (MAE): {mae}")
#       print(f"Mean Squared Error (MSE): {mse}")
#       print(f"Root Mean Squared Error (RMSE): {rmse}")
#       print(f"Confidence Level: {confidence}\n")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sktime.forecasting.base import ForecastingHorizon

# Use pastel Seaborn style
sns.set(style="whitegrid", palette="pastel")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sktime.forecasting.base import ForecastingHorizon
from Modules import calculate_scores  # assuming this is defined elsewhere

# Suppress specific sklearn FutureWarnings
warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite' in 1.6.*",
    category=FutureWarning
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sns.set(style="whitegrid", palette="pastel")

def sktime_forecast(dataset, horizon, forecaster, validation=False, confidence=0.90, frequency="YE", title = None):
    """
    Forecasts time series using sktime forecaster and visualises predictions.
    
    Parameters:
        dataset (pd.DataFrame): Time series dataset with datetime index.
        horizon (int): Number of periods to forecast.
        forecaster: A sktime-compatible forecasting model.
        validation (bool): Whether to perform backtesting.
        confidence (float): Confidence level for prediction intervals.
        frequency (str): Resampling frequency, e.g., "YE" for year-end.
    """

    # Resample and interpolate missing values
    forecast_df = dataset.resample(rule=frequency).sum().interpolate(method="time")

    for col in dataset.columns:
        df = forecast_df[col]

        if validation:
            y_train = df[:-horizon]
            y_test = df.tail(horizon)

            forecaster.fit(y_train)
            fh = ForecastingHorizon(y_test.index, is_relative=False)
            y_pred = forecaster.predict(fh)
            ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
            y_true = y_test
        else:
            df = df.dropna()
            forecaster.fit(df)
            last_date = df.index.max()
            fh = ForecastingHorizon(
                pd.date_range(str(last_date), periods=horizon, freq=frequency),
                is_relative=False
            )
            y_pred = forecaster.predict(fh)
            ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
            y_true = df.tail(horizon) if not np.isnan(y_pred).any() else pd.Series(index=fh.to_pandas(), dtype=float)

        # Calculate scores
        mae, mse, rmse = calculate_scores(y_true, y_pred) if not np.isnan(y_pred).any() else (np.nan, np.nan, np.nan)

        # Plotting
        actual_window = df.tail(horizon * 3)
        combined = pd.concat([actual_window, y_pred], axis=0)

        plt.figure(figsize=(10, 5))
        sns.lineplot(data=actual_window, label="Actual", color="black")
        sns.lineplot(data=y_pred, label="Predicted", color="coral")
        plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="skyblue", alpha=0.3, label="Confidence Interval")
        plt.axvline(y_pred.index.min(), color="grey", linestyle="--", alpha=0.7, label="Forecast Start")
        plt.title(f"{horizon}-Period Forecast for {title} in '{col}' (Confidence: {int(confidence*100)}%)")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'../images/{title}_predictions_{col}.png')

        # Organised output as DataFrame
        results = pd.DataFrame({
            "Actual": np.around(y_true.to_numpy(), 2),
            "Predicted": np.around(y_pred.to_numpy(), 2),
            "Lower CI": np.around(ci.iloc[:, 0].to_numpy(), 2),
            "Upper CI": np.around(ci.iloc[:, 1].to_numpy(), 2)
        }, index=y_pred.index)

        scores = pd.Series({
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "Confidence Level": f"{int(confidence*100)}%"
        })

        print(f"📊 Results for column: {col}")
        print(results)
        print("\n📈 Performance Scores:")
        print(scores)
        print("\n" + "-"*60 + "\n")

