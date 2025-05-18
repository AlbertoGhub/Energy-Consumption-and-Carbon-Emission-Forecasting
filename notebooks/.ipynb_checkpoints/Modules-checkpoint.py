# FUNTIONS:
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
  '''# TAKES A COUNTRY NAME AS INPUT TO SORT THEM INTO CONTINENT FOR EASIER ANALYSIS'''

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

def plot_trend(country_1, country_2, country_3, years, title):

  '''THIS FUNCTION WILL PLOT THE TREND TAKING 5 PARAMETERS:
    1. TAKEN THE DATA FROM 3 COUNTRIES
    2. TIME FRAME (YEARS)
    3. TITLE'''
  # COL
  plt.plot(years, country_1, color = 'red',  marker = 'o', label = 'Colombia')
  # UK
  plt.plot(years, country_2, color = 'blue', label = 'United Kingdom', marker = 'o')
  # THAI
  plt.plot(years, country_3, color = 'orange', label = 'Thailand', marker = 'o')

  plt.title(title, fontsize = 16)
  plt.grid()
  plt.xticks(years)

def bar_plot(data, col_analysis, title):
  '''THIS FUNCTION WILL PLOT THE BAR PLOT
    1. DATA FRAME
    2. COL TO ANALYSE
    3. TITLE'''
  plt.bar(data['Entity'], data[col_analysis],
        color=plt.cm.cividis(range(len(data['Entity']))))

  plt.title(title, fontsize = 16)

#defining function to calculate scores
def calculate_scores(y_true, y_pred):
  '''SCORE CALCULATORS'''
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)
  return mae, mse, rmse

# forecasting and visualization
def sktime_forecast(dataset, horizon, forecaster, validation=False, confidence=0.90, frequency="Y"):
  '''THIS FUNCTION WOULD COMPLETE THE FORCAST AND VISUALISE THE PREDICTIONS'''

  # ADJUSTING FREQUENCY AND INTERPOLATE MISSING VALUES
  forecast_df = dataset.resample(rule=frequency).sum()
  forecast_df = forecast_df.interpolate(method="time")

  for col in dataset.columns:
      if validation:
          # VALIDATION SCENARIO
          df = forecast_df[col]

          y_train = df[:-horizon]
          y_test = df.tail(horizon)

          forecaster.fit(y_train)
          fh = ForecastingHorizon(y_test.index, is_relative=False)
          y_pred = forecaster.predict(fh)
          ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
          y_true = df.tail(horizon)

          mae, mse, rmse = calculate_scores(y_true, y_pred)
      else:
          # NON-VALIDATION SCENARIO
          df = forecast_df[col].dropna()
          forecaster.fit(df)

          last_date = df.index.max()
          fh = ForecastingHorizon(
              pd.date_range(str(last_date), periods=horizon, freq=frequency),
              is_relative=False,
          )

          y_pred = forecaster.predict(fh)
          ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")

          if not np.isnan(y_pred).any():
              y_true = df.tail(horizon)
              mae, mse, rmse = calculate_scores(y_true, y_pred)
          else:
              mae, mse, rmse = np.nan, np.nan, np.nan

      # PLOT AND DISPLAY RESULTS
      plt.plot(
          df.tail(horizon * 3),
          label="Actual",
          color="black",
      )
      plt.gca().fill_between(
          ci.index, (ci.iloc[:, 0]), (ci.iloc[:, 1]), color="b", alpha=0.1
      )
      plt.plot(y_pred, label="Predicted")
      plt.title( f"{horizon} years forecast for {col}, confidence: {confidence*100}%)")
      plt.ylim(bottom=0)
      plt.legend()
      plt.grid(True)
      plt.show()

      # DISPLAY SCORES
      print(f"Column Name: {col}")
      print(f"Actual Values: {y_true.to_numpy()}")
      print(f"Predicted Values: {y_pred.to_numpy()}")
      print(f"Confidence Interval: {ci.to_numpy()}")
      print(f"Mean Absolute Error (MAE): {mae}")
      print(f"Mean Squared Error (MSE): {mse}")
      print(f"Root Mean Squared Error (RMSE): {rmse}")
      print(f"Confidence Level: {confidence}\n")