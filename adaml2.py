# BM20A6100 Project work Forecasting power for predicting temperature using multivariate data
# Emma Hirvonen, Helmi Toropainen, Jan Vymazal

# Import dataset with Pandas
import pandas as pd
file_path = "sample_data/DailyDelhiClimateTrain.csv"
climate_data = pd.read_csv(file_path, parse_dates=["date"])
climate_data = climate_data.set_index("date") # use date as index
print(climate_data.info())
print(climate_data.head())
print(climate_data.describe())
# Data consists of 4 variables for 1462 samples, no missing values

# Visualise timerseries for all variables
import matplotlib.pyplot as plt

plt.figure()
climate_data.plot()
plt.title("Raw data")
plt.show()

plt.figure()
climate_data.drop("meanpressure", axis=1).plot()
plt.title("Pressure values excluded")
plt.show()

for var in climate_data.columns:
  plt.figure()
  climate_data[var].plot()
  plt.ylabel(var)
  plt.show()

# remove outlier
# note: not sure if allowed to do yet, wanted nicer plot
print(climate_data[climate_data["meanpressure"] > 2000]) # outlier?
climate_data_clean = climate_data.copy()

t = pd.Timestamp(climate_data_clean[climate_data_clean["meanpressure"] > 2000].index.date[0]) # outlier date
i = climate_data_clean.index.get_loc(t) # outlier index

new_pressure = (climate_data_clean.iloc[i+1].get("meanpressure")+climate_data_clean.iloc[i-1].get("meanpressure"))/2 # average of prev and next value

climate_data_clean.loc[t, "meanpressure"] = new_pressure

# new plots without outlier
plt.figure()
climate_data_clean.plot()
plt.title("One outlier removed")
plt.show()

plt.figure()
climate_data_clean["meanpressure"].plot()
plt.ylabel("meanpressure")
plt.title("One outlier removed")
plt.show()

# negative pressure?
plt.figure()
climate_data["meanpressure"].plot()
plt.ylabel("meanpressure")
plt.axhline(y=0, color='k', linestyle='--')
plt.ylim(-50, 50)
plt.title("Extreme values")
plt.show()

# pressure values from 2013-01-01 to 2016-01-01 (2016 seems unreliable?)
plt.figure()
climate_data.iloc[0:365+365+365, 3].plot()
plt.ylabel("meanpressure")
plt.title("2013-2015, 2016 excluded")
plt.show()
# all values should be between ~990-1030?

# skip outlier values
climate_data.drop("meanpressure", axis=1).plot()
climate_data.query("990 < meanpressure < 1030")["meanpressure"].plot()
plt.legend()
plt.title("Extreme pressure values excluded")
plt.show()
# todo: better outlier handling?

# broken y-axis (matplotlib documentation)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.05)
plt_data = climate_data#.query("990 < meanpressure < 1030")
ax1.plot(plt_data)
ax1.set_ylim((980, 1050))
ax2.plot(plt_data)
ax2.set_ylim((-50, 140))
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()
ax1.legend(plt_data.columns, bbox_to_anchor=(1.01, 1.0), loc='upper left')
ax1.title.set_text("Patterns in multivariate data")
fig.autofmt_xdate()
d = .5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
plt.show()

# mean center data
plt.figure()
plt.plot(climate_data.query("990 < meanpressure < 1030") - climate_data.query("990 < meanpressure < 1030").mean())
plt.gcf().autofmt_xdate()
plt.legend(climate_data.columns)
plt.title("Samples with extreme pressure excluded, mean-centered data")
plt.show()

# subplot of all variables (no mean-centering)
plt.figure(figsize=(9, 9))
for i in range(4):
  plt.subplot(2,2,i+1)
  climate_data.query("990 < meanpressure < 1030").iloc[:, i].plot()
  plt.ylabel(climate_data.columns[i])
plt.show()

#Time-series decomposition analysis in long-term trend, seasonality and residuals
#Period 365 because we measure once per day for multiple years
from statsmodels.tsa.seasonal import STL
stl = STL(climate_data["meantemp"], period=365 )
res = stl.fit()
fig = res.plot()

stl = STL(climate_data["humidity"], period=365)
res = stl.fit()
fig = res.plot()

stl = STL(climate_data["wind_speed"], period=365)
res = stl.fit()
fig = res.plot()

stl = STL(climate_data["meanpressure"], period=365)
res = stl.fit()
fig = res.plot()

#Autocorrelation analysis of the dataset
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
plot_acf(climate_data["meantemp"])
plt.title("Mean temperature autocorrelation")

plt.figure()
plot_acf(climate_data["humidity"])
plt.title("Humidity autocorrelation")

plt.figure()
plot_acf(climate_data["wind_speed"])
plt.title("Wind speed autocorrelation")

plt.figure()
plot_acf(climate_data["meanpressure"])
plt.title("Mean pressure autocorrelation")