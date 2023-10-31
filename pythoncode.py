import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#ReadMe for more info

# Load the CSV data from the path using Pandas
df = pd.read_csv('TSLA.csv')

# Date in df is stored as a string, so we need to convert it to DateTime input
df['Date'] = pd.to_datetime(df['Date'])

# Input is Date => X = Date
# Output is price => Y = Close Price
# Linear regression works with numerical features, so we need to convert DateTime to timestamps

X = np.arange(1, len(df) + 1).reshape(-1, 1)
Y = df['Close'].values.reshape(-1, 1)

# Create a linear regression model
linear_regressor = LinearRegression()

# Perform linear regression
linear_regressor.fit(X, Y)

# Make predictions
Y_pred = linear_regressor.predict(X)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(Y, Y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y, Y_pred)

# Calculate R-squared (R²) value
r2 = r2_score(Y, Y_pred)

# Create the scatterplot
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('Date (Days after Initial Date)')
plt.ylabel('Close Price')

# Print the MAE, MSE, and R² values
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Value: {r2:.2f}")

# Show the plot
plt.show()