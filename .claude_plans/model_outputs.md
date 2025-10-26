Predictions - model.predict()

Index by date for time series data

Outputs - observation level results
	- date
	- actuals
	- fitted
	- forecast (stacked actuals + fitted)
	- Residual
	- Split (whether observation is in train/test/forecast)
Model Metadata columns:
	- Model
	- Model_group_name
	- group (for use in grouped data, global, panel models)

Coefficients - variable/term level results - coefficients for statistical models, hyperparameters and feature importances for machine learning models
	- Variable
	- Coefficient
	- Std_error
	- P_value
	- VIF
	- T stat
	- [0.025
	- 0.0975]
Model Metadata columns:
	- Model
	- Model_group_name
	- group (for use in grouped data, global, panel models)

Stats - model level metrics grouped and aggregated by split - train/test/forecast
General model information:
	- Formula/model specification
	- Model type
	- Model parameters
	- Model hyperparameters
	- Train/test/forecast start and end dates
	- List of exogenous variables
Model performance metrics by split:
	- Rmse
	- Mape
	- Smape
	- Mae
	- R squared
	- Adjusted r squared
	- Mda (mean dirctional average)
Residual statistics
	- Breusch-pagan
	- Durbin-watson
	- Ljung-box
	- Shapiro-wilk
Model Metadata columns:
	- Model
	- Model_group_name
	- group (for use in grouped data, global, panel models)

Resample outputs, stats and coefficients
	- Same as above but additional column called "slice" - stores results for all cross validation/resample slice results
