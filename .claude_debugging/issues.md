Bugs

From examples in "/_md/forecasting.ipynb"

1. the stats dataframe that is returned by extract_outputs() always returns Nan values for "ljung_box_stat" and "ljung_box_p" "breesuch_pagan_stat" and "bresuch_pagan_p" are also NaN in the majority of cases including examples with exogenous regressors but I would expect for univariate time series models that these would be NaN values.

2. not all models return train_start_date and train_end_date in the stats dataframe returned by extract_outputs() - i.e. linear_reg().set_engine("statsmodels")

3. is there a way or an altnerative auto_arima engine that can be used for arima_reg and arima_boost that does not have the numpy compatibility issue? look at implementations in sktime, skforecast or statsforecast - these can be found in the "reference" folder

4. the coefficients dataframe returned from extract_outputs() for the gen_additive_mod() is not compatible with other model types

5. null_model() does not support argument strategy - i would expect it to support strategy "mean" (take mean value) - "median" - take "median" value and "last" - take "last value

6. naive_reg does not support argument strategy - i would expect it to support naive, seasonal_naive, window, drift like the sktime NaiveForecaster implementation

Enhancements

7. create a model type - "hybrid" that can support taking 2 models, i.e. 2 linear_reg models that are trained on different periods and either the fitted values from the 1st, or the residuals from the first and returns in extract_outputs the total fitted values of both models

8. also create a model "type" which is really a manual model type where I can set the coefficients for each exogenous varaible manually, treat this as a model and calculate the same extract_outputs() dataset as other models - this is hadny for using when I have pre existing forecasts or models that I want to compare with new models i may have generated using py-tidymodels

To Check

8. Is there a helper function which will produce an interactive time series plot of the model outputs, e.g. take outputs_arima, outputs_prophet etc and create a plot with the date on the x axis, the actuals on the y axis in dark blue, the fitted values on the y axis with the train split in orange and the test split in red and the ability to plot multiple models by the "model" column in the outputs table. Ability to support multiple groups using the 'group' column in the same table. Optional to also include residuals plotted on the y axis

