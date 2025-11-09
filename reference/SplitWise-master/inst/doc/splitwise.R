## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----load-package, include = FALSE--------------------------------------------
library(SplitWise)

## ----mtcars-------------------------------------------------------------------
# Load the mtcars dataset
data(mtcars)

## ----iterative-transformation-------------------------------------------------
# Apply iterative transformations with forward stepwise selection
model_iter <- splitwise(
  mpg ~ .,
  data = mtcars,
  transformation_mode = "iterative",
  direction = "backward",
  verbose = FALSE
)

# Display the summary of the model
summary(model_iter)

# Print the model details
print(model_iter)

## ----univariate-transformation------------------------------------------------
# Apply univariate transformations with backward stepwise selection
model_uni <- splitwise(
  mpg ~ .,
  data = mtcars,
  transformation_mode = "univariate",
  direction = "backward",
  verbose = FALSE
)

# Display the summary of the model
summary(model_uni)

# Print the model details
print(model_uni)

