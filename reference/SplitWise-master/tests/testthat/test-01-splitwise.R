test_that("splitwise function works correctly with default parameters", {
  # Example dataset
  data <- mtcars
  formula <- mpg ~ .

  # Run the splitwise function
  model <- splitwise(formula, data)

  # Check that the model is of the correct class
  expect_s3_class(model, "splitwise_lm")

  # Check that the model has coefficients
  expect_true(length(coef(model)) > 0)
})

test_that("splitwise function handles different transformation modes", {
  data <- mtcars
  formula <- mpg ~ .

  # Test univariate transformation mode
  model_uni <- splitwise(formula, data, transformation_mode = "univariate",
                         verbose = FALSE)
  expect_s3_class(model_uni, "splitwise_lm")

  # Test iterative transformation mode
  model_iter <- splitwise(formula, data, transformation_mode = "iterative",
                         verbose = FALSE)
  expect_s3_class(model_iter, "splitwise_lm")
})

test_that("splitwise function respects stepwise direction parameter", {
  data <- mtcars
  formula <- mpg ~ .

  # Test forward selection
  model_forward <- splitwise(formula, data, direction = "forward",
                             verbose = FALSE)
  expect_s3_class(model_forward, "splitwise_lm")

  # Test backward elimination
  model_backward <- splitwise(formula, data, direction = "backward",
                             verbose = FALSE)
  expect_s3_class(model_backward, "splitwise_lm")

  # Test both directions
  model_both <- splitwise(formula, data, direction = "both",
                          verbose = FALSE)
  expect_s3_class(model_both, "splitwise_lm")
})

test_that("splitwise function handles BIC mode", {
  data <- mtcars
  formula <- mpg ~ .

  # BIC typically uses k = log(n), but let's do the default step(k=2) for simplicity
  # or just to confirm it doesn't error out.
  model_bic <- splitwise(
    formula = formula,
    data = data,
    criterion = "BIC",
    verbose = FALSE
  )
  expect_s3_class(model_bic, "splitwise_lm")

  # Check decisions are stored
  expect_true("decisions" %in% names(model_bic$splitwise_info))

  # Optionally, check if BIC was used by verifying the final model's summary
  # or the code. But for now, just confirm no errors.
})

test_that("splitwise function respects exclude_vars parameter", {
  data <- mtcars
  formula <- mpg ~ .

  # Exclude some numeric columns from dummy transformations
  # e.g., "wt" and "hp"
  excluded <- c("wt", "hp")

  model_excl <- splitwise(
    formula = formula,
    data = data,
    exclude_vars = excluded,
    verbose = FALSE
  )
  expect_s3_class(model_excl, "splitwise_lm")

  # Check that the excluded variables are always 'linear' in the decisions
  dec <- model_excl$splitwise_info$decisions

  # "wt" and "hp" should be linear if they exist in the decisions list
  if ("wt" %in% names(dec)) {
    expect_equal(dec[["wt"]]$type, "linear")
  }
  if ("hp" %in% names(dec)) {
    expect_equal(dec[["hp"]]$type, "linear")
  }
})
