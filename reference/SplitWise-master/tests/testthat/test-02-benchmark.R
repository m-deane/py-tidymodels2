test_that("Splitwise vs step() function with different transformation modes", {

  set.seed(123)
  data(mtcars)

  directions <- c("backward", "forward", "both")
  transformation_modes <- c("univariate", "iterative")

  results <- data.frame(Direction = character(),
                        Transformation_Mode = character(),
                        Splitwise_AIC = numeric(),
                        Step_AIC = numeric())

  for (dir in directions) {
    initial_model <- lm(mpg ~ ., data = mtcars)
    step_model <- step(initial_model, direction = dir, trace = 0)
    step_aic <- AIC(step_model)

    for (mode in transformation_modes) {
      splitwise_model <- splitwise(mpg ~ ., data = mtcars,
                                   transformation_mode = mode,
                                   direction = dir, verbose = FALSE)
      splitwise_aic <- AIC(splitwise_model)

      results <- rbind(results, data.frame(Direction = dir,
                                           Transformation_Mode = mode,
                                           Splitwise_AIC = splitwise_aic,
                                           Step_AIC = step_aic))
    }
  }

  # Calculate mean AICs
  mean_splitwise_aic <- mean(results$Splitwise_AIC)
  mean_step_aic <- mean(results$Step_AIC)

  # Final test: mean Splitwise AIC should be <= mean Step AIC
  expect_lte(mean_splitwise_aic, mean_step_aic,
             label = "Mean AIC: Splitwise vs Stepwise")

  # Format and print the comparison table
  cat("\n\n")
  cat(format("Direction", width = 12, justify = "left"),
      format("Transformation Mode", width = 20, justify = "left"),
      format("SplitWise AIC", width = 15, justify = "right"),
      format("Step AIC", width = 10, justify = "right"), "\n")
  cat(rep("-", 60), sep = "", "\n")
  for (i in 1:nrow(results)) {
    cat(format(results$Direction[i], width = 12, justify = "left"),
        format(results$Transformation_Mode[i], width = 20, justify = "left"),
        format(sprintf("%.4f", results$Splitwise_AIC[i]), width = 15, justify = "right"),
        format(sprintf("%.4f", results$Step_AIC[i]), width = 10, justify = "right"), "\n")
  }
  cat("\n")
})
