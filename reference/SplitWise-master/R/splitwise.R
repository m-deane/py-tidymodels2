#' @title SplitWise Regression
#'
#' @description Transforms each numeric variable into either a single-split
#'   dummy or keeps it linear, then runs \code{stats::step()} for stepwise
#'   selection. The user can choose a simpler univariate transformation or an
#'   iterative approach.
#' @param formula A formula specifying the response and (initial) predictors,
#'   e.g. \code{mpg ~ .}.
#' @param data A data frame containing the variables used in \code{formula}.
#' @param transformation_mode Either \code{"iterative"} or \code{"univariate"}.
#'   Default = \code{"iterative"}.
#' @param direction Stepwise direction: \code{"backward"}, \code{"forward"}, or
#'   \code{"both"}.
#' @param min_support Minimum fraction (between 0 and 0.5) of observations
#'   needed in either group when making a dummy split. Prevents over-fragmented
#'   or tiny dummy groups. Default = \code{0.1}.
#' @param min_improvement Minimum required improvement (in AIC/BIC units) for
#'   accepting a dummy split or variable transformation. Helps guard against
#'   overfitting from marginal improvements. Default = \code{2}.
#' @param criterion Either \code{"AIC"} or \code{"BIC"}. Default = \code{"AIC"}.
#'   \strong{Note}: If you choose \code{"BIC"}, you typically want
#'   \code{k = log(nrow(data))} in stepwise.
#' @param exclude_vars A character vector naming variables that should be
#'   forced to remain linear (i.e., no dummy splits allowed).
#'   Default = \code{NULL}.
#' @param verbose Logical; if \code{TRUE}, prints debug info in transformation
#'   steps. If \code{FALSE}, the stepwise selection process is run quietly
#'   (trace = 0 in \code{step()}). Default = \code{FALSE}.
#' @param steps Maximum number of steps for \code{step()}. Default = 1000.
#' @param k Penalty multiple for the number of degrees of freedom
#'   (used by \code{step()}). E.g. 2 for AIC, \code{log(n)} for BIC.
#'   Default = 2.
#' @param ... Additional arguments passed to \code{step()} or the iterative
#'   function.
#' @return An S3 object of class \code{c("splitwise_lm", "lm")}, storing:
#'   \item{splitwise_info}{List containing transformation decisions, final
#'   data, and call.}
#' @examples
#' # Load the mtcars dataset
#' data(mtcars)
#'
#' # Univariate transformations (AIC-based, backward stepwise)
#' model_uni <- splitwise(
#'   mpg ~ .,
#'   data               = mtcars,
#'   transformation_mode = "univariate",
#'   direction           = "backward"
#' )
#' summary(model_uni)
#'
#' # Iterative approach (BIC-based, forward stepwise)
#' # Note: typically set k = log(nrow(mtcars)) for BIC in step().
#' model_iter <- splitwise(
#'   mpg ~ .,
#'   data               = mtcars,
#'   transformation_mode = "iterative",
#'   direction           = "forward",
#'   criterion           = "BIC",
#'   k                   = log(nrow(mtcars))
#' )
#' summary(model_iter)
#' @importFrom stats lm step model.frame setNames summary.lm as.formula AIC BIC
#' @export
#'
splitwise <- function(
    formula,
    data,
    transformation_mode = c("iterative", "univariate"),
    direction           = c("backward", "forward", "both"),
    min_support         = 0.1,
    min_improvement     = 3,
    criterion           = c("AIC", "BIC"),
    exclude_vars        = NULL,
    verbose             = FALSE,
    steps               = 1000,
    k                   = 2,
    ...
) {

  transformation_mode <- match.arg(transformation_mode)
  direction           <- match.arg(direction)
  criterion           <- match.arg(criterion)

  validate_splitwise_args(
    min_support = min_support,
    min_improvement = min_improvement
  )

  # 1) Extract model frame
  mf   <- stats::model.frame(formula, data)
  Y    <- mf[[1]]
  X    <- mf[-1]
  resp <- all.vars(formula)[1]

  # Check for non-numeric predictors
  vars_to_check <- setdiff(names(X), exclude_vars)
  non_numeric_vars <- vars_to_check[!sapply(X[vars_to_check], is.numeric)]

  if (length(non_numeric_vars) > 0) {
    stop(
      sprintf(
        paste0(
          "SplitWise requires all predictors to be numeric.\n",
          "The following variables are not numeric: %s\n",
          "Consider transforming these to numeric ",
          "or excluding them via `exclude_vars`."
        ),
        paste(non_numeric_vars, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # 2) Choose transformation approach
  if (transformation_mode == "univariate") {
    # Univariate approach
    decisions <- decide_variable_type_univariate(
      X,
      Y,
      min_support     = min_support,
      min_improvement = min_improvement,
      criterion       = criterion,
      exclude_vars    = exclude_vars,
      verbose         = verbose
    )
    X_trans <- transform_features_univariate(X, decisions)
  } else {
    # Iterative approach
    decisions <- decide_variable_type_iterative(
      X,
      Y,
      min_support     = min_support,
      min_improvement = min_improvement,
      direction       = direction,
      criterion       = criterion,
      exclude_vars    = exclude_vars,
      verbose         = verbose,
      ...
    )
    X_trans <- transform_features_iterative(X, decisions)
  }

  # 3) Combine with response
  df_final <- cbind(X_trans, setNames(list(Y), resp))

  # 4) Fit full model
  if (direction == "forward") {
    # Nullmodell, csak intercept
    initial_formula <- stats::as.formula(paste(resp, "~ 1"))
    initial_model <- stats::lm(initial_formula, data = df_final)
  } else {
    # Full model
    full_formula <- stats::as.formula(
      paste(resp, "~", paste(colnames(X_trans), collapse = " + "))
    )
    initial_model <- stats::lm(full_formula, data = df_final)
  }

  # 5) Stepwise model selection
  scope_list <- list(
    lower = stats::as.formula(paste(resp, "~ 1")),
    upper = stats::as.formula(
      paste(resp, "~", paste(colnames(X_trans), collapse = " + "))
    )
  )

  trace <- if (isTRUE(verbose)) 1 else 0

  dots <- list(...)
  if ("trace" %in% names(dots)) dots$trace <- NULL

  step_model <- do.call(stats::step, c(
    list(
      object    = initial_model,
      scope     = scope_list,
      direction = direction,
      trace     = trace,
      steps     = steps,
      k         = k
    ),
    dots
  ))

  # 6) Attach metadata
  step_model$splitwise_info <- list(
    transformation_mode = transformation_mode,
    decisions           = decisions,
    final_data          = df_final,
    call                = match.call()
  )

  class(step_model) <- c("splitwise_lm", class(step_model))
  return(step_model)
}

# ------------------------------------------------------------------------------

#' @title Internal Argument Validation for SplitWise
#'
#' @description Validates arguments for the SplitWise regression functions.
#'   Throws an error if an argument is not valid.
#' @param min_support Numeric scalar. Minimum support for a dummy split
#'   (between 0 and 0.5, exclusive).
#' @param min_improvement Numeric scalar. Minimum improvement in criterion
#'   required for a transformation (>= 0).
#' @return \code{NULL} (called for its side effects; throws error if invalid)
#' @keywords internal
#' @noRd
#'
validate_splitwise_args <- function(min_support, min_improvement) {
  # min_support: must be numeric scalar, 0 < min_support < 0.5
  if (!is.numeric(min_support) || length(min_support) != 1 ||
      min_support <= 0 || min_support >= 0.5) {
    stop("`min_support` must be a numeric value between 0 and 0.5 (exclusive).")
  }
  # min_improvement: must be numeric scalar >= 0
  if (!is.numeric(min_improvement) || length(min_improvement) != 1 ||
      min_improvement < 0) {
    stop("`min_improvement` must be a non-negative numeric value.")
  }
  # ... add more checks as needed
}

# ------------------------------------------------------------------------------

#' @title Print Method for SplitWise Linear Model
#'
#' @param x A \code{"splitwise_lm"} object returned by \code{splitwise}.
#' @param ... Additional arguments (unused).
#' @describeIn splitwise Prints a summary of the splitwise_lm object.
#' @importFrom stats coef AIC BIC
#' @export
#'
print.splitwise_lm <- function(x, ...) {
  cat("SplitWise Linear Models\n")
  cat("Transformation mode:", x$splitwise_info$transformation_mode, "\n")
  cat("Call:\n")
  print(x$splitwise_info$call)

  # Grab all decisions
  decisions <- x$splitwise_info$decisions

  # Filter out the dummy ones
  dummy_vars <- names(Filter(function(d) d$type == "dummy", decisions))

  if (length(dummy_vars) > 0) {
    cat("\nDummy-Encoded Variables:\n")
    for (var_name in dummy_vars) {
      cut_vals <- decisions[[var_name]]$cutoff
      if (length(cut_vals) == 1) {
        # Single cutoff: 1 if x >= cutoff
        cat("  -", var_name,
            ": 1 if x >= ",
            format(round(cut_vals, 3), nsmall = 3),
            "; else 0\n"
        )
      } else if (length(cut_vals) == 2) {
        # Range dummy: 1 if lower < x < upper
        lower_val <- round(cut_vals[1], 3)
        upper_val <- round(cut_vals[2], 3)
        cat("  -", var_name,
            ": 1 if ", lower_val, " < x < ", upper_val,
            "; else 0\n"
        )
      } else {
        # fallback if something else is stored
        cat("  -", var_name, ": (unknown dummy definition?)\n")
      }
    }
  } else {
    cat("\nNo variables were dummy encoded.\n")
  }

  cat("\nCoefficients:\n")
  print(stats::coef(x))

  # Print both AIC & BIC for clarity
  cat("\nModel Fit Statistics:\n")
  cat("  AIC:", format(stats::AIC(x), digits = 5), "\n")
  cat("  BIC:", format(stats::BIC(x), digits = 5), "\n")

  invisible(x)
}

# ------------------------------------------------------------------------------

#' @title Summary Method for SplitWise Linear Model
#'
#' @param object A \code{"splitwise_lm"} object returned by \code{splitwise}.
#' @param ... Additional arguments passed to \code{summary.lm}.
#' @describeIn splitwise Provides a detailed summary, including how dummies
#'   were created.
#' @importFrom stats summary.lm AIC BIC
#' @export
#'
summary.splitwise_lm <- function(object, ...) {

  # 1) Print the standard lm summary
  base_summary <- stats::summary.lm(object, ...)
  print(base_summary)

  # 2) Grab transformations + produce custom info
  cat("Transformation Mode:", object$splitwise_info$transformation_mode, "\n")

  # Identify dummy variables
  decisions   <- object$splitwise_info$decisions
  dummy_vars  <- names(Filter(function(d) d$type == "dummy", decisions))

  if (length(dummy_vars) > 0) {
    cat("\nDummy-Encoded Variables:\n")
    for (var_name in dummy_vars) {
      cut_vals <- decisions[[var_name]]$cutoff

      if (length(cut_vals) == 1) {
        # Single cutoff: "1 if x >= cutoff"
        cat("  -", var_name,
            ": 1 if x >=",
            format(round(cut_vals, 3), nsmall = 3),
            "; else 0\n"
        )
      } else if (length(cut_vals) == 2) {
        # Range dummy: "1 if a < x < b"
        lower_val <- format(round(min(cut_vals), 3), nsmall = 3)
        upper_val <- format(round(max(cut_vals), 3), nsmall = 3)
        cat("  -", var_name,
            ": 1 if", lower_val, "< x <", upper_val,
            "; else 0\n"
        )
      } else {
        # Fallback if something else is stored
        cat("  -", var_name, ": (unknown dummy definition?)\n")
      }
    }
  } else {
    cat("\nNo variables were dummy encoded.\n")
  }

  # Print both AIC & BIC for clarity
  cat("\nFinal AIC:", format(stats::AIC(object), digits = 5), "\n")
  cat("Final BIC:", format(stats::BIC(object), digits = 5), "\n")

  invisible(base_summary)
}

# ------------------------------------------------------------------------------

#' @title Predict Method for SplitWise Linear Models
#'
#' @param object An object of class \code{splitwise_lm}, as returned by
#'   \code{\link{splitwise}}.
#' @param newdata A data frame of new data (with original predictors) to
#'   generate predictions for. The appropriate dummy variables will be
#'   generated using the transformation rules learned during model training. If
#'   omitted, predictions for the training data are returned.
#' @param ... Additional arguments passed to \code{predict.lm}.
#' @describeIn splitwise Generate predictions from a \code{splitwise_lm} object
#'   using learned transformation rules.
#' @importFrom stats predict.lm
#' @export
#'
predict.splitwise_lm <- function(object, newdata, ...) {
  # Extract transformation info
  info <- object$splitwise_info
  decisions <- info$decisions

  # Defensive: if newdata missing, use fitted
  if (missing(newdata)) {
    return(stats::predict.lm(object, ...))
  }
  stopifnot(is.data.frame(newdata))

  # Apply transformations
  for (var in names(decisions)) {
    rule <- decisions[[var]]
    # Remove any previously created _dummy column just in case
    dummy_col <- paste0(var, "_dummy")
    newdata[[dummy_col]] <- NULL
    if (rule$type == "dummy") {
      cutoffs <- rule$cutoff
      if (length(cutoffs) == 1) {
        newdata[[dummy_col]] <- as.numeric(newdata[[var]] >= cutoffs[1])
      } else if (length(cutoffs) == 2) {
        c1 <- min(cutoffs)
        c2 <- max(cutoffs)
        newdata[[dummy_col]] <- as.numeric(newdata[[var]] > c1 &
                                             newdata[[var]] < c2)
      }
    }
    # If linear, do nothing (column already exists)
  }

  # Determine which columns are in the model coefficients
  coef_names <- names(stats::coef(object))
  coef_names <- coef_names[coef_names != "(Intercept)"]

  # Ensure all predictors required by model are present in newdata
  missing_vars <- setdiff(coef_names, names(newdata))
  if (length(missing_vars) > 0) {
    stop(
      sprintf(
        paste0(
          "The following variables required by the SplitWise model ",
          "are missing in newdata: %s"
        ),
        paste(missing_vars, collapse = ", ")
      )
    )
  }

  # Subset newdata to model predictors (order matters!)
  newdata_model <- newdata[, coef_names, drop = FALSE]

  # Use base lm prediction
  pred <- stats::predict.lm(object, newdata = newdata_model, ...)
  return(pred)
}

# ------------------------------------------------------------------------------

#' @title Extract Coefficients from a SplitWise Model
#'
#' @param object An object of class \code{splitwise_lm}, as returned by
#'   \code{\link{splitwise}}.
#' @describeIn splitwise Extract model coefficients from a SplitWise linear
#'   model.
#' @importFrom stats coef
#' @export
#'
coef.splitwise_lm <- function(object, ...) {
  stats::coef(unclass(object))
}

# ------------------------------------------------------------------------------

#' @title Extract Fitted Values from a SplitWise Model
#'
#' @param object An object of class \code{splitwise_lm}, as returned by
#'   \code{\link{splitwise}}.
#' @describeIn splitwise Extract fitted values from a SplitWise linear model.
#' @importFrom stats fitted.values
#' @export
#'
fitted.splitwise_lm <- function(object, ...) {
  class(object) <- setdiff(class(object), "splitwise_lm")
  stats::fitted.values(object, ...)
}

# ------------------------------------------------------------------------------

#' @title Extract Residuals from a SplitWise Model
#'
#' @param object An object of class \code{splitwise_lm}, as returned by
#'   \code{\link{splitwise}}.
#' @describeIn splitwise Extract residuals from a SplitWise linear model.
#' @importFrom stats residuals
#' @export
#'
residuals.splitwise_lm <- function(object, ...) {
  class(object) <- setdiff(class(object), "splitwise_lm")
  stats::residuals(object, ...)
}

# ------------------------------------------------------------------------------


#' @title Extract Model Matrix from a SplitWise Model
#'
#' @param object An object of class \code{splitwise_lm}, as returned by
#'   \code{\link{splitwise}}.
#' @describeIn splitwise Extract the model matrix from a SplitWise linear model.
#' @importFrom stats model.matrix
#' @export
#'
model.matrix.splitwise_lm <- function(object, ...) {
  class(object) <- setdiff(class(object), "splitwise_lm")
  stats::model.matrix(object, ...)
}
