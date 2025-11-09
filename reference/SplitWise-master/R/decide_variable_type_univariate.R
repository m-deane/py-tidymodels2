#' @title Decide Variable Type (Univariate)
#'
#' @description For each numeric predictor, this function fits a shallow
#'   (\code{maxdepth = 2}) \code{rpart} tree directly on \code{Y ~ x} and tests
#'   whether a dummy transformation improves model fit.
#' @details By default, no split is allowed with fewer than 5 observations
#'   (i.e., minsplit is max(5, ceiling(min_support * n))). This is not
#'   user-configurable.
#'
#'   Dummy forms come from a shallow (\code{maxdepth = 2}) \code{rpart} tree
#'   fit to the data. We extract up to two splits:
#'   \itemize{
#'     \item Single cutoff dummy (e.g., \code{x >= c})
#'     \item Double cutoff dummy (e.g., \code{c1 < x < c2})
#'   }
#'
#'   The function then picks the form (linear, single-split dummy, or
#'   double-split dummy) that yields the lowest AIC/BIC. If a variable is
#'   listed in \code{exclude_vars}, it will always be used as a linear
#'   predictor (dummy transformation is never attempted).
#' @param X A data frame of numeric predictors (no response).
#' @param Y A numeric response vector.
#' @param min_support Minimum fraction (0-0.5) of observations required in
#'   either group after a dummy split. Default = 0.1.
#' @param min_improvement Minimum required improvement in AIC/BIC for accepting
#'   a dummy split or variable transformation. Default = 3.
#' @param criterion A character string: either \code{"AIC"} or \code{"BIC"}.
#'   Default = \code{"AIC"}.
#' @param exclude_vars A character vector of variable names to exclude from
#'   dummy transformations. These variables will always be treated as linear.
#'   Default = \code{NULL}.
#' @param verbose Logical; if \code{TRUE}, prints messages for debugging.
#'   Default = \code{FALSE}.
#' @return A named list of decisions, where each element is a list with:
#' \describe{
#'   \item{type}{Either \code{"dummy"} or \code{"linear"}.}
#'   \item{cutoffs}{A numeric vector (length 1 or 2) if \code{type = "dummy"},
#'   or \code{NULL} if linear.}
#'   \item{tree_model}{The fitted \code{rpart} model (for reference) or
#'   \code{NULL} if excluded.}
#' }
#' @importFrom stats lm AIC BIC
#' @importFrom rpart rpart rpart.control
#' @keywords internal
#'
decide_variable_type_univariate <- function(X, Y,
                                            min_support = 0.1,
                                            min_improvement = 3,
                                            criterion = c("AIC", "BIC"),
                                            exclude_vars = NULL,
                                            verbose   = FALSE) {

  criterion <- match.arg(criterion)
  get_crit  <- if (criterion == "AIC") AIC else BIC

  if (is.null(exclude_vars)) {
    exclude_vars <- character(0)
  }

  decisions <- list()

  for (col_name in names(X)) {
    x_vec <- X[[col_name]]

    # --------------------------------------------------------------------------
    # 0) If user requested exclusion, skip dummy check
    # --------------------------------------------------------------------------
    if (col_name %in% exclude_vars) {
      if (verbose) {
        message(sprintf(
          "Excluding '%s' from dummy transformations; using linear only.",
          col_name))
      }
      decisions[[col_name]] <- list(type = "linear", cutoffs = NULL,
                                    tree_model = NULL)
      next
    }

    # Basic edge-case check: skip columns with no variance or all NA
    if (all(is.na(x_vec))) {
      if (verbose) {
        message(sprintf("Skipping '%s': all values are NA.", col_name))
      }
      decisions[[col_name]] <- list(type = "linear", cutoffs = NULL,
                                    tree_model = NULL)
      next
    }
    if (length(unique(x_vec[!is.na(x_vec)])) == 1) {
      if (verbose) {
        message(sprintf("Skipping '%s': no variance in predictor.", col_name))
      }
      decisions[[col_name]] <- list(type = "linear", cutoffs = NULL,
                                    tree_model = NULL)
      next
    }

    # --------------------------------------------------------------------------
    # 1) Fit a shallow rpart on Y ~ x
    # --------------------------------------------------------------------------
    temp_df <- data.frame(x = x_vec, y = Y)
    minsplit <- max(5, ceiling(min_support * nrow(temp_df)))
    tree_model <- rpart::rpart(
      formula = y ~ x,
      data    = temp_df,
      model   = TRUE,
      control = rpart::rpart.control(maxdepth = 2, minsplit = minsplit)
    )

    # --------------------------------------------------------------------------
    # 2) Evaluate "linear" vs. "dummy"
    #    Compare AIC/BIC of:
    #      - a simple LM with x as linear
    #      - a LM with 1- or 2-split dummy (only if supports are large enough)
    # --------------------------------------------------------------------------
    # Linear
    df_lin  <- data.frame(y = Y, x = x_vec)
    mod_lin <- lm(y ~ x, data = df_lin)
    lin_val <- get_crit(mod_lin)

    best_val    <- lin_val
    best_type   <- "linear"
    best_cutoff <- NULL

    if (verbose) {
      message(sprintf("[%s] Linear %s: %.3f", col_name, criterion, lin_val))
    }

    # If rpart found at least one split => try single vs double
    best_dummy_val <- Inf
    best_dummy_cuts <- NULL

    if (!is.null(tree_model$splits) && nrow(tree_model$splits) >= 1) {
      all_splits <- sort(tree_model$splits[, "index"])

      # Single-split dummy
      dcol1 <- as.numeric(x_vec >= all_splits[1])
      s1_support <- mean(dcol1)
      if (s1_support > min_support && s1_support < (1 - min_support)) {
        s1_df <- data.frame(
          y = Y,
          dummy = dcol1
        )
        mod_s1  <- lm(y ~ dummy, data = s1_df)
        s1_val  <- get_crit(mod_s1)
      } else {
        s1_val <- Inf
      }
      best_dummy_val  <- s1_val
      best_dummy_cuts <- all_splits[1]

      # Double-split dummy
      if (length(all_splits) >= 2) {
        c1 <- all_splits[1]
        c2 <- all_splits[2]
        dcol2 <- as.numeric(x_vec > c1 & x_vec < c2)
        s2_support <- mean(dcol2)
        if (s2_support > min_support && s2_support < (1 - min_support)) {
          s2_df <- data.frame(
            y = Y,
            dummy = dcol2
          )
          mod_s2  <- lm(y ~ dummy, data = s2_df)
          s2_val  <- get_crit(mod_s2)
        } else {
          s2_val <- Inf
        }

        if (s2_val < s1_val) {
          best_dummy_val  <- s2_val
          best_dummy_cuts <- c(c1, c2)
        }
      }

      if (verbose && is.finite(best_dummy_val)) {
        if (length(best_dummy_cuts) == 1) {
          message(sprintf("[%s] Dummy single %s: %.3f", col_name,
                          criterion, best_dummy_val))
        } else {
          message(sprintf("[%s] Dummy double %s: %.3f", col_name,
                          criterion, best_dummy_val))
        }
      }
    }

    # Only accept dummy if it improves enough AND support is valid
    if (is.finite(best_dummy_val) &&
        (lin_val - best_dummy_val) >= min_improvement) {
      best_val    <- best_dummy_val
      best_type   <- "dummy"
      best_cutoff <- best_dummy_cuts
    }

    if (verbose) {
      message(sprintf("[%s] Chosen: %s (%.3f)", col_name, best_type, best_val))
    }

    decisions[[col_name]] <- list(
      type       = best_type,
      cutoffs    = best_cutoff,
      tree_model = if (best_type == "dummy") tree_model else NULL
    )
  }

  return(decisions)
}
