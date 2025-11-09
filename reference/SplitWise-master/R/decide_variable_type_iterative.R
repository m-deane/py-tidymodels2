#' @title Decide Variable Type (Iterative)
#'
#' @description A stepwise variable-selection method that iteratively chooses
#'   each variable's best form: \code{"linear"}, single-split \code{"dummy"},
#'   or double-split ("middle=1") dummy, based on AIC/BIC improvement. Supports
#'   "forward", "backward", or "both" strategies.
#' @details By default, no split is allowed with fewer than 5 observations
#'   (i.e., minsplit is max(5, ceiling(min_support * n))). This is not
#'   user-configurable.
#'
#'   Dummy forms come from a shallow (\code{maxdepth = 2})
#'   \code{rpart} tree fit to the partial residuals of the current model. We
#'   extract up to two splits:
#'   \itemize{
#'     \item Single cutoff dummy (e.g., \code{x >= c})
#'     \item Double cutoff dummy (e.g., \code{c1 < x < c2})
#'   }
#'
#'   The function then picks the form (linear, single-split dummy, or
#'   double-split dummy) that yields the lowest AIC/BIC. Variables listed in
#'   \code{exclude_vars} will be forced to remain linear (dummy transformations
#'   are never attempted).
#' @param X A data frame of predictors (no response).
#' @param Y A numeric vector (the response).
#' @param min_support Minimum fraction (0-0.5) of observations required in
#'   either group after a dummy split. Default = 0.1.
#' @param min_improvement Minimum required improvement in AIC/BIC for accepting
#'   a dummy split or variable transformation. Default = 3.
#' @param direction Stepwise strategy: \code{"forward"}, \code{"backward"}, or
#'   \code{"both"}. Default = \code{"backward"}.
#' @param criterion A character string: either \code{"AIC"} or \code{"BIC"}.
#'   Default = \code{"AIC"}.
#' @param exclude_vars A character vector of variable names to exclude from
#'   dummy transformations. These variables will always be treated as linear.
#'   Default = \code{NULL}.
#' @param verbose Logical; if \code{TRUE}, prints messages for debugging.
#'   Default = \code{FALSE}.
#' @param ... Additional arguments (currently unused).
#' @return A named list of decisions, where each element is a list with:
#' \describe{
#'   \item{type}{Either \code{"linear"} or \code{"dummy"}.}
#'   \item{cutoff}{A numeric vector of length 1 or 2 (the chosen split points).}
#' }
#' @importFrom stats lm AIC BIC coef
#' @importFrom rpart rpart rpart.control
#' @keywords internal
#'
decide_variable_type_iterative <- function(X, Y,
                                           min_support = 0.1,
                                           min_improvement = 3,
                                           direction = c(
                                             "backward","forward","both"),
                                           criterion = c("AIC", "BIC"),
                                           exclude_vars = NULL,
                                           verbose   = FALSE,
                                           ...) {

  direction <- match.arg(direction)
  criterion <- match.arg(criterion)
  get_crit  <- if (criterion == "AIC") AIC else BIC

  if (is.null(exclude_vars)) {
    exclude_vars <- character(0)
  }

  df_design <- data.frame(Y = Y)  # initial design matrix with just Y
  decisions <- list()

  #-----------------------------------------------------------------------------
  # 1) HELPER FUNCTIONS
  #-----------------------------------------------------------------------------
  add_var_to_design <- function(df, x_col, var_name, type, cutoffs = NULL) {
    df_new <- df
    if (type == "linear") {
      df_new[[var_name]] <- x_col
    } else {
      # type == "dummy"
      dummy_name <- paste0(var_name, "_dummy")
      if (length(cutoffs) == 1) {
        # single cutoff => x >= c
        cval <- cutoffs[1]
        df_new[[dummy_name]] <- as.numeric(x_col >= cval)
      } else if (length(cutoffs) == 2) {
        # double cutoff => c1 < x < c2 => 1 in the "middle"
        c1 <- min(cutoffs)
        c2 <- max(cutoffs)
        df_new[[dummy_name]] <- as.numeric(x_col > c1 & x_col < c2)
      } else {
        # fallback: no dummy created
        df_new[[dummy_name]] <- 0
      }
    }
    df_new
  }

  remove_var_from_design <- function(df, var_name, type) {
    df_new <- df
    if (type == "linear") {
      df_new[[var_name]] <- NULL
    } else {
      # If dummy, remove the "_dummy" column
      dummy_name <- paste0(var_name, "_dummy")
      df_new[[dummy_name]] <- NULL
    }
    df_new
  }

  get_partial_resid <- function(mod, x_col, var_name) {
    # If var_name is in model, partial residual = residual + coefficient*x
    coefs <- coef(mod)
    if (var_name %in% names(coefs)) {
      mod$residuals + coefs[[var_name]] * x_col
    } else {
      mod$residuals
    }
  }

  find_best_form <- function(mod_current, df_current, x_col,
                             var_name, min_support) {
    if (var_name %in% exclude_vars) return(list(aic = Inf, type = "linear",
                                                cutoff = NULL))

    df_lin  <- add_var_to_design(df_current, x_col, var_name, "linear")
    mod_lin <- lm(Y ~ ., data = df_lin)
    lin_val <- get_crit(mod_lin)

    part_resid <- get_partial_resid(mod_current, x_col, var_name)
    tree_df    <- data.frame(y = part_resid, x = x_col)
    minsplit <- max(5, ceiling(min_support * nrow(tree_df)))
    tmp_tree   <- rpart::rpart(
      formula = y ~ x,
      data    = tree_df,
      control = rpart::rpart.control(maxdepth = 2, minsplit = minsplit)
    )

    dummy_val  <- Inf
    chosen_cut <- NULL
    dummy_support <- 0

    if (!is.null(tmp_tree$splits) && nrow(tmp_tree$splits) >= 1) {
      splits_sorted <- sort(tmp_tree$splits[, "index"])
      single_cut    <- splits_sorted[1]

      # Single dummy
      dcol1 <- as.numeric(x_col >= single_cut)
      s1_support <- mean(dcol1)
      if (s1_support > min_support && s1_support < (1 - min_support)) {
        df_dummy_s1  <- add_var_to_design(df_current, x_col,
                                          var_name, "dummy", single_cut)
        mod_dummy_s1 <- lm(Y ~ ., data = df_dummy_s1)
        s1_val       <- get_crit(mod_dummy_s1)
      } else {
        s1_val <- Inf
      }
      best_dummy_val  <- s1_val
      best_dummy_cuts <- single_cut

      # Double-split if possible
      if (length(splits_sorted) >= 2) {
        c1 <- splits_sorted[1]
        c2 <- splits_sorted[2]
        dcol2 <- as.numeric(x_col > c1 & x_col < c2)
        s2_support <- mean(dcol2)
        if (s2_support > min_support && s2_support < (1 - min_support)) {
          df_dummy_s2  <- add_var_to_design(df_current, x_col,
                                            var_name, "dummy", c(c1, c2))
          mod_dummy_s2 <- lm(Y ~ ., data = df_dummy_s2)
          s2_val       <- get_crit(mod_dummy_s2)
        } else {
          s2_val <- Inf
        }

        if (s2_val < s1_val) {
          best_dummy_val  <- s2_val
          best_dummy_cuts <- c(c1, c2)
        }
      }

      dummy_val  <- best_dummy_val
      chosen_cut <- best_dummy_cuts
    }

    # min_improvement threshold (only allow if improvement is big enough)
    if ((lin_val - dummy_val) >= min_improvement) {
      list(aic = dummy_val, type = "dummy", cutoff = chosen_cut)
    } else {
      list(aic = lin_val, type = "linear", cutoff = NULL)
    }
  }

  #-----------------------------------------------------------------------------
  # 2) INITIAL SETUP
  #-----------------------------------------------------------------------------
  # "backward" => start by picking best univariate form for each var
  if (direction == "backward") {
    for (var_name in names(X)) {
      x_col <- X[[var_name]]

      # Skip if all NA or no variance
      if (all(is.na(x_col)) || length(unique(x_col[!is.na(x_col)])) == 1) {
        if (verbose) {
          message(sprintf("Skipping '%s': no variance or all NA.", var_name))
        }
        decisions[[var_name]] <- list(type = "linear", cutoff = NULL)
        next
      }

      # Intercept-only for partial resid
      univariate_mod <- lm(Y ~ 1, data = df_design)
      best_form <- find_best_form(
        mod_current = univariate_mod,
        df_current  = df_design,
        x_col       = x_col,
        var_name    = var_name,
        min_support = min_support
      )
      decisions[[var_name]] <- list(type = best_form$type,
                                    cutoff = best_form$cutoff)
      df_design <- add_var_to_design(
        df_design, x_col, var_name, best_form$type, best_form$cutoff
      )

      if (verbose) {
        message(sprintf("Initial [%s]: %s = %.3f", var_name, criterion,
                        best_form$aic))
      }
    }
  }

  mod_current <- lm(Y ~ ., data = df_design)
  best_val    <- get_crit(mod_current)

  if (verbose) {
    message(sprintf("Starting model %s: %.3f", criterion, best_val))
  }

  unused_vars <- setdiff(names(X), names(decisions))

  #-----------------------------------------------------------------------------
  # 3) MAIN LOOP
  #-----------------------------------------------------------------------------
  improvement <- TRUE
  while (improvement) {
    improvement      <- FALSE
    best_var_choice  <- NULL
    best_type        <- NULL
    best_cutoff      <- NULL
    best_switch_from <- NULL
    best_new_val     <- best_val

    # (A) Forward or both => try adding new variables
    #---------------------------------------------------------------------------
    if (direction %in% c("forward", "both") && length(unused_vars) > 0) {
      for (var_name in unused_vars) {
        x_col <- X[[var_name]]

        if (all(is.na(x_col)) || length(unique(x_col[!is.na(x_col)])) == 1) {
          next
        }

        form_info <- find_best_form(mod_current, df_design, x_col,
                                    var_name, min_support)

        if ((best_val - form_info$aic) > min_improvement &&
            form_info$aic < best_new_val) {
          best_new_val     <- form_info$aic
          best_var_choice  <- var_name
          best_type        <- form_info$type
          best_cutoff      <- form_info$cutoff
          best_switch_from <- NULL
        }
      }
    }

    # (B) Backward or both => try removing or switching existing variables
    #---------------------------------------------------------------------------
    if (direction %in% c("backward", "both")) {
      vars_in_model <- names(decisions)

      for (var_name in vars_in_model) {
        old_type <- decisions[[var_name]]$type
        x_col    <- X[[var_name]]

        df_drop  <- remove_var_from_design(df_design, var_name, old_type)
        mod_drop <- lm(Y ~ ., data = df_drop)
        drop_val <- get_crit(mod_drop)

        if ((best_val - drop_val) > min_improvement &&
            drop_val < best_new_val) {
          best_new_val     <- drop_val
          best_var_choice  <- var_name
          best_type        <- "remove"
          best_cutoff      <- NULL
          best_switch_from <- old_type
        }

        # Try switching forms
        df_temp  <- df_drop
        mod_temp <- mod_drop
        form_info <- find_best_form(mod_temp, df_temp, x_col,
                                    var_name, min_support)

        if ((best_val - form_info$aic) > min_improvement &&
            form_info$aic < best_new_val) {
          best_new_val     <- form_info$aic
          best_var_choice  <- var_name
          best_type        <- form_info$type
          best_cutoff      <- form_info$cutoff
          best_switch_from <- old_type
        }
      }
    }

    # (C) Forward or both => re-check switching existing variables
    #---------------------------------------------------------------------------
    if (direction %in% c("forward", "both")) {
      for (var_name in names(decisions)) {
        current_type <- decisions[[var_name]]$type
        x_col        <- X[[var_name]]

        df_temp  <- remove_var_from_design(df_design, var_name, current_type)
        mod_temp <- lm(Y ~ ., data = df_temp)
        form_info <- find_best_form(mod_temp, df_temp, x_col,
                                    var_name, min_support)

        if ((best_val - form_info$aic) > min_improvement &&
            form_info$aic < best_new_val) {
          best_new_val     <- form_info$aic
          best_var_choice  <- var_name
          best_type        <- form_info$type
          best_cutoff      <- form_info$cutoff
          best_switch_from <- current_type
        }
      }
    }

    # (D) Implement best improvement if found
    #---------------------------------------------------------------------------
    if (!is.null(best_var_choice) && best_new_val < best_val) {
      improvement <- TRUE

      if (!is.null(best_switch_from) && best_switch_from != "remove" &&
          best_type == "remove") {

        # Removing a variable
        df_design <- remove_var_from_design(df_design, best_var_choice,
                                            best_switch_from)
        decisions[[best_var_choice]] <- NULL
        unused_vars <- union(unused_vars, best_var_choice)

        if (verbose) {
          message(sprintf("Removed '%s': new %s = %.3f", best_var_choice,
                          criterion, best_new_val))
        }

      } else if (!is.null(best_switch_from) &&
                 best_switch_from %in% c("linear","dummy") &&
                 best_type != "remove") {
        # Switching forms
        df_design <- remove_var_from_design(df_design, best_var_choice,
                                            best_switch_from)
        df_design <- add_var_to_design(df_design, X[[best_var_choice]],
                                       best_var_choice,
                                       best_type, best_cutoff)
        decisions[[best_var_choice]] <- list(type = best_type,
                                             cutoff = best_cutoff)

        if (verbose) {
          message(sprintf("Switched '%s' to '%s': new %s = %.3f",
                          best_var_choice, best_type, criterion, best_new_val))
        }

      } else {
        # Adding a brand-new variable
        df_design <- add_var_to_design(df_design, X[[best_var_choice]],
                                       best_var_choice,
                                       best_type, best_cutoff)
        decisions[[best_var_choice]] <- list(type = best_type,
                                             cutoff = best_cutoff)
        unused_vars <- setdiff(unused_vars, best_var_choice)

        if (verbose) {
          message(sprintf("Added '%s' as '%s': new %s = %.3f",
                          best_var_choice, best_type, criterion, best_new_val))
        }
      }

      # Update model & best_val
      mod_current <- lm(Y ~ ., data = df_design)
      best_val    <- best_new_val
    }
  }

  return(decisions)
}
