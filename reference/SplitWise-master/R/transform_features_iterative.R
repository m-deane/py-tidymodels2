#' @title Transform Features (Iterative Logic)
#'
#' @description Once \code{decide_variable_type_iterative} has chosen which
#'   variables to add (and how), we can build a final data frame from those
#'   decisions.
#' @param X Original predictor data frame.
#' @param decisions Output of \code{decide_variable_type_iterative}.
#' @return A data frame with the chosen variables in their final forms (dummy
#'   or linear).
#' @keywords internal
#'
transform_features_iterative <- function(X, decisions) {
  transformed <- data.frame(matrix(nrow = nrow(X), ncol = 0))

  for (var_name in names(decisions)) {
    d <- decisions[[var_name]]
    if (d$type == "linear") {
      transformed[[var_name]] <- X[[var_name]]
    } else {
      dummy_name <- paste0(var_name, "_dummy")
      transformed[[dummy_name]] <- as.numeric(X[[var_name]] >= d$cutoff)
    }
  }
  return(transformed)
}
