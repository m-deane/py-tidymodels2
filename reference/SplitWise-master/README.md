
<!-- README.md is generated from README.Rmd. Please edit that file -->

# SplitWise R Package

<!-- badges: start -->
[![R CMD check via R-hub](https://github.com/mtkurbucz/SplitWise/actions/workflows/rhub.yaml/badge.svg)](https://github.com/mtkurbucz/SplitWise/actions/workflows/rhub.yaml)
<!-- badges: end -->

SplitWise is a hybrid stepwise regression package that intelligently transforms numeric predictors using single- or double-split dummy encoding. Each variable can be retained as a continuous feature or transformed into binary indicators based on model fit, evaluated using AIC or BIC.

By default, SplitWise uses an iterative transformation mode, which evaluates each variable in the context of others—enabling more accurate and interpretable models by capturing feature synergies. For faster execution on large datasets, a simpler univariate mode is also available, which transforms each variable independently. While computationally efficient, this mode may miss interactions captured by the iterative approach.

<blockquote style="font-size: 85%; font-style: italic; border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;">
For a full description of the methodology, see the accompanying arXiv preprint: Kurbucz, Marcell T.; Tzivanakis, Nikolaos; Aslam, Nilufer Sari; Sykulski, Adam M. (2025). <i>SplitWise Regression: Stepwise Modeling with Adaptive Dummy Encoding.</i> arXiv preprint <a href="https://arxiv.org/abs/2505.15423">https://arxiv.org/abs/2505.15423</a>.
</blockquote>

## Installation

To install from GitHub:

```r
# if not already installed:
install.packages("devtools")
devtools::install_github("mtkurbucz/SplitWise")
```

## Usage

``` r
library(SplitWise)
data(mtcars)

# Univariate transformation (backward selection using AIC)
model_uni <- splitwise(
  mpg ~ .,
  data = mtcars,
  transformation_mode = "univariate",
  direction = "backward",
  criterion = "AIC",
  trace = 0
)
summary(model_uni)

# Iterative transformation (forward selection using BIC)
model_iter <- splitwise(
  mpg ~ .,
  data = mtcars,
  transformation_mode = "iterative",
  direction = "forward",
  criterion = "BIC",
  k = log(nrow(mtcars)), # BIC penalty
  trace = 0
)
print(model_iter)
```

## Further Documentation

Further documentation is available in the [doc/](https://github.com/mtkurbucz/SplitWise/tree/main/doc) folder on GitHub.

## Citation

If you use `SplitWise` in your research, please cite:

<blockquote style="font-size: 85%; font-style: italic; border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;">
Kurbucz, Marcell T.; Tzivanakis, Nikolaos; Aslam, Nilufer Sari; Sykulski, Adam M. (2025). <i>SplitWise Regression: Stepwise Modeling with Adaptive Dummy Encoding.</i> arXiv preprint <a href="https://arxiv.org/abs/2505.15423">https://arxiv.org/abs/2505.15423</a>.
</blockquote>

## License

This package is licensed under the GPL-3 License.
