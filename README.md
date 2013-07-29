# Julia GARCH package

Generalized Autoregressive Conditional Heteroskedastic models for Julia.

## What is implemented

* garchFit - estimates parameters of univariate normal GARCH process.
* predict - make prediction using fitted object returned by garchFit

## What is not ready yet

* more complex GARCH models
* Residuals tests
* n-step forecasts
* Error analysis

## Example

using GARCH
using Quandl
quotes = quandl("YAHOO/INDEX_GSPC")
ret = diff(log(quotes["Close"]*100))
ret = ret - mean(ret)
garchFit(convert(Vector,ret[end-199:end]))
