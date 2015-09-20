using MarketData

facts("arima builds models") do

  context("works when it supposed to") do
      @fact length(cl.values) --> 500
      params, mode = ar(cl.values, 2)
  end

  context("fails when it's supposed to") do
      @fact_throws cl.value
  end

  context("test ARX(p) against R output") do
      params, mod = arx(ohlcv["Close"].values, ohlcv["Volume"].values, 2)
  end
end

