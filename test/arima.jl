using MarketData

facts("arima builds models") do

  context("works when it supposed to") do
      @fact length(cl.values) => 500
  end

  context("fails when it's supposed to") do
      @fact_throws cl.value
  end
end

