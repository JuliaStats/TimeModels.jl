using MarketData

facts("diagnostic tests produce accurate results") do

  context("jaques-berra test") do
      @fact length(cl.values) --> 500
  end
end

