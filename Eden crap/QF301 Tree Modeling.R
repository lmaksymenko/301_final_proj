#set's working directory to current folder
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#goal: use the previous 5 days of VaR of each sector to predict the next 
#5 day VaR of the market

library(PerformanceAnalytics)
library(quantmod)
getSymbols("SPY", from = "2007-01-01", to = "2010-12-31")
spyret = dailyReturn(SPY$SPY.Adjusted, type = "log")[-1]
head(spyret)
vec = c(6, rep(5, 200))
spl = split(spyret,rep(1:201,vec))
spy_var = as.numeric(lapply(spl, VaR))[-1]



train = read.csv("train_data.csv", row.names = "X")
tail(train)

vec = c(6, rep(5, 200))
spl = split(train,rep(1:201,vec))
splvar = lapply(spl, VaR)
sec_var = do.call(rbind.data.frame, splvar)
sec_var1 = as.data.frame(apply(sec_var, 2, Lag, k=1))[-1,] # capital Lag lags the data forward
head(sec_var1)
head(data.frame(spy_var, sec_var1))




