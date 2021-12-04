# Univariate GARCH
library(quantmod)
library(rugarch)
library(zoo)

from_train = "2007-01-01"
to_train = "2010-12-31"
symbol = "SPY"

from_test = "2019-01-01"
to_test = "2021-11-20"

#Get Data from Yahoo
TS <- getSymbols("SPY", from = from_train, to = to_train, auto.assign = F)
TS_test = getSymbols(symbol, from = from_test,to =  to_test, auto.assign = F)
SMI_test = TS_test[,ncol(TS_test)]
SMI <- TS[,ncol(TS)]
SMI <- dailyReturn(SMI, method = "log")
SMI_test <- dailyReturn(SMI_test, method = "log")

#Plot SMI Returns
plot.zoo(TS$SPY.Adjusted)

SMI = as.timeSeries(SMI)

histPlot(SMI, main = "SMI Returns")

SMIdf <- as.data.frame(SMI)

hv <- rollapply(SMI_test, 20, sd)

l = length(hv)
hv = hv[200:l]


#garch model
gspec11 <- ugarchspec(variance.model = list(model = "sGARCH", 
                                            garchOrder = c(1, 1)),
                      mean.model=list(armaOrder=c(0,0), 
                                      include.mean = FALSE), 
                       distribution="norm")


#Rolling Estimation
roll11 <- ugarchroll(gspec11, SMI_testdf, n.start=200,
                     refit.every = 25, refit.window = "moving",
                     VaR.alpha = c(0.025, 0.05))

fit = ugarchfit(gspec11, data = SMI_testdf, VaR.alpha = c(0.025, 0.05))
show(fit)

?ugarchroll

SMI_testdf = as.data.frame(SMI_test)

SMI_test = SMI_test[200:l]

VaRStatic <- sd(SMI_test) * qnorm(0.05) #Static, unconditional
VaRMA <- hv * qnorm(0.05)                       #Moving Average
VaRGARCH <- roll11@forecast$VaR[,2]
returns = roll11@forecast$VaR[,3]
head(VaRGARCH)
tail(VaRGARCH)

summary(roll11@model)

show(roll11)

xaxis = rownames(SMI_testdf)

xaxis <- rownames(roll11@forecast$VaR)
xaxis <- c(xaxis[1], xaxis[100], xaxis[200], xaxis[300], xaxis[400], xaxis[500])


plot(1:length(returns), returns, type = "l", pch = 16, cex = 0.8,  col = gray(0.2, 0.5),
     ylab = "Returns", main = "95% VaR Forecasting", xaxt = "n", xlab = "")
axis(1, at=c(1, 100, 200, 300, 400, 500), labels=xaxis)
lines(1:length(VaRGARCH), VaRGARCH, col = "black")
lines(as.numeric(VaRMA[,1]), col = 4)
abline(h=VaRStatic, col = 2)
legend('topright', c("GARCH(1,1)", "MA 20 days", "Static VaR") , 
       lty=1, col=c(1,4,2), bty='n', cex=.75)



length(xaxis)
length(VaRGARCH)

plot(SMI_test)
lines(VaRGARCH, col = 4)
lines(VaRMA, col = 4)
