#set's working directory to current folder
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#goal: use the previous 5 days of VaR of each sector to predict the next 
#5 day VaR of the market

library(PerformanceAnalytics)
library(quantmod)

train = read.csv("train_data.csv", row.names = "X")
tail(train)

vec = c(6, rep(5, 200))
spl = split(train,rep(1:201,vec))
splvar = lapply(spl, VaR)
sec_var = do.call(rbind.data.frame, splvar)
spy_var = sec_var[-1,1]
sec_var = sec_var[,-1] #gets rid of SPY
sec_var1 = as.data.frame(apply(sec_var, 2, Lag, k=1))[-1,] # capital Lag lags the data forward
head(sec_var1)
data = data.frame(spy_var, sec_var1)

########################################Data Exploration and Plotting
library(ggplot2)
library(reshape2)
data['index'] = 1:200
df <- melt(data, id.vars  = 'index', variable.name = 'series')

#create line plot for each column in data frame
ggplot(df, aes(index, value)) +
  geom_line(aes(colour = series)) + 
  ggtitle("Value at Risk Over Time")



#training data plot 
train['index'] = as.Date(rownames(train))
head(train)
df <- melt(train, id.vars  = 'index', variable.name = 'series')

train = xts(train[, -which(names(train) == 'index')], order.by = as.POSIXct(train$index))
chart.TimeSeries(train, legend.loc = "bottomright", main = "Summary of Returns")

ggplot(df, aes(index, value)) +
  geom_line(aes(colour = series))

as.POSIXct(train$index)



##############################################Linear Model

data  = na.exclude(data)

trainsub  = sample(nrow(data),0.8*nrow(data),replace=FALSE)

linmod = glm(spy_var~., data=data, subset = trainsub)
summary(linmod)

linmod.train.mse = mean((data$spy_var[trainsub] - predict(linmod, data[trainsub,]))^2) #train MSE
linmod.train.mse

pred = predict(linmod, data[-trainsub,])
ar2.mse = mean((pred-data$spy_var[-trainsub])^2)
ar2.mse


################################### Random Forest
library(randomForest)

rf.reg = randomForest(spy_var~.,data=data, subset = trainsub, 
                      ntree=100,mtry=3,importance=TRUE, na.action = na.roughfix) #mtry is number of variables 
rf.reg #Gives both the number of variables at each split but also provides out-of-bag estimate of error rate

rf.train.mse = mean((data$spy_var[trainsub] - predict(rf.reg, data[trainsub,]))^2) #TRAIN MSE
rf.train.mse

spyvar.rf.pred = predict(rf.reg, data[-trainsub,]) # Predict with bagging
rf.MSE = mean((data$spy_var[-trainsub] - spyvar.rf.pred)^2)
rf.MSE #Test MSE

rf.reg$importance # MeanDecreaseGini is MDI
# MeanDecreaseAccuracy is MDA (based on bagging errors)
varImpPlot(rf.reg)

####################################principla component analysis (data exploration)
prdata = data[-1] #without direction
pr.out=prcomp(prdata,scale=TRUE) #Make sure you scale the vectors for principal component analysis

pr.var=pr.out$sdev^2 #Variance explained

pve=pr.var/sum(pr.var) #Proportion of variance explained
pve

plot(cumsum(pve),xlab="Principal Component",ylab="Cumulative Proportion",ylim=c(0,1),type='b', main = "Proportion of Variance Explained")
abline(h=0.8, col = "red")

pr.out$rotation
biplot(pr.out,scale=0)

       