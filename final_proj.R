#set's working directory to current folder
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#goal: use the previous 5 days of VaR of each sector to predict the next 
#5 day VaR of the market


###IMPORTS (not exports)
library(PerformanceAnalytics)
library(quantmod)
library(randomForest)
library(ggplot2)
library(reshape2)
library(keras)
library(rmgarch)

### ### SETUP ### ###

train = read.csv("train_data.csv", row.names = "X")

#tail(train)

vec = c(6, rep(5, 200))
spl = split(train, rep(1:201,vec))
splvar = lapply(spl, VaR)
sec_var = do.call(rbind.data.frame, splvar)
spy_var = sec_var[-1,1]
sec_var = sec_var[,-1] #gets rid of SPY
sec_var1 = as.data.frame(apply(sec_var, 2, Lag, k=1))[-1,] # capital Lag lags the data forward

#head(sec_var1)

data = data.frame(spy_var, sec_var1)
data['index'] = 1:200



### ### EDA ### ###

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



###PCA (data exploration)

data = data[, -which(names(data) == 'index')]


prdata = data[-1] #without SPY
pr.out=prcomp(prdata,scale=TRUE) #Make sure you scale the vectors for principal component analysis

pr.var=pr.out$sdev^2 #Variance explained

pve=pr.var/sum(pr.var) #Proportion of variance explained
pve

plot(cumsum(pve),xlab="Principal Component",ylab="Cumulative Proportion",ylim=c(0,1),type='b', main = "Proportion of Variance Explained")
abline(h=0.8, col = "red")

pr.out$rotation
biplot(pr.out,scale=0)



### ### MODELING ### ###

###TRAIN TEST SPLIT
data  = na.exclude(data)
trainsub  = sample(nrow(data),0.8*nrow(data),replace=FALSE)


###LINEAR MODEL

linmod = glm(spy_var~., data=data, subset = trainsub)

linmod.train.mse = mean((data$spy_var[trainsub] - predict(linmod, data[trainsub,]))^2)

pred = predict(linmod, data[-trainsub,])
linmod.test.mse = mean((pred - data$spy_var[-trainsub])^2)


cat('LINEAR MODEL \n',
    'Summary: \n', 
    strrep('-', 80),
    sapply(capture.output(summary(linmod)), function(x) paste(x, '\n')), '\n',
    strrep('-', 80), '\n', 
    'Train MSE: ', linmod.train.mse, '\n',
    'Test MSE: ', linmod.test.mse, '\n')


###RANDOM FOREST

rf.reg = randomForest(spy_var~.,data=data, subset = trainsub, 
                      ntree=100,mtry=3,importance=TRUE, na.action = na.roughfix) #mtry is number of variables 

rf.train.mse = mean((data$spy_var[trainsub] - predict(rf.reg, data[trainsub,]))^2)

spyvar.rf.pred = predict(rf.reg, data[-trainsub,]) # Predict with bagging
rf.test.mes = mean((data$spy_var[-trainsub] - spyvar.rf.pred)^2)


cat('RANDOM FOREST MODEL \n',
    'Summary: \n', 
    strrep('-', 80),
    sapply(capture.output(rf.reg), function(x) paste(x, '\n')), '\n',
    strrep('-', 80), '\n', 
    'Importance: \n', 
    strrep('-', 80), '\n',
    #Gives both the number of variables at each split but also provides 
    #out-of-bag estimate of error rate
    # MeanDecreaseGini is MDI
    # MeanDecreaseAccuracy is MDA (based on bagging errors)
    sapply(capture.output(rf.reg$importance), function(x) paste(x, '\n')), '\n',
    strrep('-', 80), '\n', 
    'Train MSE: ', rf.train.mse, '\n',
    'Test MSE: ', rf.test.mes, '\n')

varImpPlot(rf.reg)

###NEURON ACTIVATION NETWORK
#https://www.youtube.com/watch?reload=9&v=roUIWGr9rqo

#Params
HIDDEN_SIZE <- 128
BATCH_SIZE <- 128
LAYERS <- 2

x_train = as.matrix(subset(data[trainsub,], select = -c(spy_var)))
y_train = as.matrix(data[trainsub, "spy_var"])

#Build Model
model <- keras_model_sequential() 

model %>%
  #layer_lstm(HIDDEN_SIZE, input_shape=c(MAXLEN, length(char_table))) %>%
  layer_dense(HIDDEN_SIZE, activation = 'relu', input_shape = c(11))
  
  for(i in 1:LAYERS){
    model %>% layer_dense(HIDDEN_SIZE, activation = 'relu')
  }
    
  model %>% layer_dense(1, activation = 'relu')
  
  
  
model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = "adam", 
  metrics = "accuracy"
)
###

model %>% fit( 
  x = x_train, 
  y = y_train, 
  batch_size = BATCH_SIZE, 
  epochs = 70
)

x_test = as.matrix(subset(data[-trainsub,], select = -c(spy_var)))

result <- predict(model, x_test)



