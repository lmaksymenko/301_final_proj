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
test = read.csv("test_data.csv", row.names = "X")

#tail(train)

vec = c(6, rep(3, 333)) #1006
spl = split(train, rep(1:201,vec))
splvar = lapply(spl, VaR)
sec_var = do.call(rbind.data.frame, splvar)
spy_var_train = sec_var[-1,1]
sec_var = sec_var[,-1] #gets rid of SPY
sec_var1_train = as.data.frame(apply(sec_var, 2, Lag, k=1))[-1,] # capital Lag lags the data forward

vec = c(8, rep(5, 720/5)) #728 rows
spl = split(test, rep(1:(1+720/5),vec))
splvar = lapply(spl, VaR)
sec_var = do.call(rbind.data.frame, splvar)
spy_var_test = sec_var[-1,1]
sec_var = sec_var[,-1] #gets rid of SPY
sec_var1_test = as.data.frame(apply(sec_var, 2, Lag, k=1))[-1,]

#head(sec_var1)

data_train = data.frame(spy_var_train, sec_var1_train)
data_test = data.frame(spy_var_test, sec_var1_test)

data_train = na.omit(data_train)
data_test = na.omit(data_test)

data_train['index'] = 1:nrow(data_train)
data_test['index'] = 1:nrow(data_test)

head(data_test)


spy_uni_train = as.data.frame(na.exclude(cbind(spy_var_train, Lag(spy_var_train, k=1))))
spy_uni_test = as.data.frame(na.exclude(cbind(spy_var_test, Lag(spy_var_test, k=1))))

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
data_test = na.exclude(data_test)
data_train = na.exclude(data_train)

###LINEAR MODEL

linmod = glm(spy_var_train~., data=data_train)

linmod.train.mse = mean((data_train$spy_var_train - predict(linmod, data_train))^2)

pred = predict(linmod, data_test)
linmod.test.mse = mean((pred - data_test$spy_var_test)^2)


#Results

cat('LINEAR MODEL \n',
    'Summary: \n', 
    strrep('-', 80),
    sapply(capture.output(summary(linmod)), function(x) paste(x, '\n')), '\n',
    strrep('-', 80), '\n', 
    'Train MSE: ', linmod.train.mse, '\n',
    'Test MSE: ', linmod.test.mse, '\n')


plot(data_test$spy_var_test, type = "l", main = "LINEAR MODEL MultiVar VaR Predicted vs Empirical",
     ylab = "VaR")
lines(pred, lty = 2, col = "blue")
legend(80, y = -0.05, c("Actual", "Predicted"), lty = c(1,2), col = c("black", "blue"))

names(data_train)



######### Univariate LM 

linmod = glm(spy_var_train~., data=spy_uni_train)

linmod.train.mse = mean((spy_uni_train$spy_var_train - predict(linmod, spy_uni_train))^2)

pred = predict(linmod, spy_uni_test)
linmod.test.mse = mean((pred - spy_uni_test$spy_var_test)^2)


plot(spy_uni_test$spy_var_test, type = "l", main = "LINEAR MODEL UniVar VaR Predicted vs Empirical",
     ylab = "VaR")
lines(pred, lty = 2, col = "blue")
legend(80, y = -0.05, c("Actual", "Predicted"), lty = c(1,2), col = c("black", "blue"))



###RANDOM FOREST

rf.reg = randomForest(spy_var_train~.,data=data_train, 
                      ntree=100,mtry=3,importance=TRUE, na.action = na.roughfix) #mtry is number of variables 

rf.train.mse = mean((data_train$spy_var_train - predict(rf.reg, data_train))^2)

spyvar.rf.pred = predict(rf.reg, data_test) # Predict with bagging
rf.test.mes = mean((data_test$spy_var_test - spyvar.rf.pred)^2)

#Results

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

plot(data_test$spy_var_test, type = "l", main = "RANDOM FOREST MultiVar VaR Predicted vs Empirical",
     ylab = "VaR")
lines(spyvar.rf.pred, lty = 2, col = "blue")
legend(80, y = -0.05, c("Actual", "Predicted"), lty = c(1,2), col = c("black", "blue"))



#####Uni variate RF model 

rf.reg = randomForest(spy_var_train~.,data=spy_uni_train, 
                      ntree=100,mtry=1,importance=TRUE, na.action = na.roughfix) #mtry is number of variables 

rf.train.mse = mean((spy_uni_train$spy_var_train - predict(rf.reg, spy_uni_train))^2)

spyvar.rf.pred = predict(rf.reg, spy_uni_test) # Predict with bagging
rf.test.mes = mean((spy_uni_test$spy_var_test - spyvar.rf.pred)^2)

plot(spy_uni_test$spy_var_test, type = "l", main = "RANDOM FOREST UniVar VaR Predicted vs Empirical",
     ylab = "VaR")
lines(spyvar.rf.pred, lty = 2, col = "blue")
legend(80, y = -0.05, c("Actual", "Predicted"), lty = c(1,2), col = c("black", "blue"))




###NEURON ACTIVATION NETWORKS
#https://www.youtube.com/watch?reload=9&v=roUIWGr9rqo

###All Network Params
HIDDEN_SIZE <- 5
BATCH_SIZE <- 10
LAYERS <- 10

x_train = as.matrix(subset(data_train, select = -c(spy_var_train, index)))
x_train_u = x_train[,1]
x_test = as.matrix(subset(data_test, select = -c(spy_var_test, index)))
x_test_u = x_test[,1]
y_train = as.matrix(subset(data_train, select = c(spy_var_train)))

#LSTM specific
x_train_vec = array(data = x_train, dim = c(nrow(x_train), 1, 11))
x_test_vec = array(data = x_test, dim = c(nrow(x_test), 1, 11))




##NN UNIVAR
#Build Model
model <- keras_model_sequential() 

model %>%
  layer_dense(HIDDEN_SIZE, activation = 'linear', input_shape = ncol(x_train_u))

  for(i in 1:LAYERS){
    model %>% layer_dense(HIDDEN_SIZE, activation = 'tanh')
  }

  model %>% layer_dense(1, activation = 'linear')

model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam", 
  metrics = "accuracy"
)

model %>% fit( 
  x = x_train_u, 
  y = y_train, 
  batch_size = BATCH_SIZE, 
  epochs = 15
)

result <- predict(model, x_test_u)

nn.train.mse = mean((data_train[,'spy_var_train'] - predict(model, x_train_u)) ^ 2) 
nn.train.mse

nn.test.mse = mean((data_test[,'spy_var_test'] - result) ^ 2) 
nn.test.mse

#Results

plot(-data_test$spy_var_test, type = "l", main = "Univariate NN VaR Predicted vs Empirical",
     ylab = "VaR")
lines(-result, lty = 2, col = "blue")
legend(80, y = 0.06, c("Actual", "Predicted"), lty = c(1,2), col = c("black", "blue"))

summary(model)

model

save_model_weights_hdf5(model, 'nnmodel')
get_weights(model)

#> nn.train.mse
#[1] 0.0002585674
#> nn.test.mse
#[1] 0.0002378646



##NN MULTIVAR

#Build Model
model <- keras_model_sequential() 

model %>%
  layer_dense(HIDDEN_SIZE, activation = 'linear', input_shape = ncol(x_train))
  
  for(i in 1:LAYERS){
    model %>% layer_dense(HIDDEN_SIZE, activation = 'tanh')
  }
    
  model %>% layer_dense(1, activation = 'linear')
  
model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam", 
  metrics = "accuracy"
)

model %>% fit( 
  x = x_train, 
  y = y_train, 
  batch_size = BATCH_SIZE, 
  epochs = 15
)

result <- predict(model, x_test)

nn.train.mse = mean((data_train[,'spy_var_train'] - predict(model, x_train)) ^ 2) 
nn.train.mse

nn.test.mse = mean((data_test[,'spy_var_test'] - result) ^ 2) 
nn.test.mse

summary(model)
#Results

plot(-data_test$spy_var_test, type = "l", main = "Multivariate NN VaR Predicted vs Empirical",
     ylab = "VaR")
lines(-result, lty = 2, col = "blue")
legend(70, y = 0.10, c("Actual", "Predicted"), lty = c(1,2), col = c("black", "blue"))
text(80, y = 0.15, paste("Test mse: ", signif(nn.test.mse), '\n', 'Train mse: ',signif(nn.train.mse)))


#nn.train.mse
#[1] 0.0002418074
#> nn.test.mse
#[1] 0.0002378646


##LSTM Uni

#Build Model


x_train_vec = array(data = x_train_u, dim = c(nrow(x_train), 10, 1))
x_test_vec = array(data = x_test_u, dim = c(nrow(x_test), 10, 1))

x_train_vec = head( x_train_vec, -3)

#Build Model
model <- keras_model_sequential() 

model %>%
  layer_lstm(units = 5, activation = 'tanh', batch_input_shape = c(10, 10, 1), return_sequences = T, stateful = T)

for(i in 1:2){
  model %>% layer_lstm(units = 5, activation = 'tanh', return_sequences = T, stateful = T)
}

model %>% layer_lstm(units = 1, activation = 'tanh') #%>% 
#model %>%layer_dense(1, activation = 'linear')

model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam", 
  metrics = "accuracy"
)

#summary(model)

model %>% fit( 
  x = x_train_vec, 
  y = head(y_train, -3), 
  batch_size = 10, 
  epochs = 50
)


result <- predict(model, head(x_test_vec, -1), batch_size = 10)
result

nn.train.mse = mean((head(data_train[,'spy_var_train'], -3) - predict(model, x_train_vec, batch_size = 10)) ^ 2) 
nn.train.mse

nn.test.mse = mean((head(data_test[,'spy_var_test'], -1) - result) ^ 2) 
nn.test.mse


plot(-data_test$spy_var_test, type = "l", main = "LSTM Univariate VaR Predicted vs Empirical",
     ylab = "VaR")
lines(-result, lty = 2, col = "blue")
legend(80, y = 0.07, c("Actual", "Predicted"), lty = c(1,2), col = c("black", "blue"))

#> nn.train.mse
#[1] 0.0002666164

#> nn.test.mse
#[1] 0.0002804499



##LSTM Multi


x_train_vec = array(data = x_train, dim = c(nrow(x_train), 10, 11))
x_test_vec = array(data = x_test, dim = c(nrow(x_test), 10, 11))

x_train_vec = head( x_train_vec, -3)

#Build Model
model <- keras_model_sequential() 

model %>%
  layer_lstm(units = 5, activation = 'tanh', batch_input_shape = c(10, 10, 11), return_sequences = T, stateful = T)

for(i in 1:2){
  model %>% layer_lstm(units = 5, activation = 'tanh', return_sequences = T, stateful = T)
}

model %>% layer_lstm(units = 1, activation = 'tanh') #%>% 
#model %>%layer_dense(1, activation = 'linear')

model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam", 
  metrics = "accuracy"
)

summary(model)

model %>% fit( 
  x = x_train_vec, 
  y = head(y_train, -3), 
  batch_size = 10, 
  epochs = 50
)


model.get()


result <- predict(model, head(x_test_vec, -1), batch_size = 10)
result

nn.train.mse = mean((head(data_train[,'spy_var_train'], -3) - predict(model, x_train_vec, batch_size = 1)) ^ 2) 
nn.train.mse

nn.test.mse = mean((head(data_test[,'spy_var_test'], -1) - result) ^ 2) 
nn.test.mse

#Results

plot(-data_test$spy_var_test, type = "l", main = "LSTM Multivariate VaR Predicted vs Empirical",
     ylab = "VaR")
lines(-result, lty = 2, col = "blue")
legend(80, y = 0.07, c("Actual", "Predicted"), lty = c(1,2), col = c("black", "blue"))

summary(model)
#> nn.train.mse
#[1] 0.0002463397

#> nn.test.mse
#[1]  0.000262263



###DCC-GARCH
x_train = as.matrix(subset(data_train, select = -c(index)))#, byrow = TRUE)
y_train = as.matrix(subset(data_train, select = c(spy_var_train)))


uspec = ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(1,1)),
                  mean.model = list(armaOrder = c(1, 0)))

mspec = multispec(replicate(12, uspec))
dspec = dccspec(mspec)

#dat = dccfilter(dspec, data = subset(data_train, select = -c(spy_var_train, index)))

dcc = dccfit(dspec, data = x_train)

roll = dccroll(dspec, x_train, 
        n.ahead = 0, 
        refit.window = 'moving')#, window.size = 25)

show(roll)
roll@mforecast

result = dccforecast(dcc, n.ahead = 1)
result@mforecast

#result = ugarchforecast()

dspec
??dccroll

plot(result, which = 2, series=c(1,2))



###TESTING PORITON

##LSTM Multi


x_train_vec = array(data = x_train_u, dim = c(nrow(x_train), 10, 1))
x_test_vec = array(data = x_test_u, dim = c(nrow(x_test), 10, 1))

x_train_vec = head( x_train_vec, -3)

#Build Model
model <- keras_model_sequential() 

model %>%
  layer_lstm(units = 5, activation = 'tanh', batch_input_shape = c(10, 10, 1), return_sequences = T, stateful = T)
  
  for(i in 1:2){
   model %>% layer_lstm(units = 5, activation = 'tanh', return_sequences = T, stateful = T)
  }
  
  model %>% layer_lstm(units = 1, activation = 'tanh') #%>% 
    #model %>%layer_dense(1, activation = 'linear')

model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam", 
  metrics = "accuracy"
)

#summary(model)

model %>% fit( 
  x = x_train_vec, 
  y = head(y_train, -3), 
  batch_size = 10, 
  epochs = 50
)


result <- predict(model, head(x_test_vec, -1), batch_size = 10)
result

nn.train.mse = mean((head(data_train[,'spy_var_train'], -3) - predict(model, x_train_vec, batch_size = 10)) ^ 2) 
nn.train.mse

nn.test.mse = mean((head(data_test[,'spy_var_test'], -1) - result) ^ 2) 
nn.test.mse


plot(-data_test$spy_var_test, type = "l", main = "LSTM Multivariate VaR Predicted vs Empirical",
     ylab = "VaR")
lines(-result, lty = 2, col = "blue")
legend(80, y = 0.07, c("Actual", "Predicted"), lty = c(1,2), col = c("black", "blue"))

