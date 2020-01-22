library(hydroGOF)
library(urca)
library(tseries)
library(fGarch)
library(rugarch)#garch
library(TSA)#
library(ggplot2)#
#library(ccgarch)#JB
library(quantmod)
library(zoo)
library(forecast)
library(zoo)
library(forecast)
data<-read.table("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv", sep=",", header=T)
ts.data <- ts(as.matrix(data[,c(2:11)]), start=c(1990,1), frequency=12)
datafit<-data.frame(data[1:329,])
ts.datafit <- ts(as.matrix(datafit[,c(2:11)]), start=c(1990,1), frequency=12)
ussafit<-ts(datafit$USA..,frequency = 12,start=c(1991,1))
par(mfrow=c(1,1))
plot(ussafit,ylab='USA House Price')


first_order_difference<-diff(diff(ussafit),12)
par(mfrow=c(3,1))
plot(first_order_difference)
acf(first_order_difference)
pacf(first_order_difference)# 不通过，长期在零轴一边，具有单调趋势。
main

second_order_difference<-diff(first_order_difference)
summary(ur.df(ussafit,type = 'trend',selectlags = 'AIC'))#
summary(ur.df(ussafit,type ='drift' ,selectlags = 'AIC'))#
summary(ur.df(ussafit,type = 'none',selectlags = 'AIC'))#
par(mfrow=c(3,1))
plot(second_order_difference)
acf(second_order_difference)#
pacf(second_order_difference)
#Q TEST
for(i in 1:6) print(Box.test(ussafit.dif2 ,type="Ljung-Box",lag=2*i))#

#autoarima
arimasafit<-auto.arima(ussafit)
par(mfrow=c(1,1))
title(main = "LM-Test")
lmresult=McLeod.Li.test(y=residuals(arimasafit))

#fit
arimasa.fit<-arima(ussafit,order=c(0,2,1),seasonal=list(order=c(0,0,2),period=12))
#diagnose
acf(arimasa.fit$residual)
tsdiag(arimasa.fit)
cpgram(arimasa.fit$residuals)

#forecast
par(mfrow=c(1,1))
arimasa.fitfore <-forecast(arimasa.fit ,h=12)
plot(arimasa.fitfore)
pre<-arimasa.fitfore$mean
test<-ts.data[330:341]
sqrt(mse(sim = pre, obs = test, weights = 1, na.rm = FALSE))

#normalized 
myNormalize <- function (target) {
  (target - min(target))/(max(target) - min(target))}
nomorized<-myNormalize(ussafit)
arimasafitn<-auto.arima(nomorized)
arimasa.fitforen <-forecast(arimasafitn ,h=12)
plot(arimasa.fitforen)
arimasa.fitforen$mean
