data<-read.table("C:/Users/Administrator/Desktop/dissertation/HPI_PO_monthly_hist.csv", sep=",", header=T)
datafit<-data.frame(data[1:324,])
datatest<-data.frame(data[325:341,])
#取log 想解决残差的方差齐性问题
ussafit<-ts(log(datafit$USA),frequency = 12,start=c(1991,1))
plot(ussafit)
ussatest<-ts(datatest$USA,frequency = 12,start = c(2018,1))
plot(usnafit)#############与上图变成一图
#有序列图可知不平稳，递增
#检验平稳性方法1.时序图2》自相关图3.单位根检验 (unit root test)
#调整平稳性
ussafit.dif<-diff(diff(ussafit),12)
plot(ussafit.dif)
acf(ussafit.dif)# 不通过，长期在零轴一边，具有单调趋势。
pacf(ussafit.dif)
ussafit.dif2<-diff(ussafit.dif)

plot(ussafit.dif2)
acf(ussafit.dif2)#平稳性通过
pacf(ussafit.dif2)
#白噪声检验
#for(i in 1:6) print(Box.test(ussafit.dif2 ,type="Ljung-Box",lag=2*i))#非白噪声序列


#模型拟合
library(zoo)
library(forecast)
arimasafit<-auto.arima(ussafit)
summary(arimasafit)
#检验残差白噪声
tsdiag(arimasafit)
cpgram(arimasafit$residuals)#通过
#yuce
arimasa.fitfore <-forecast(arimasafit ,h=40)
plot(arimasa.fitfore)
lines(ussatest,col="red")
legend(1993,280, c("test", "forcasts"),lty=c(1,1.5), col=c("red","blue"))
#检验残差方差齐性
library(FinTS)
library(tseries)
library(fGarch)
library(rugarch)#garch拟合与预测
library(TSA)#BIC准则确定arma阶数  eacf确定garch阶数
library(ggplot2)#绘图
#library(ccgarch)#JB统计量
library(quantmod)
#LM检验
par(mfrow=c(1,1))
title(main = "LM-Test")
lmresult=McLeod.Li.test(y=residuals(arimasafit))
lmresult=McLeod.Li.test(y=residuals(arimasafit))
#两个检验都不通过，所以残差方差齐性不成立，试对残差拟合garch模型

r.fit<-garch(arimasafit$residual,order=c(1, 1))
summary(r.fit)
r.pred<-predict(r.fit)
plot (r.pred)

?return

