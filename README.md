# GSoC 2021 Report - Machine Learning for Macro Diffusion Indexes, The R Project

<img src="https://user-images.githubusercontent.com/56316487/130404951-c2840ba9-a6b7-4487-952e-f322fb761b1d.png" width="100" height="100">  <img src="https://user-images.githubusercontent.com/56316487/130405276-cd1b0d26-017f-49cd-84d9-343c3324c95e.png" width="100" height="100">


## Introduction

Macroeconomics (from the Greek prefix makro- meaning "large" + economics) is a branch of economics dealing with the performance, structure, behavior, and decision-making of an economy as a whole. For example, using interest rates, taxes and government spending to regulate an economy's growth and stability.

The main goal of the project is to help macroeconomists to obtain useful insights from a dataset as a whole with the help of useful Machine Learning Algorithms by creating it's potential diffusion indexes.
Time series analysis is an area of statistics that focuses on analyzing time-dependent data. Time series can be analyzed either descriptively or inferentially. This has led to different approaches depending on the type of information available in time-series data.

Diffusion indexes are a way to frame and determine what the key market drivers may be, but more importantly allows a strategist to be current on a number of different subjects and communicate on those subjects effectively with investors or other stakeholders.

The Project Machine Learning for Macro Diffusion Indexes aims on creating series of potentially useful diffusion indexes and the data that may be used to construct them. Then applying random forest and/or other appropriate machine learning techniques to the data, with the goal of demonstrating the relative performance of those methods.

__We have used the method proposed by Stock and Watson in their paper called "Macroeconomic Forecasting Using Diffusion Indexes" published in 2002 for generating the forecasts initially and later proceed with the Machine Learning approach.__


## Objectives

* Research,Decide and Discuss about Data and it's sources                   ☑
* Parse Data available from FRED                                            ☑
* Construct a loop to parse CSV data and format it                          ☑
* Scale and Normalize the Data                                              ☑
* Build function to create Diffusion Indexes (Dimensionality Reduction)     ☑
* Obtain Visualisations of Data                                             ☑
* Parse, Aliign and format Output Data                                      ☑
* Train and Test Machine Learning Algorithms to obtain Forecasts            ☑
* Compare the Relative Performances                                         ☑


## Approach

We started by calling the required R libraries for Data parsing, processing and manipulation.
Here we have used the library, "quantmod" developed by [Joshua Ulrich](https://github.com/joshuaulrich).

Federal Reserve Economic Data (FRED) is an online database consisting of hundreds of thousands of economic data time series from scores of national, international, public, and private sources. FRED, created and maintained by the Research Department at the Federal Reserve Bank of St. Louis, goes far beyond simply providing data: It combines data with a powerful mix of tools that help the user understand, interact with, display, and disseminate the data. In essence, FRED helps users tell their data stories. 

First we created a new environment and a vector object containing the specific names(symbols) of the dataset we will be using.

We have identified around 70+ datasets for our forecasts which we have divided into three main categories i.e Fundamental, Behavioral and Catalyst.

We use 'getSymbols' function to get data from FRED.

```
getSymbols(Symbols = symbols1,
           src='FRED',
           env = fundamental_data)

getSymbols(Symbols = symbols2,
           src='FRED',
           env = behavioral_data)

getSymbols(Symbols = symbols3,
           src='FRED',
           env = catalyst_data)
```           
        
We also used some *CSV* datasets in out forecasts, But you can use datasets of your choise for building DI.

After going through some thorough Data Processing we finally use our modified SWfore function from MTS package to build Diffusion Indexes.

```
## Using SWFore to build Diffusion Indexes - Function
"DiffIdx" <- function(x,orig,m){
  ### Builds Stock and Watson's diffusion index prediction
  ### x: observed regressors
  ### orig: forecast origin
  ### m: selected number of PCs
  ###
  ### Output: Forecasts and MSE of forecasts (if data available)
  if(!is.matrix(x))x=as.matrix(x)
  nT=dim(x)[1]
  k=dim(x)[2]
  if(orig > nT)orig=nT
  if(m > k)m=k; if(m < 1)m=1
  # standardize the predictors
  x1=x[1:orig,]
  me=apply(x1,2,mean)
  se=sqrt(apply(x1,2,var))
  x1=x
  for (i in 1:k){
    x1[,i]=(x1[,i]-me[i])/se[i]
  }
  #
  V1=cov(x1[1:orig,])
  m1=eigen(V1)
  sdev=m1$values
  M=m1$vectors
  M1=M[,1:m]
  Dindex=x1%*%M1
  # y1=y[1:orig]
  # DF=Dindex[1:orig,]
  DF=Dindex
  # mm=lm(y1~DF)
  # coef=matrix(mm$coefficients,(m+1),1)
  # coef=matrix(mm$coefficients[-1],(m),1) # exclude the intercept
  #cat("coefficients: ","\n")
  #print(round(coef,4))
  yhat=NULL; MSE=NULL
  # if(orig < nT){
  #    newx=cbind(rep(1,(nT-orig)),Dindex[(orig+1):nT,])
  #    yhat=mm$coefficients[1]+(t(newx)%*%coef)
  #    err=y[(orig+1):nT]-yhat
  #    MSE=mean(err^2)
  #    cat("MSE of out-of-sample forecasts: ",MSE,"\n")
  # }

  DiffIdx <- xts(DF, index(x1))
}
```

Here are some visualizations,

<img src="https://user-images.githubusercontent.com/56316487/130423560-dcb1b87a-3649-4778-8e68-b5ba6f334648.png" width="600" height="300">  <img src="https://user-images.githubusercontent.com/56316487/130423900-63857598-8b33-4ecd-b2a8-b5b8ff021dd2.png" width="600" height="300"> 


We have chosen to make forecasts of S&P 500 from our Diffusion Indexes.
The Standard and Poor's 500, or simply the S&P 500, is a stock market index tracking the performance of 500 large companies listed on stock exchanges in the United States. It is one of the most commonly followed equity indices.

A quick scatter plot to check the Correlation between our Diffusion Indexes and the S&P 500 Data,


<img src="https://user-images.githubusercontent.com/56316487/130424466-598be8c4-c318-4e80-a82b-fef31fe768d7.png" width="700" height="350">


### Applying the Machine Learning Algorithm(s)

We have decided to use different ML Algo's to train and make forecasts, our primary algorithm as of now is Random Forest with approximately 72% accuracy. We further aim on improving the alogrithms and comparing the relative performances.

We have used the caret package (short for Classification And Regression Training) which contains functions to streamline the model training process for complex regression and classification problems.

Here we first load all the required packages then use one hot encoding and try to convert the S&P 500 dataset into a binary dataset. This allows us to successfully carry out our Random Forest classification or any other related methods. Then we split our data into training and testing with 80:20 ratio and train on the first 80% of our data with the remaining to test and make forecasts.

Below are some visualisations,

<img src="https://user-images.githubusercontent.com/56316487/130425316-11de36d0-0e7a-4eff-800c-1f8579756e23.png" width="700" height="350">

<img src="https://user-images.githubusercontent.com/56316487/130425445-c00c7add-dec2-4205-b295-0c55368ede7a.png" width="500" height="250">  <img src="https://user-images.githubusercontent.com/56316487/130425646-f15d048f-72bd-46bb-9946-e9fbdaf535a1.png" width="500" height="250">   <img src="https://user-images.githubusercontent.com/56316487/130425532-9c836b5b-b70d-413a-9e2e-63c005d50d14.png" width="500" height="250"> 


__This was a short description of the overall project, You can find the entire code in a form of vignette which you can regenerate and knit according to your needs(Click on the link below to access the repository containing the entire code)__


#### Link:

[https://github.com/Rishi0812/MacroDiffusionIndex/blob/main/vignettes/ML_for_MacroDiffusionIndexes.Rmd]


## Journey Ahead

There is always a room for improvement and refinement, Especially in Machine Learning projects there are plenty of things to try out and refine. Here are few todo's:

* Try forecasting the actual values of S&P 500 or any other Index through Regression
* Refine and tune the models for better performance
* Try out different ML Algorithms and improve accuracy

It was a sheer pleasure and an amazing learning experince contributing to The R Project of Statistical Computing, I will keep contributing to the Open Source Softwares and organisations even after the Google Summer of Code.


## Conclusion

Macroeconomic indicators are important to any trader because they can have a significant influence on market movements. This is why most fundamental analysis will incorporate macroeconomic indicators.

There is no way to be certain that these indicators are reliable on their own, but they do have a role in shaping the economy. Even if these indicators just influence other traders to open and close positions, this can be enough to create volatility in the market.
Market participants will be keeping an eye on analysts’ predictions of the data ahead of their release. The bigger the difference between the analysts’ predictions and the actual figure, the more volatility can be expected in financial markets – as positions are adjusted to reflect the actual figure.

Also the role of Machine Learning in macroeconomic analysis is huge and has an amazing impact on the real-world data. With further R&D of Machine Learning Algorithms there is always a space for improved accuracy of forecast.


## References

* Stock, James H, and Mark W Watson. “Macroeconomic Forecasting Using Diffusion Indexes.” Journal of Business & Economic Statistics 20, no. 2 (April 2002): 147–162. doi:10.1198/073500102317351921. [Available here](https://www.princeton.edu/~mwatson/papers/Stock_Watson_JBES_2002.pdf)

* quantmod: Quantitative Financial Modelling Framework. Specify, build, trade, and analyse quantitative financial trading strategies. [Available here](https://cran.r-project.org/web/packages/quantmod/index.html)

* MTS: All-Purpose Toolkit for Analyzing Multivariate Time Series (MTS) and Estimating Multivariate Volatility Models [Available here](https://cran.r-project.org/web/packages/MTS/index.html)

* A Short Introduction to the caret Package. [Available here](https://cran.r-project.org/web/packages/caret/vignettes/caret.html)
