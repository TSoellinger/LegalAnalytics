LegalAnalytics
==============
## __I. Introduction__
### __Problem:__ Wine Classifier _In_,An Introduction to Machine Learning (with Applications in R) By Michael Clark, Notre Dame -- http://www3.nd.edu/~mclark19/learn/ML.pdf
### __Due:__ 2/28/14 at 5:00 p.m.
### __Author:__ Tyler Soellinger
### __Data Used:__ Machine Learning Repository, http://archive.ics.uci.edu/ml/datasets/Wine+Quality 
## __II. Code__
### __A. Load Libraries__  
```{r message=FALSE}
    library(pROC)
    library(class)
    library(lattice)
    library(ggplot2)
    library(caret)
    library(corrplot)
    library(e1071)
```    
### __B. Load And Preview Data Set__  
````{r}    
    wine <- read.csv("http://www.nd.edu/~mclark19/learn/data/goodwine.csv")
    summary(wine)
```                 
#### _Correlation Matrix Display_
```{r}
    corrplot(cor(wine[, -c(13, 15)]), method = "number", tl.cex = 0.5)
``` 
### __C. Partition 20% Of The Data For Cross Validation Test__ 
#### _First, Set Indices To Be The Same When Re-Run_
```{r}    
    set.seed(1234)
```
#### _Then, Separate The Data Into 80% Training Set And 20% Test Set_
```{r}
    trainIndices = createDataPartition(wine$good, p = 0.8, list = F)
    wanted = !colnames(wine) %in% c("free.sulfur.dioxide", "density", "quality", "color", "white")
    wine_train = wine[trainIndices, wanted]
    wine_test = wine[-trainIndices, wanted]
```
### __D. Normalize And Scale The Data Sets For Preliminary Display. This Does NOT Account For Interaction Effects, It Is Merely A "Peak" At The Data__
```{r}
    wine_trainplot = predict(preProcess(wine_train[, -10], method="range"), wine_train[, -10])
    featurePlot(wine_trainplot, wine_train$good, "box")
```

### __E. Tune The Model With The Training Set Data Using 10-fold Cross Validation__
```{r}
    set.seed(1234)
    cv_opts = trainControl(method="cv", number=10)
    knn_opts = data.frame(.k=c(seq(3, 11, 2), 25, 51, 101))
    results_knn = train(good~., data=wine_train, method="knn", preProcess="range", trControl=cv_opts, tuneGrid = knn_opts)
 
    results_knn
```

### __F. Test The Model Against The Test Set Data__
```{r}
 preds_knn = predict(results_knn, wine_test[,-10])
 confusionMatrix(preds_knn, wine_test[,10], positive='Good')
```
### __G. Visualy Display The Results For Evaluation__

```{r}
 dotPlot(varImp(results_knn))
```
