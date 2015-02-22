# Practical Machine Learning Course Project
## Monika Chauhan

### Things to do
The project works on the Weight Lifting Exercise Dataset and the training and testing data is made available from various devices.   
The goal of your project is to predict the manner in which they did the exercise. This report contains information about:   
1. How I built model,   
2. How I used cross validation,   
3. What I think the expected out of sample error is,   
4. Why I made the choices you did. 


### Loading the data
```{r}
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
set.seed(12345)
        training <- read.csv("trainFilePath.csv", na.strings=c("NA","#DIV/0!",""))
        testing <- read.csv("testFilePath.csv", na.strings=c("NA","#DIV/0!",""))
```    

### Partitioning the training data for cross validation      
The training data set contains 53 variables and 19622 obs. The testing data set contains 53 variables and 20 obs.   
In order to perform cross-validation, the training data set is partionned into 2 sets: TrainData (60%) and TestData (40%).   
This will be performed using random subsampling without replacement.

```{r}
Train <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
TrainData <- training[Train, ]
TestData <- training[-Train, ]
dim(TrainData)
dim(TestData)
```    
### Training-Set cleaning 

#### 1. Cleaning NearZeroVariance Variables
```{r}
DataNZV <- nearZeroVar(TrainData, saveMetrics=TRUE)
NZVvars <- names(TrainData) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt",
"kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt",
"max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm",
"var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm",
"stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm",
"kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm",
"max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm",
"kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell",
"skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell",
"amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm",
"skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm",
"max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm",
"amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm",
"avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm",
"stddev_yaw_forearm", "var_yaw_forearm")
TrainData <- TrainData[!NZVvars]
dim(TrainData)
```   
#### 2. Killing first column of Dataset - ID Removing first ID variable so that it does not interfer with ML Algorithms:    
```{r}
TrainData <- TrainData[c(-1)]
```   
#### 3. Cleaning Variables with too many NAs  
```{r}
temp <- TrainData #creating another subset to iterate in loop
for(i in 1:length(TrainData)) { #for every column in the training dataset
        if( sum( is.na( TrainData[, i] ) ) /nrow(TrainData) >= .6 ) { #if n?? NAs > 60% of total observations
        for(j in 1:length(temp)) {
            if( length( grep(names(TrainData[i]), names(temp)[j]) ) ==1)  { #if the columns are the same:
                temp <- temp[ , -j] #Remove that column
            }   
        } 
    }
}
dim(temp)
TrainData <- temp
rm(temp)
```
### Testing-set cleaning
#### The above three steps are repeated on TestData   
```{r}
Test1 <- colnames(TrainData)
Clean <- colnames(TrainData[ , -58]) 
Testdata <- TestData[Test1]
testing <- testing[Clean]
dim(TestData)
dim(testing)   
```   

### In order to ensure proper functioning of Decision Trees and especially RandomForest Algorithm with the Test data set (data set provided), we need to coerce the data into the same type.   
```{r}
for (i in 1:length(testing) ) {
        for(j in 1:length(TrainData)) {
        if( length( grep(names(TrainData[i]), names(testing)[j]) ) ==1)  {
            class(testing[j]) <- class(TrainData[i])
        }      
    }      
}
```   
#Verifying Coercion really worked, 
```{r}
testing <- rbind(TrainData[2, -58] , testing) #note row 2 does not mean anything, this will be removed right.. now:
testing <- testing[-1,]
```   
### Using ML algorithms for prediction: Decision Tree   
```{r}   
modFitA1 <- rpart(classe ~ ., data=TrainData, method="class")
fancyRpartPlot(modFitA1)
```   
##### Predicting in-sample error   
```{r}
predictionsA1 <- predict(modFitA1, TestData, type = "class")
```   
##### Using confusion Matrix to test results:
```{r}
confusionMatrix(predictionsA1, TestData$classe)
```   
### Using ML algorithms for prediction: Random Forests
```{r}
modFitB1 <- randomForest(classe ~. , data=TrainData)
```   
#### Predicting in-sample error 
```{r}
predictionsB1 <- predict(modFitB1, TestData, type = "class")
```   
#### Using confusion Matrix to test results 
```{r}
confusionMatrix(predictionsB1, TestData$classe)
```   

##### Random Forests yielded better Results. Hence we use it to generate files to submit using the provided Test Set out-of-sample error.   
```{r}
predictionsB2 <- predict(modFitB1, testing, type = "class")
```   
### Generating Files to submit as answers
```{r}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictionsB2)
```   