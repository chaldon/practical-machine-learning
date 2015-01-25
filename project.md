
## Practical Machine Learning Project

A. Shchelokov, 2015-01-23 17:42:54

The task is to build a predictive model using quantitive data from various sensors attached to a body of athlete during an exercise. The model should infer "classe" of exercise. "classe"s attached to source data fall to one of categories "A","B","C","D" or "E".
The following steps was done to build model:

Perform libraries load, pseudo-random generator initialization.


```r
library(caret); library(doParallel);
set.seed(1421979345)
```
Read data while performing some data cleanup. More specifically - replace "NA", "#DIV/0!" and empty string to R's NA values.

```r
train <- read.csv("pml-training.csv", na.strings=c("NA", "#DIV/0!", ""));
test <- read.csv("pml-testing.csv"); 
```
Split to a train(75%) and validation(25%) sets. It's different from "rule of thumb - 60%/40%" recommendation because I want more data in training set.

```r
inTrain <- createDataPartition( y=train$classe, p=0.75, list = FALSE);
```
I observed a list of variables presentred in pml-testing.csv and do a manual selection of columns which I want to use for prediction (non-empty varaibles, some houskeeping variables like user name, time stamp was removed).

```r
para <- c("classe", "roll_belt","pitch_belt","yaw_belt","total_accel_belt","gyros_belt_x","gyros_belt_y","gyros_belt_z","accel_belt_x","accel_belt_y","accel_belt_z","magnet_belt_x","magnet_belt_y","magnet_belt_z","roll_arm","pitch_arm","yaw_arm","total_accel_arm","gyros_arm_x","gyros_arm_y","gyros_arm_z","accel_arm_x","accel_arm_y","accel_arm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z","roll_dumbbell","pitch_dumbbell","yaw_dumbbell","total_accel_dumbbell","gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z","accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z","roll_forearm","pitch_forearm","yaw_forearm","total_accel_forearm","gyros_forearm_x","gyros_forearm_y","gyros_forearm_z","accel_forearm_x","accel_forearm_y","accel_forearm_z","magnet_forearm_x","magnet_forearm_y","magnet_forearm_z");
```
Caret package support multiple algorithms for classification task, Random Forest is good option to test what we can get from the data before trying sometning else.
So, train model using random forest (in parallel, using advantage of modern multicore processors).

```r
Sys.time()
```

```
[1] "2015-01-23 17:43:04 EST"
```

```r
cl <- makeCluster(detectCores());
registerDoParallel(cl);

model <- train(classe~., data=train[inTrain,para], method="rf", prox=FALSE);

stopCluster(cl)
Sys.time()
```

```
[1] "2015-01-23 18:38:40 EST"
```
Few charactreristics of trained model.

```r
model
```

```
Random Forest 

14718 samples
   52 predictors
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 

Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 

Resampling results across tuning parameters:

  mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
   2    0.9889443  0.9860158  0.001447937  0.001833900
  27    0.9883387  0.9852514  0.001644710  0.002080496
  52    0.9798205  0.9744764  0.004796258  0.006068763

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 2. 
```
Save it in case I want to play with it later.

```r
save(model,file = "model.RData")
```
Test solution against validation set.

```r
conf <- confusionMatrix(train[-inTrain,"classe"],predict(model,train[-inTrain,para]));
conf;
```

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1395    0    0    0    0
         B    4  944    1    0    0
         C    0   12  843    0    0
         D    0    0   13  791    0
         E    0    0    0    2  899

Overall Statistics
                                          
               Accuracy : 0.9935          
                 95% CI : (0.9908, 0.9955)
    No Information Rate : 0.2853          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9917          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9971   0.9874   0.9837   0.9975   1.0000
Specificity            1.0000   0.9987   0.9970   0.9968   0.9995
Pos Pred Value         1.0000   0.9947   0.9860   0.9838   0.9978
Neg Pred Value         0.9989   0.9970   0.9965   0.9995   1.0000
Prevalence             0.2853   0.1949   0.1748   0.1617   0.1833
Detection Rate         0.2845   0.1925   0.1719   0.1613   0.1833
Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
Balanced Accuracy      0.9986   0.9931   0.9903   0.9972   0.9998
```
Obserwing Accuracy = 0.9934747 we can conclude that model is quite good and expected out of sample errors will be quite low.

Generate data for submission.

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(predict(model,test[,para[-1]]))
```
