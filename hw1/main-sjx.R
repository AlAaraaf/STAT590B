#### dependencies and env. settings ####
setwd('D:sjx/ISU/Modules/STAT590B/hw1')
library(reticulate)
library(dplyr)
library(tidyr)
library(ggplot2)
library(GGally)
library(keras)
library(tensorflow)
library(e1071)
use_condaenv('jiaxin')
tf <- import('tensorflow')
seed = 0590
set.seed(seed)
py_set_seed(seed, disable_hash_randomization = TRUE)
set_random_seed( 
  seed, 
  disable_gpu = TRUE 
) 

#### Q1 - drop NA from me1 and me2 ####

##### preprocessing #####

## read in data

data = read.csv('sdss-all-c.csv', header = F)
colname = c('id','class','brightness','size','texture','me1','me2')
colnames(data) <- colname
data$class[data$class == 3] <- 0 # galaxy
data$class[data$class == 6] <- 1 # star
data$me1 = as.numeric(data$me1)
data$me2 = as.numeric(data$me2)

## drop NA observations
data_complete7 = data %>% drop_na()
data_complete5 = data[,1:5]

## standardized
data_complete7[,3:7] = scale(data_complete7[,3:7])
data_complete5[,3:5] = scale(data_complete5[,3:5])

## pack datasets
packing = function(data){
  
  ## randomly partition the dataset (.75 training)
  num = dim(data)[1]
  training_idx = sample(1:num, num*0.75, replace = F)
  data.train <-  data[training_idx,]
  data.test <-  data[-training_idx,]
  
  ## gathering data
  train.inputs <- data.train[,3:dim(data.train)[2]]
  train.targets = as.matrix(data.train$class)
  test.inputs <- data.test[, 3:dim(data.test)[2]]
  test.targets = as.matrix(data.test$class)
  
  dataset = list()
  dataset$train = list(input = train.inputs, targets = train.targets)
  dataset$test = list(input = test.inputs, targets = test.targets)
  
  return(dataset)
}

## check classification
table(data_complete7$class)

## get dataset
dataset = packing(data_complete7)

##### linear classification by tensorflow #####

sq_loss = function(targets, pred){
  tf$reduce_mean(tf$square(tf$subtract(targets, pred)))
}

# model
model = function(inputs){tf$matmul(inputs, w) + b}

## run step
run = function(t_input, targets, lr){
  with(tf$GradientTape() %as% tape, {
    pred <- model(t_input)
    loss <- sq_loss(targets, pred)
  })
  update <- tape$gradient(loss, list(w=w, b=b))
  
  w$assign_sub(update$w * lr)
  b$assign_sub(update$b * lr)
  loss
}

## training
model_train = function(t_input, targets, lr){
  step = 0
  thres = 1e-3
  curr = 0
  
  repeat{
    
    step <- step + 1
    t_loss <- run(t_input, targets, lr)
    loss <- as.numeric(sprintf("%.3f\n", t_loss))

    if (step %% 5 == 0) cat(sprintf("Loss at step %s: %.6f\n", step, t_loss))
    updates <- abs(loss - curr)
    
    if (step > 0 & updates <= thres) {
      cat(sprintf('Finish after %s steps.\n', step))
      break
    }
    curr <- loss
  }
}

## evaluation
eval = function(targets, pred){
  acc = sum(targets == round(pred))
  mse = sum(targets - pred)^2 / dim(targets)[1]
  cat(sprintf('Accuracy: %.6f(%s/%s)  MSE: %.6f\n', acc/dim(targets)[1],acc, dim(targets)[1], mse))
}

## iteration for different learning rates
network_training = function(dataset, lr_list){
  
  input_dim = dim(dataset$train$input)[2]
  output_dim = dim(dataset$train$targets)[2]
  
  for (lr in lr_list){
    cat('Learning rate: ', lr,'\n')
    
    # initialize hidden layer parameters
    w = tf$Variable(tf$random$uniform(shape(input_dim, output_dim)))
    b = tf$Variable(tf$zeros(shape(output_dim)))
    train.t_input = as_tensor(dataset$train$input, dtype='float32')
    
    # train
    model_train(train.t_input, dataset$train$targets, lr)
    
    # eval
    test.t_input = as_tensor(dataset$test$input, dtype='float32')
    prediction = as.vector(model(test.t_input))
    eval(dataset$test$targets, prediction)
    cat('-------------------------------------\n')
  }
}

## result
lr_list = c(0.01, 0.05, 0.1, 0.15, 0.2)
network_training(dataset, lr_list) #still have a bug

# According to the evaluation result, the best linear classification model achieves an 98.1% accuracy on the test dataset with learning rate = 0.15 and updates for 20 steps. Another comparable model also achieves 98.1% accuracy on the test dataset with learning rate = 0.20 and updates for only 12 steps, while the MSE on test dataset is a little bit larger than the best model.

##### classification by svm #####
formula = class~brightness+size+texture+me1+me2
tuning = function(kernel, cv = F){
  result = tune(method = svm,
       train.x = formula,
       data = data.train,
       validation.x = ifelse(cv, NULL, data.test),
       kernel = kernel,
       ranges = list(cost = c(0.001, 0.01, 0.1, 1,5,10,100)))
  summary(result)
}

## linear DB
tuning('linear')
svm.linear = svm(formula, data.train, cost = 0.1, kernel = 'linear')

## nonlinear DB with radial kernel
tuning('radial')
svm.radial = svm(formula, data.train, cost = 1, kernel = 'radial')

## nonlinear DB with polynomial kernel
tuning('polynomial')
svm.polynomial = svm(formula, data.train, cost = 5, kernel = 'polynomial')

## nonlinear DB with sigmoidal kernel
tuning('sigmoid')
sum.sigmoid = svm(formula, data.train, cost = 0.1, kernel = 'sigmoid')


#### Q1 - drop me1 and me2 ####
dataset = packing(data_complete5)

##### linear classification by tensorflow #####
network_training(dataset, lr_list) # BUG!!!!

##### classification by svm #####
tbd

#### q2 ####

##### preprocessing #####

## read in data ##
inputs = read.table('ziptrain.dat')
targets = read.table('zipdigit.dat')
