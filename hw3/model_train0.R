# dependencies
library(reticulate)
use_condaenv('connectr')
library(tensorflow)
library(keras)
tf$config$run_functions_eagerly(T)
library(tiff)
source('prep_functions.R')
source('model_structure.R')
setwd('/work/classtmp/carrice/STAT590B/hw3/')

##### get data #####
folder = './data1/'
classlist = c('bangla', 'devnagari')
size = c(1000, 1000)
sample_size = list()
sample_size$train = 300
sample_size$test = 10
dataset <- make.dataset0(folder, classlist, size, sample_size, readTIFF)

train_id = sample(dim(dataset$train$data)[1])
dataset$train$data <- dataset$train$data[train_id,,,]
dataset$train$class <- dataset$train$class[train_id]

##### model train #####
checkpoint_folder = './checkpoint/'
historys = list()
dropout_rate = 0
lr_list = c(0.05,0.01, 0.001)
epoch_list = c(20, 50, 100)
batch_list = c(128, 128, 256)

train_shape = c(224,224)

filter_list <- list(2 ^ (3:4), 2 ^ (4:5),
                    2 ^ (3:5), 2 ^ (4:6), 2 ^ (3:6), 2 ^ (4:7),
                    2 ^ (5:8), 2 ^ (3:7), 2 ^ (3:8))
pool_list = c(T,T,T,T,F)

metric_record = c()

for (i in 1:length(lr_list)){
  for (j in 1:length(filter_list)){
    lr = lr_list[i]
    filter = filter_list[[j]]
    
    model_name = paste(checkpoint_folder, 'data1_', lr, '_filterid_', j, sep = '')
    cat(model_name, '\n')
    model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                save_weights_only = T,
                                                save_best_only = T,
                                                monitor = 'val_accuracy',
                                                mode = 'max')
    
    model <- build_cnn_model(filter, train_shape, dropout = dropout_rate)
    model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                             loss = "binary_crossentropy",
                             metrics =  list("accuracy"))
    
    historys <- model |>
      fit(x = dataset$train$data, 
          y = dataset$train$class,
          epochs = epoch_list[i], batch_size = batch_list[i], validation_split = 0.5,
          callbacks = list(model_cp))
    
    
    val_acc = max(current_history$metrics$val_accuracy)
    train_acc = max(current_history$metrics$accuracy)
    current_record = c(lr, j, val_acc, train_acc)
    metric_record = rbind(metric_record, current_record)
    
    cat(current_record, '\n------------------------------------------\n')
  }
}
metric_record = data.frame(metric_record)
colnames(metric_record) <- c('lr', 'filter', 'val_acc','acc')
write.csv(metric_record, 'record0.csv')