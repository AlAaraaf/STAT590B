# dependencies
library(reticulate)
use_condaenv('connectr')
library(tensorflow)
tf <- import('tensorflow')
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
ag_size = list()
ag_size$train = 10
ag_size$test = 1
dataset <- make.dataset(folder, classlist, size, sample_size, ag_size, readTIFF)

train_id = sample(dim(dataset$train$data)[1])
dataset$train$data <- dataset$train$data[train_id,,,]
dataset$train$class <- dataset$train$class[train_id]

##### model train #####
checkpoint_folder = './checkpoint/'
historys = list()
dropout_rate = 0
lr = 0.001
train_shape = c(224,224)

filter_list = c(8,16,32,64,128)
pool_list = c(T,T,T,T,F)

model_name = paste(checkpoint_folder, 'data1_', lr,sep = '')
model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                            save_weights_only = T,
                                            save_best_only = T,
                                            monitor = 'val_accuracy',
                                            mode = 'max')

model <- build_cnn_model(filter_list, train_shape, dropout = dropout_rate)
model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                         loss = "binary_crossentropy",
                         metrics =  list("accuracy"))

historys <- model |>
  fit(x = dataset$train$data, 
      y = dataset$train$class,
      epochs = 100, batch_size = 64, validation_split = 0.2,
      callbacks = list(model_cp))
