setwd('D:/sjx/ISU/Modules/STAT590B/hw3')

##### dependecies #####
library(reticulate)
use_condaenv('connectr')
library(tensorflow)
tf <- import('tensorflow')
library(keras)
tf$config$run_functions_eagerly(T)
library(tiff)

##### source functions #####
source('./prep_functions.R')
source('./model_structure.R')

##### Q1 #####
folder = './data1/'
classlist = c('bangla', 'devnagari')
size = c(1024, 1024)
dataset = make.dataset(folder, classlist, size)

checkpoint_folder = './checkpoint/'
historys = list()
lrlist = c(0.001, 0.01, 0.05)
dropout_rate = 0.5
lr = 0.01

model_name = paste(checkpoint_folder, 'data1_', lr,sep = '')
model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                            save_weights_only = T,
                                            save_best_only = T,
                                            monitor = 'val_accuracy',
                                            mode = 'max')

model <- construct_cnn_model(dropout = dropout_rate)
model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                         loss = "binary_crossentropy",
                         metrics =  list("accuracy"))

i=1
historys[[i]] <- model |>
  fit(x = dataset$train$data, 
      y = dataset$train$class,
      epochs = 10, batch_size = 64, validation_split = 0.2,
      callbacks = list(model_cp))
