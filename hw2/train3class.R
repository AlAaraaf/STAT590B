source('./prep_funcs.R')
library(reticulate)
use_condaenv('connectr')
library(tensorflow)
tf <- import('tensorflow')
library(keras)
tf$config$run_functions_eagerly(T)
library(jpeg)

# set dataset
classlist <- c('cat','dog', 'wild')
folders = c('./dataset/train/','./dataset/val/')

# model hyperparameters
layerlist = c(2,4,8,12,16)
epochs = c(10,10,10,20,20)
unitlists =list()
unitlists[[1]] <- c(128,64) # 2 layers
unitlists[[2]] <- c(128,64,64,32) # 4 layers
unitlists[[3]] <- c(rep(128,3),rep(64,5)) # 8 layers
unitlists[[4]] <- c(rep(128,3),rep(64,6), rep(32,3)) # 12 layers
unitlists[[5]] <- c(rep(128,3),rep(64,9), rep(32,4)) # 16 layers

checkpoint_folder = './checkpoint/'
historys = list()
lrlist = c(0.001, 0.01, 0.05)
dropout_rate = c(0, 0.5)

# read in data
filelist = read.in.all.file(folders, classlist)
dataset = make.dataset(dim(filelist)[1], filelist, 0.75)
dataset = convert_onehot(dataset)

# model fit
for(dp in dropout_rate){
  for (lr in lrlist){
    for (i in 1:5){
      model_name = paste(checkpoint_folder, '3class_', i,'layer', '_',lr,'_',dp,sep = '')
      model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                  save_weights_only = T,
                                                  save_best_only = T,
                                                  monitor = 'val_accuracy',
                                                  mode = 'max')
      
      model <- construct_fc_model_multiclass(layerlist[i], unitlists[[i]], dropout = dp)
      model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                               loss = "categorical_crossentropy",
                               metrics =  list("accuracy"))
      
      historys[[i]] <- model |>
        fit(x = dataset$train$data, 
            y = dataset$train$class,
            epochs = epochs[i], batch_size = 256, validation_split = 0.2,
            callbacks = list(model_cp))
    }
  }
}

# model evaluate
for (dp in dropout_rate){
  for (lr in lrlist){
    cat("current lr:", lr, 'dropout: ', dp, '\n')
    for (i in 1:5){
      model_name = paste(checkpoint_folder, '3class_', i,'layer','_',lr, '_',dp, sep = '')
      model <- construct_fc_model_multiclass(layerlist[i], unitlists[[i]], dropout = dp)
      
      model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                               loss = "categorical_crossentropy",
                               metrics =  list("accuracy"))
      
      model %>% load_model_weights_tf(filepath = model_name)
      model %>% evaluate(dataset$test$data, dataset$test$class, batch_size = 256)
    }
  }
}
