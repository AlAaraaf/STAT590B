library(reticulate)
use_condaenv('connectr')
library(tensorflow)
library(keras)
tf <- import('tensorflow')
np <- import('numpy')
pil <- import('PIL')
source('midterm-code.R')
setwd('/work/classtmp/carrice/STAT590B/midterm/')


classlist = c('CT_NonCOVID','CT_COVID')
filelist = read.in.all.file(classlist)
dataset2 = get.dataset2(filelist, 0.9)

filter_list = c(32,64,128)
lrlist = c(0.001, 0.01, 0.05)

metric_record = c()

for (lr in lrlist[1]){
    model_name = paste(checkpoint_folder, 'q2a4_',lr,'_', sep = '')
    model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                save_weights_only = T,
                                                save_best_only = T,
                                                monitor = 'val_accuracy',
                                                mode = 'max')
    model <- build_cnn_model_resid(filter_list)
    model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                             loss = "binary_crossentropy",
                             metrics =  list("accuracy"))
    
    current_history <- model |>
      fit(x = dataset2$train$data, 
          y = dataset2$train$class,
          epochs = 10, batch_size = 128, validation_split = 0.5,
          callbacks = list(model_cp))
    val_acc = max(current_history$metrics$val_accuracy)
    current_record = c(lr, 'resid', val_acc)
    metric_record = rbind(metric_record, current_record)
}

for (lr in lrlist[1]){
  model_name = paste(checkpoint_folder, 'q2a4A_',lr,'_', sep = '')
  model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                              save_weights_only = T,
                                              save_best_only = T,
                                              monitor = 'val_accuracy',
                                              mode = 'max')
  model <- build_cnn_model_bn(filter_list)
  model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                           loss = "binary_crossentropy",
                           metrics =  list("accuracy"))
  
  current_history <- model |>
    fit(x = dataset2$train$data, 
        y = dataset2$train$class,
        epochs = 10, batch_size = 128, validation_split = 0.5,
        callbacks = list(model_cp))
  val_acc = max(current_history$metrics$val_accuracy)
  current_record = c(lr, 'bn', val_acc)
  metric_record = rbind(metric_record, current_record)
}

for (lr in lrlist[1]){
  model_name = paste(checkpoint_folder, 'q2a4A_',lr,'_', sep = '')
  model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                              save_weights_only = T,
                                              save_best_only = T,
                                              monitor = 'val_accuracy',
                                              mode = 'max')
  model <- build_cnn_model_sep(filter_list)
  model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                           loss = "binary_crossentropy",
                           metrics =  list("accuracy"))
  
  current_history <- model |>
    fit(x = dataset2$train$data, 
        y = dataset2$train$class,
        epochs = 10, batch_size = 128, validation_split = 0.5,
        callbacks = list(model_cp))
  val_acc = max(current_history$metrics$val_accuracy)
  current_record = c(lr, 'sep', val_acc)
  metric_record = rbind(metric_record, current_record)
}

metric_record = data.frame(metric_record)
colnames(metric_record) <- c('lr', 'type', 'acc')

write.csv(metric_record, 'metric-cnn4.csv')