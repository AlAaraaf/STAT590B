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

filter_list = list()
filter_list[[1]] <- c(32,64,128)
filter_list[[2]] <- c(32,64,128,256)
lrlist = c(0.001, 0.01, 0.05)

metric_record = c()

for (lr in lrlist){
  for (i in 1:2){
    model_name = paste(checkpoint_folder, 'q2a3_',lr,'_','filter',i, sep = '')
    model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                save_weights_only = T,
                                                save_best_only = T,
                                                monitor = 'val_accuracy',
                                                mode = 'max')
    model <- build_cnn_model_ag(filter_list[[i]])
    model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                             loss = "binary_crossentropy",
                             metrics =  list("accuracy"))
    
    current_history <- model |>
      fit(x = dataset2$train$data, 
          y = dataset2$train$class,
          epochs = 20, batch_size = 64, validation_split = 0.5,
          callbacks = list(model_cp))
    val_acc = max(current_history$metrics$val_accuracy)
    current_record = c(lr, i, val_acc)
    metric_record = rbind(metric_record, current_record)
  }
}

metric_record = data.frame(metric_record)
colnames(metric_record) <- c('lr', 'filter', 'acc')

write.csv(metric_record, 'metric-cnn3.csv')