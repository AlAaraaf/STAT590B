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

filter_list = c(32, 64, 128, 256, 512)
lrlist = c(0.001, 0.01, 0.05)
seedlist = c(1001, 1002, 1003, 1004, 1005)

metric_record = c()

for (lr in lrlist){
  for (current_size in filter_list){
    val_acc = c()
    for (seed in seedlist){
      set.seed(seed)
      model_name = paste(checkpoint_folder, 'q2a_',lr,'_',current_size,'filter_', seed, sep = '')
      model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                  save_weights_only = T,
                                                  save_best_only = T,
                                                  monitor = 'val_accuracy',
                                                  mode = 'max')
      model <- build_cnn_model1(c(current_size))
      model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                               loss = "binary_crossentropy",
                               metrics =  list("accuracy"))
      
      current_history <- model |>
        fit(x = dataset2$train$data, 
            y = dataset2$train$class,
            epochs = 20, batch_size = 512, validation_split = 0.5,
            callbacks = list(model_cp))
      val_acc = c(val_acc, max(current_history$metrics$val_accuracy))
    }
    current_record = c(lr, i, val_acc)
    metric_record = rbind(metric_record, current_record)
  }
}

metric_record = data.frame(metric_record)
colnames(metric_record) <- c('lr', 'filter', 'acc')

write.csv(metric_record, 'metric-cnn1.csv')