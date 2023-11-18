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
lr=0.001

model_name = paste(checkpoint_folder, 'q2b_',lr,'_','filterlist', i, sep = '')
model <-  build_cnn_model1(filter_list)

model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                         loss = "binary_crossentropy",
                         metrics =  list("accuracy"))

model %>% load_model_weights_tf(filepath = model_name)
model %>% evaluate(dataset2$test$data, dataset$test$class, batch_size = 128)