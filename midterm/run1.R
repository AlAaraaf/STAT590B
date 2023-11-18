library(reticulate)
use_condaenv('connectr')
library(tensorflow)
library(keras)
tf <- import('tensorflow')
pil <- import('PIL')
np <- import('numpy')
source('midterm-code.R')


dataset = get.dataset1()
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
seedlist = c(1001, 1002, 1003, 1004, 1005)

for (seed in seedlist){
  set.seed(seed)
  for (lr in lrlist){
    for (i in 1:5){
      model_name = paste(checkpoint_folder, 'q1_', i,'layer', '_',lr,'_',seed,sep = '')
      model_cp <- keras$callbacks$ModelCheckpoint(filepath = model_name,
                                                  save_weights_only = T,
                                                  save_best_only = T,
                                                  monitor = 'val_accuracy',
                                                  mode = 'max')
      
      model <- build_ff_model1(layerlist[i], unitlists[[i]])
      model %>% keras::compile(optimizer = optimizer_rmsprop(learning_rate = lr),
                               loss = "categorical_crossentropy",
                               metrics =  list("accuracy"))
      
      historys[[i]] <- model |>
        fit(x = dataset$train$data, 
            y = dataset$train$class,
            epochs = epochs[i], batch_size = 32, validation_split = 0.5,
            callbacks = list(model_cp))
    }
  }
}