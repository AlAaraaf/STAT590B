##### dependencies #####
#library(reticulate)
#use_condaenv('connectr')
#library(tensorflow)
#library(keras)
#f <- import('tensorflow')
#pil <- import('PIL')
#np <- import('numpy')
#pd <- import('pandas')

##### wrapped functions #####
get.dataset1 <- function(){
  data = pd$read_csv('All_Script_Str(1-42)Frac(51-52)Mor(53-124)-1-116.csv')
  train_id = sample(1:(dim(data)[1]), dim(data)[1])
  data = data[train_id,]
  
  feature = data[,1:116]
  class = data[,117]
  class = unlist(lapply(class, function(x) {x-1}))
  
  dataset = list()
  dataset$train = list()
  dataset$train$data = np$array(data[,1:116])
  dataset$train$class = to_categorical(class)
  
  return(dataset)
}

build_ff_model1 <- function(nlayers, unitlist){
  model <- keras_model_sequential(input_shape = c(116))
  
  for (i in 1:nlayers){
    model %>% 
      layer_dense(units = unitlist[i], activation = 'relu')
  }
  
  model %>% layer_dense(11, activation = 'softmax')
  
  return(model)
}

build_ff_model2 <- function(nlayers, unitlist){
  model <- keras_model_sequential(input_shape = c(116))
  
  for (i in 1:nlayers){
    model %>% 
      layer_dense(units = unitlist[i], 
                  activation = 'relu', 
                  kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01))
  }
  
  model %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(11, activation = 'softmax')
  
  return(model)
}


read.in.all.file <- function(class){
  
  filelist = data.frame()
  for (i in 0:1){
    current_dir = paste('./data2/', class[i+1], '/',sep = '')
    allfiles = list.files(current_dir, full.names = T)
    current_filelist = data.frame(filedir = allfiles,
                                  class = rep(i, length(allfiles)))
    filelist = rbind(filelist, current_filelist)
  }
  return(filelist)
}

get.dataset2 <- function(filelist, p){
  # INPUT
  # filelist - the dataframe [filedir, class]
  # p - the percentage of training data to total data
  
  # OUTPUT
  # the dataset list [train[data, tag], test[data, tag]]
  
  n = dim(filelist)[1]
  train_id = sample(n, n*p)
  train_file = filelist[train_id,]
  test_file = filelist[-train_id,]
  img_width = 256
  img_height = 256
  
  dataset = list()
  trainset = list()
  trainset$data = array(0, dim = c(length(train_id), img_width, img_height, 3))
  testset = list()
  testset$data = array(0, dim = c(dim(test_file)[1], img_width, img_height, 3))
  
  size = as.integer(c(img_width, img_height))
  for (i in 1:dim(train_file)[1]){
    current_file = train_file[i,]
    current_img = pil$Image$open(current_file$filedir)
    current_img = np$array(current_img$convert('RGB'))
    trainset$data[i,,,] <- tf$keras$preprocessing$image$smart_resize(current_img, size)
  }
  trainset$class = train_file$class
  
  for (i in 1:dim(test_file)[1]){
    current_file = test_file[i,]
    current_file = train_file[i,]
    current_img = pil$Image$open(current_file$filedir)
    current_img = np$array(current_img$convert('RGB'))
    testset$data[i,,,] <- tf$keras$preprocessing$image$smart_resize(current_img, size)
  }
  testset$class = test_file$class
  
  dataset$train <-  trainset
  dataset$test <-  testset
  
  return(dataset)
}

build_cnn_model1 <- function(filter_list, dropout = 0){

  img_shape = c(256, 256)
  input_shape = as_tensor(c(img_shape[1], img_shape[2], 3), dtype = 'int32')
  
  model <- keras_model_sequential()
  model %>% layer_rescaling(1/255)
  for (i in 1:length(filter_list)){
    model %>% 
      layer_conv_2d(filters = filter_list[i], kernel_size = 3, activation = 'relu') %>% 
      layer_max_pooling_2d(pool_size = 2)
  }
  model %>%
    layer_dropout(rate = dropout) %>% 
    layer_flatten() %>%
    layer_dense(1, activation = 'sigmoid')
  return(model)
}

build_cnn_model_ag <- function(filter_list, dropout = 0){
  
  img_shape = c(256, 256)
  input_shape = as_tensor(c(img_shape[1], img_shape[2], 3), dtype = 'int32')
  
  model <- keras_model_sequential()
  model %>% 
    layer_random_flip('horizontal') %>% 
    layer_random_rotation(0.1) %>% 
    layer_random_zoom(0.2) %>% 
    layer_rescaling(1/255)
  for (i in 1:length(filter_list)){
    model %>% 
      layer_conv_2d(filters = filter_list[i], kernel_size = 3, activation = 'relu') %>% 
      layer_max_pooling_2d(pool_size = 2)
  }
  model %>%
    layer_dropout(rate = dropout) %>% 
    layer_flatten() %>%
    layer_dense(1, activation = 'sigmoid')
  return(model)
}

build_cnn_model_bn <- function(filter_list, pool_list, dropout = 0){
  img_shape = c(256, 256)
  input_shape = as_tensor(c(img_shape[1], img_shape[2], 3), dtype = 'int32')
  
  model <- keras_model_sequential()
  for (i in 1:length(filter_list)){
    model %>% 
      layer_conv_2d(filters = filter_list[i], kernel_size = 3) %>% 
      layer_batch_normalization() %>% 
      layer_activation('relu') %>% 
      layer_max_pooling_2d(pool_size = c(pool_list[i], pool_list[i]))
  }
  model %>%
    layer_dropout(rate = dropout) %>% 
    layer_flatten() %>%
    layer_dense(1, activation = 'sigmoid')
  return(model)
  
}

build_cnn_model_resid <- function(filter_list, pool_list, dropout = 0){
  model <- keras_model_sequential()
  for (i in 1:length(filter_list)){
    model %>% 
      residual_block(filters = filter_list[i], pooling = pool_list)
  }
  model %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(1, activation = 'sigmoid')
  return(model)
}

residual_block <- function(x, filters, pooling = FALSE) {
  residual <- x
  x <- x |>
    layer_conv_2d(filters, 3, activation = "relu", padding = "same") |>
    layer_conv_2d(filters, 3, activation = "relu", padding = "same")
  if (pooling) {
    x <- x |> layer_max_pooling_2d(pool_size = 2, padding = "same")
    residual <- residual |> layer_conv_2d(filters, 1, strides = 2)
  } else if (filters != dim(residual)[4]) {
    ## Without max pooling only project residual if number of channels has changed.
    residual <- residual |> layer_conv_2d(filters, 1)
  }
  layer_add(list(x, residual))
}


##### hyper params for Q1 #####
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


##### model training #####
seedlist = c(1001, 1002, 1003, 1004, 1005)
for (seed in seedlist){
  set.seed(seed)
  
}