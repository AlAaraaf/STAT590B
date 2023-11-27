build_cnn_model <- function(filter_list, img_shape = c(224, 224), dropout = 0.5){
  model <- keras_model_sequential()
  model %>%
    layer_resizing(height = img_shape[1], width = img_shape[2])
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

build_cnn_model_resid <- function(filter_list, pool_list, img_shape = c(224, 224),dropout = 0){
  model <- keras_model_sequential()
  model %>%
    layer_resizing(height = img_shape[1], width = img_shape[2])
  for (i in 1:length(filter_list)){
    model %>% 
      residual_block(filters = filter_list[i], pooling = pool_list[i])
  }
  model %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(1, activation = 'sigmoid')
  return(model)
}