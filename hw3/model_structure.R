construct_cnn_model <- function(img_width = 224, dropout = 0.5){
  
  # OUTPUT
  # model - the constructed model
  input_shape = as_tensor(c(img_width, img_width, 3), dtype = 'int32')
  
  model <- keras_model_sequential()
  model %>% 
    layer_resizing(height = img_width, width = img_width, 
                   input_shape = input_shape) %>% 
    layer_conv_2d(filters = 32,kernel_size = 3,activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(rate = dropout) %>% 
    layer_flatten() %>%
    layer_dense(1, activation = 'sigmoid')
  return(model)
}
