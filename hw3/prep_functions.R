img.preprocessing <- function(imglink, size){
  img = readTIFF(imglink)
  size = as.integer(size)
  smallimg = tf$keras$preprocessing$image$smart_resize(img, size)
  return(smallimg)
}

make.dataset <- function(folder, classlist, size){
  trainset = list()
  testset = list()
  trainset$data = array(0, dim = c(20, size[1], size[2], 3))
  testset$data = array(0, dim = c(10, size[1], size[2], 3))
  trainset$class = c()
  testset$class = c()
  
  filelink = c()
  dataclass = c()
  train_id = sample(1:30, 20, replace = F)
  
  # get data links
  for (i in 1:length(classlist)){
    
    current_class = classlist[i]
    current_folder = paste(folder, current_class, '/', sep = '')
    filelist = list.files(current_folder, full.names = T)
    cur_data = sample(filelist, 15)
    filelink = c(filelink, cur_data)
    dataclass = c(dataclass, rep((i-1), 15))
  }
  
  datasource = data.frame(filedir = filelink, class = dataclass)
  train_file = datasource[train_id,]
  test_file = datasource[-train_id,]
  
  # get cropped data
  for (i in 1:dim(train_file)[1]){
    current_file = train_file[i,]
    current_img = img.preprocessing(current_file$filedir, size)
    trainset$data[i,,,] <- current_img
  }
  trainset$class = train_file$class
  
  for (i in 1:dim(test_file)[1]){
    current_file = test_file[i,]
    current_img = img.preprocessing(current_file$filedir, size)
    testset$data[i,,,] <- current_img
  }
  testset$class = test_file$class
  
  
  dataset = list()
  dataset$train <-trainset
  dataset$test <- testset
  
  return(dataset)
}