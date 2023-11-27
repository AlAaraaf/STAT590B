img.preprocessing <- function(imglink, size, readfunc){
  img = readfunc(imglink)
  imglist = get.word(img)
  size = as.integer(size)
  smallimg_list = lapply(imglist, function(x) {tf$keras$preprocessing$image$smart_resize(x, size)})
  return(smallimg_list)
}

get.boundary <- function(candidates, thres = 30){
  i=0
  gaps = c()
  while(i+1 < length(candidates)){
    i=i+1
    if ((candidates[i+1] - candidates[i]) <=thres){
      next
    }else{
      gaps = c(gaps, candidates[i])
    }
  }
  return(gaps)
}

find.splitor <- function(img, dir, thres=0.9, gap_thres=30){
  len = ifelse(dir == 1, dim(img)[2], dim(img)[1])
  whiteratio = apply(img, dir, function(x) {sum(x == 1) / len})
  candidate = which(whiteratio > thres)
  splitor = get.boundary(candidate, gap_thres)
  
  return(splitor)
}

split.rows <- function(img, rowthres, rowgap){
  rowpieces = list()
  row_splitor = find.splitor(img[,,1], 1, thres = rowthres, gap_thres = rowgap)
  row_splitor = c(row_splitor, dim(img)[1])
  for(i in 1:(length(row_splitor)-1)){
    rowpieces[[i]] <- img[row_splitor[i]:row_splitor[i+1],1:dim(img)[2], 1:3]
  }
  
  return(rowpieces)
}

split.cols <- function(rowpieces, colthres, colgap){
  img_pieces = list()
  i=1
  for(current_piece in rowpieces){
    col_splitor = find.splitor(current_piece[,,1], 2, thres = colthres, gap_thres = colgap)
    col_splitor = c(col_splitor, dim(current_piece)[2])
    for(j in 1:(length(col_splitor)-1)){
      candidate <- current_piece[1:dim(current_piece)[1], col_splitor[j]:col_splitor[j+1], 1:3]
      whiteratio = sum(candidate[,,1] == 1) / (dim(candidate)[1] * dim(candidate)[2])
      if(whiteratio <= 0.85 & dim(candidate)[2] >= 100){
        img_pieces[[i]] <- candidate
        i = i+1
      }
    }
  }
  return(img_pieces)
}

get.word <- function(img, rowthres=0.9, colthres=0.95, rowgap=30, colgap=30){
  # split row first
  rowpieces <- split.rows(img, rowthres, rowgap)
  
  # split col next
  img_pieces <- split.cols(rowpieces, colthres, colgap)
  
  # choose first 
  
  return(img_pieces)
}

splice <- function(imglink, imgsize, readfunc, size, thres=0.9) {
  img <- readfunc(imglink)
  imgsize = as.integer(imgsize)
  img <- tf$keras$preprocessing$image$smart_resize(img, imgsize)
  repeat {
    pos_y <- sample(1:(dim(img)[1]-(size-1)), 1)
    pos_x <- sample(1:(dim(img)[2]-(size-1)), 1)
    candidate <- img[pos_y:(pos_y+(size-1)), pos_x:(pos_x+(size-1)), 1:3]
    
    # keep selecting images until you find one that isn't mostly whitespace
    white <- length(which(round(unlist(candidate)) == 1))/(dim(candidate)[1]*dim(candidate)[2])
    
    # add fail safe in case valid images can't be found
    if (exists('attempts') == FALSE) {
      attempts <- 1
    } else {
      attempts <- attempts + 1
    }
    if (white < thres | attempts > 20) {
      break
    }
  }
  return(candidate)
}

make.dataset <- function(folder, classlist, size, sample_size, ag_size, readfunc){
  trainset = list()
  testset = list()
  trainset$data = array(0, dim = c(sample_size$train*ag_size$train, 128,128, 3))
  testset$data = array(0, dim = c(sample_size$test*ag_size$test, 128,128, 3))
  trainset$class = c()
  testset$class = c()
  
  filelink = c()
  dataclass = c()
  train_id = sample(1:(sample_size$train + sample_size$test), sample_size$train, replace = F)
  
  # get data links
  for (i in 1:length(classlist)){
    
    current_class = classlist[i]
    current_folder = paste(folder, current_class, '/', sep = '')
    filelist = list.files(current_folder, full.names = T)
    cur_data = sample(filelist, as.integer(0.5*(sample_size$train + sample_size$test)))
    filelink = c(filelink, cur_data)
    dataclass = c(dataclass, rep((i-1), length(cur_data)))
  }
  
  datasource = data.frame(filedir = filelink, class = dataclass)
  datasource = datasource[sample(dim(datasource)[1], dim(datasource)[1]),]
  train_file = datasource[train_id,]
  test_file = datasource[-train_id,]
  
  # get cropped data
  classlist = c()
  ind = 1
  cat("get train data: ")
  for (i in 1:dim(train_file)[1]){
    current_file = train_file[i,]
    current_imglist = list()
    for (j in 1:ag_size$train){
      current_imglist[[j]] <- splice(current_file$filedir, size, readfunc, 128)
    }
    for (j in 1:ag_size$train){
      trainset$data[ind,,,] <- current_imglist[[j]]
      cat(ind ,' ')
      ind = ind + 1
    }
    classlist = c(classlist, rep(current_file$class, 10))
  }
  cat('\n')
  trainset$class = classlist
  
  classlist = c()
  ind = 1
  cat('get test data: ')
  for (i in 1:dim(test_file)[1]){
    current_file = test_file[i,]
    current_imglist = list()
    for(j in 1:ag_size$test){
      current_imglist[[j]] <- splice(current_file$filedir, size, readfunc, 128)
    }
    for (j in 1:ag_size$test){
      testset$data[ind,,,] <- current_imglist[[j]]
      cat(ind, ' ')
      ind = ind + 1
    }
    classlist = c(classlist, rep(current_file$class, 10))
  }
  cat('\n')
  testset$class = classlist
  
  
  dataset = list()
  dataset$train <-trainset
  dataset$test <- testset
  
  return(dataset)
}
