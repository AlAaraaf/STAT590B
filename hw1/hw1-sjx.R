#### dependencies and env. settings####
setwd('F:/repos/STAT590B/hw1')
library(reticulate)
use_condaenv('jiaxin')
tf <- import('tensorflow')

#### Q1 ####

##### readin dataset #####
data = read.csv('sdss-all-c.csv', header = F)
colname = c('id','class','brightness','size','texture','me1','me2')
colnames(data) <- colname
num = dim(data)[1]

## randomly partition the dataset (.75 training)
set.seed(0590)
training_idx = sample(1:num, num*0.75, replace = F)
data.train = data[training_idx,]
data.test = data[-training_idx,]

## linear classification by tensorflow
