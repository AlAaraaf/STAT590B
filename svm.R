library(reticulate)
library(tensorflow)
library(MASS)

use_condaenv('jiaxin')
tf <- import('tensorflow')
num_samp <- 1000
Sigma <- rbind(c(1, 0.5), c(0.5, 1))
neg_samp <- mvrnorm(n =num_samp, mu = c(0, 3), Sigma = Sigma)
pos_samp <- mvrnorm(n = num_samp, mu = c(3,0), Sigma = Sigma)
inputs <- rbind(neg_samp, pos_samp)
targets <- rbind(array(0, dim = c(num_samp, 1)), array(1, dim = c(num_samp, 1)))
#plot(x = inputs[, 1], y = inputs[, 2],col =ifelse(targets[, 1] == 0, "purple", "green"))


input_dim <- 2 # 2D example.
output_dim <- 1 # The output predictions will be a single
W <- tf$Variable(initial_value = tf$random$uniform(shape(input_dim, output_dim)))
b <- tf$Variable(initial_value = tf$zeros(shape(output_dim)))

#### svm ####
library(e1071)
training_data = data.frame(input = inputs, class = targets)
svmfit <- svm(class~., data = training_data, kernel = 'linear', cost = 10, scale = F)
plot(svmfit)

svmfit <- svm(class~., data = training_data, kernel = 'linear', cost = 0.1, scale = F)
plot(svmfit, training_data)

### choose cost for svm ####
