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
model <- function(inputs) {tf$matmul(inputs, W) + b}

square_loss <- function(targets, predictions)
  {
  per_sample_losses <- (targets - predictions)^2
  return (mean(per_sample_losses)) # average per-sample loss score
}

training_step <- function(inputs, targets) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model(inputs)
    #Forward pass, inside a gradient tape scope
    loss <- square_loss(predictions, targets)
  })
  grad_loss_wrt <- tape$gradient(loss, list(W = W, b = b))
  # Retrieve the gradient of the loss with regard to weights.
  W$assign_sub(grad_loss_wrt$W * learning_rate)
  b$assign_sub(grad_loss_wrt$b * learning_rate)
  # the two steps above update the weights. loss
}

#### training ####
inputs <- as_tensor(inputs, dtype = "float32")
learning_rate = 0.1
for (step in seq(40)) 
{
  loss <- training_step(inputs, targets)
  cat(sprintf("Loss at step %s: %.4f", step, loss))
  cat("\n")
}
  
#### prediction ####
  