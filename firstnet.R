library(tensorflow)
library(keras)
mnist <- dataset_mnist()


train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y


## train_images and train_labels form the training set, the data that the model will learn from. The model will then be tested on the test set, test_images and test_labels. The images are encoded as R arrays, and the labels are an array of digits, ranging from 0 to 9. The images and labels have a one-to-one correspondence. Let’s look at the training data, shown here:


str(train_images)
str(train_labels)

## And here’s the test data:

str(test_images)
str(test_labels)

## The workflow will be as follows: first, we’ll feed the neural network the training data, train_images and train_labels. Then the network will learn to associate images and labels. Finally, we’ll ask the network to produce predictions for test_images, and we’ll verify whether these predictions match the labels from test_labels.

model <- keras_model_sequential(list(
layer_dense(units = 512, activation = "relu"),
layer_dense(units = 10, activation = "softmax")
))

## The core building block of neural networks is the layer. We can think of a layer as a filter for data: some data goes in, and it comes out in a more useful form. Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand. Most of deep learning consists of chaining together simple layers that will implement a form of progressive data distillation.

## Here, our model consists of a sequence of two Dense layers, which are densely connected (also called fully connected) neural layers. The second (and last) layer is a 10-way softmax classification layer, which means it will return an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image belongs to one of our 10 digit classes. To make the model ready for training, we need to pick the following three things as part of the compilation step:

##    An optimizer—The mechanism through which the model will update itself based on the training data it sees, so as to improve its performance.
##    A loss function—How the model will be able to measure its performance on the training data, and thus how it will be able to steer itself in the right direction.
##    Metrics to monitor during training and testing—Here, we care only about accuracy (the fraction of the images that were correctly classified).

##  The compilation step

compile(model,
        optimizer = "rmsprop",
        loss = "sparse_categorical_crossentropy",
        metrics = "accuracy")

## Note that we don’t save the return value from compile() because the model is modified in place. Before training, we’ll preprocess the data by reshaping it into the shape the model expects and scaling it so that all values are in the [0, 1] interval, as shown next. Previously, our training images were stored in an array of shape (60000, 28, 28) of type integer with values in the [0, 255] interval. We’ll transform it into a double array of shape (60000, 28 * 28) with values between 0 and 1.

## Preparing the image data

train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

## Note that we use the array_reshape() function rather than the dim•() function to reshape the array (for tensor reshaping).

## Fitting the model:

fit(model, train_images, train_labels, epochs = 5, batch_size = 128)

## Two quantities are displayed during training: the loss of the model over the training data, and the accuracy of the model over the training data. We reasonably quickly reach an accuracy of 0.9888 (98.88%) on the training data. 
## Now that we have a trained model, we can use it to predict class probabilities for new digits—images that weren’t part of the training data, like those from the test set.

## Using the model to make predictions:

test_digits <- test_images[1:10, ]
predictions <- predict(model, test_digits)
str(predictions)

predictions[1, ]

## Each number of index i in that array (predictions[1, ]) corresponds to the probability that digit image test_digits[1, ] belongs to class i. This first test digit has the highest probability score (0.9999473, almost 1) at index 8, so according to our model, it must be a 7 (because we start counting at 0):

which.max(predictions[1, ])

predictions[1, 8]

## We can check that the test label agrees:
                                     
test_labels[1]

## Looks good!

## Evaluating the model on new data

metrics <- evaluate(model, test_images, test_labels)
metrics["accuracy"]

## The test set accuracy turns out to be 98.05%—that’s quite a bit lower than the training set accuracy (98.9%). This gap between training accuracy and test accuracy is an example of overfitting: the fact that machine learning models tend to perform worse on new data than on their training data.

##
## Reimplementing our first example in TensorFlow
##

## A SIMPLE DENSE CLASS
## We’ve seen earlier that the Dense layer implements the following input transformation, where W and b are model parameters, and activation() is an element-wise function (usually relu(), but it would be softmax() for the last layer):

## output <- activation(dot(W, input) + b)

## Let’s implement a simple Dense layer as a plain R environment with a class attribute NaiveDense, two TensorFlow variables, W and b, and a call() method that applies the preceding transformation:

random_array <- function(dim, min = 0, max = 1) (array(runif(prod(dim), min, max), dim))

layer_naive_dense <- function(input_size, output_size, activation) {
    self <- new.env(parent = emptyenv())
    attr(self, "class") <- "NaiveDense"
    self$activation <- activation
    w_shape <- c(input_size, output_size)
    w_initial_value <- random_array(w_shape, min = 0, max = 1e-1)
    self$W <- tf$Variable(w_initial_value)
    ## Create a matrix, W, of shape (input_size, output_size), initialized with random values.
    b_shape <- c(output_size)
    b_initial_value <- array(0, b_shape)
    self$b <- tf$Variable(b_initial_value) ## Create a vector, b, of shape (output_size), initialized with zeros.
    self$weights <- list(self$W, self$b)    ## Convenience property for retrieving all the layer’s weights
    self$call <- function(inputs) {
        self$activation(tf$matmul(inputs, self$W) + self$b)
    }
    self ## Apply the forward pass in a function named call.
self
}

## We stick to TensorFlow operations in this function, so that GradientTape can  track them. 
##
## A SIMPLE SEQUENTIAL CLASS
## Now, let’s create a naive_model_sequential() to chain these layers, as shown in the next code snippet. It wraps a list of layers and exposes a call() method that simply calls the underlying layers on the inputs, in order. It also features a weights property to easily keep track of the layers’ parameters:

naive_model_sequential <- function(layers) {
    self <- new.env(parent = emptyenv())
    attr(self, "class") <- "NaiveSequential"
    self$layers <- layers
    weights <- lapply(layers, function(layer) layer$weights)
    self$weights <- do.call(c, weights)
    ## Flatten the nested list one level.
    self$call <- function(inputs) {
        x <- inputs
        for (layer in self$layers)
            x <- layer$call(x)
        x
    }
    self
}

## Using this NaiveDense class and this NaiveSequential class, we can create a mock Keras model:

model <- naive_model_sequential(list(
    layer_naive_dense(input_size = 28 * 28, output_size = 512,
                      activation = tf$nn$relu),
    layer_naive_dense(input_size = 512, output_size = 10,
                      activation = tf$nn$softmax)
))
stopifnot(length(model$weights) == 4)

## A BATCH GENERATOR
## Next, we need a way to iterate over the MNIST data in mini-batches. This is easily done as follows:

new_batch_generator <- function(images, labels, batch_size = 128) {
    self <- new.env(parent = emptyenv())
    attr(self, "class") <- "BatchGenerator"
    stopifnot(nrow(images) == nrow(labels))
    self$index <- 1
    self$images <- images
    self$labels <- labels
    self$batch_size <- batch_size
    self$num_batches <- ceiling(nrow(images) / batch_size)
    self$get_next_batch <- function() {
        start <- self$index
        if(start > nrow(images))
            return(NULL) ##  Generator is finished.
        end <- start + self$batch_size - 1
        if(end > nrow(images))
            end <- nrow(images) ## Last batch may be smaller.
        self$index <- end + 1
        indices <- start:end
        list(images = self$images[indices, ],
             labels = self$labels[indices])
    }
    self
}

## Running one training step
## The most difficult part of the process is the “training step”: updating the weights of the model after running it on one batch of data. We need to do the following:

## 1. Compute the predictions of the model for the images in the batch.
## 2. Compute the loss value for these predictions, given the actual labels.
## 3. Compute the gradient of the loss with regard to the model’s weights.

##Move the weights by a small amount in the direction opposite to the gradient.
## To compute the gradient, we will use the TensorFlow GradientTape object we introduced earlier:

one_training_step <- function(model, images_batch, labels_batch) {
    with(tf$GradientTape() %as% tape, {
        predictions <- model$call(images_batch)
        per_sample_losses <-
            loss_sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss <- mean(per_sample_losses)
        ## Run the forward pass (compute the model’s predictions under a GradientTape scope).
    })
    gradients <- tape$gradient(average_loss, model$weights)
    update_weights(gradients, model$weights)
    ## Compute the gradient of the loss  with regard to the weights. The output gradients is a list where each entry corresponds to a weight rom the model$weights list. 
    average_loss
    ## Update the weights using the gradients (we will define this function shortly).
}

## The purpose of the “weight update” step (represented by the preceding update_weights() function) is to move the weights by “a bit” in a direction that will reduce the loss on this batch. The magnitude of the move is determined by the “learning rate,” typically a small quantity. The simplest way to implement this update_weights() function is to subtract gradient * learning_rate from each weight:

learning_rate <- 1e-3
update_weights <- function(gradients, weights) {
    stopifnot(length(gradients) == length(weights))
    for (i in seq_along(weights))
        weights[[i]]$assign_sub(
                         ## x$assign_sub(value) is the equivalent of x <- x - value for TensorFlow variables.
                         gradients[[i]] * learning_rate)
}
    
## In practice, we would almost never implement a weight update step like this by hand. Instead, we would use an Optimizer instance from Keras:

optimizer <- optimizer_sgd(learning_rate = 1e-3)
update_weights <- function(gradients, weights)
    optimizer$apply_gradients(zip_lists(gradients, weights))

## zip_lists() is a helper function that we use to turn the lists of gradients and weights into a list of (gradient, weight) pairs. We use it to pair gradients with weights for the optimizer. For example:

str(zip_lists(
    gradients = list("grad_for_wt_1", "grad_for_wt_2", "grad_for_wt_3"),
    weights = list("weight_1", "weight_2", "weight_3")))

## Now that our per-batch training step is ready, we can move on to implementing an entire epoch of training.

## The full training loop:
## An epoch of training simply consists of repeating the training step for each batch in the training data, and the full training loop is simply the repetition of one epoch:

fit <- function(model, images, labels, epochs, batch_size = 128) {
    for (epoch_counter in seq_len(epochs)) {
        cat("Epoch ", epoch_counter, "\n")
        batch_generator <- new_batch_generator(images, labels)
        for (batch_counter in seq_len(batch_generator$num_batches)) {
            batch <- batch_generator$get_next_batch()
            loss <- one_training_step(model, batch$images, batch$labels)
            if (batch_counter %% 100 == 0)
                cat(sprintf("loss at batch %s: %.2f\n", batch_counter, loss))
        }
    }
}

## Let us try it out:

fit(model, train_images, train_labels, epochs = 10, batch_size = 128)

## Evaluating the model
## We can evaluate the model by taking the max.col() of its predictions over the test images, and comparing it to the expected labels:


predictions <- model$call(test_images)
predictions <- as.array(predictions) ##  Convert the TensorFlow Tensor to an R array.
predicted_labels <- max.col(predictions) - 1 ## max.col(x) is a vectorized implementation of apply(x, 1, which.max)).
matches <- predicted_labels == test_labels
cat(sprintf("accuracy: %.2f\n", mean(matches)))

## All done! As we can see, it’s quite a bit of work to do “by hand” what we can do in a few lines of Keras code. But because we’ve gone through these steps, you should now have some understanding of what goes on inside a neural network when you call fit(), and help in being able to leverage the high-level features of the Keras API.

