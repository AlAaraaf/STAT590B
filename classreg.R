##
## This material builds off of Chapter 4 of the R book. However, the book itself
## is not necessary.
##

library(keras)

## The IMDB dataset is a set of 50,000 highly polarized reviews from the Internet Movie Database. They are split into 25,000 reviews for training and 25,000 reviews for testing, each set consisting of 50% negative and 50% positive reviews. It comes packaged with Keras. It has already been preprocessed: the reviews (sequences of words) have been turned into sequences of integers, where each integer stands for a specific word in a dictionary.

## Loading the IMDB dataset

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

## Using the multiassignment (%<-%) operator
##
##The datasets built into Keras are all nested lists of training and test data. Here, we use the multiassignment operator (%<-%) from the zeallot package to unpack the list into a set of distinct variables. This could equally be written as follows:

imdb <- dataset_imdb(num_words = 10000)
train_data <- imdb$train$x
train_labels <- imdb$train$y
test_data <- imdb$test$x
test_labels <- imdb$test$y

## The multiassignment version  is more compact. The %<-% operator is automatically available whenever the R Keras package is attached.
## The argument num_words = 10000 means we keep only the top 10,000 most fre- quently occurring words in the training data. Rarer words are discarded.

str(train_data)
str(train_labels)

## Because we’re restricting ourselves to the top 10,000 most frequent words, no word index will exceed 10,000:

max(sapply(train_data, max))

##
## For some amusement, let us extract a review
## 

## word_index is a named vector mapping words to an integer index.
word_index <- dataset_imdb_word_index()

## Reverse it,  mapping integer indices to words
reverse_word_index <- names(word_index)
names(reverse_word_index) <- as.character(word_index)
decoded_words <- train_data[[1]] |>
sapply(function(i) {
    if (i > 3) reverse_word_index[[as.character(i - 3)]]
    ## Decodes the review. Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices for “padding,” “start of sequence,” and “unknown.”
    else "?"
})
decoded_review <- paste0(decoded_words, collapse = " ")
cat(decoded_review, "\n")


## Preparing the data
## 
## We can't directly feed lists of integers into a neural network. They all have different lengths, but a neural network expects to process contiguous batches of data. We turn your lists into tensors. We can do that in the following two ways:
##
## Pad our lists so that they all have the same length, turn them into an integer tensor of shape (samples, max_length), and start our model with a layer capable of handling such integer tensors (the Embedding layer, later).

## Multi-hot encode our lists to turn them into vectors of 0s and 1s. This would mean, for instance, turning the sequence [8, 5] into a 10,000-dimensional vector that would be all 0s except for indices 8 and 5, which would be 1s. Then we could use a layer_dense(), capable of handling floating-point vector data, as the first layer in our model.

## Let’s go with the latter solution to vectorize the data, which we do manually for maximum clarity.


vectorize_sequences <- function(sequences, dimension = 10000) {
    results <- array(0, dim = c(length(sequences), dimension))
    ## Create an all-zero matrix of shape (length(sequences), dimension).
    for (i in seq_along(sequences)) {
        results[i, sequences[[i]]] <- 1
        ## Set specific indices of results to 1s.
    }
    results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

str(x_train)

## We should also vectorize our labels, which is a straightforward cast of integers to floats:

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

## Now the data are ready to be fed into a neural network.

## 
## Building our model
##
## The input data is vectors, and the labels are scalars (1s and 0s): this is one of the simplest problem setups. A type of model that performs well on such a problem is a plain stack of densely connected layers (layer_dense()) with relu activations.

## We make two key architecture decisions about such a stack of dense layers:
##
## - How many layers to use
## - How many units to choose for each layer
##
## For now, we make the following architecture choices:
##
## - Two intermediate layers with 16 units each
## - A third layer that will output the scalar prediction
##

## Model definition
model <- keras_model_sequential() |>
layer_dense(16, activation = "relu") |>
layer_dense(16, activation = "relu") |>
layer_dense(1, activation = "sigmoid")

## The first argument being passed to each layer_dense() is the number of units in the layer: the dimensionality of representation space of the layer. Recall that each such layer_dense() with a relu activation implements the following chain of tensor operations: output <- relu(dot(input, W) + b)

##  Having 16 units means the weight matrix W will have shape (input_dimension, 16): the  dot product with W will project the input data onto a 16-dimensional representation space (and then we’ll add the bias vector b and apply the relu operation). We can intuitively understand the dimensionality of our representation space as "how much freedom we're allowing the model to have when learning internal representations."
## Having more units (a higher-dimensional representation space) allows your model to learn more complex representations, but it makes the model more computationally expensive and may lead to learning unwanted patterns (patterns that will improve performance on the training data but not on the test data, as we saw in the case of the MNIST dataset).

## Our intermediate layers use relu as their activation function, and the final layer uses a sigmoid activation so as to output a probability (a score between 0 and 1 indicating how likely the sample is to have the target “1”: how likely the review is to be positive). Recall from Chapter 1 that a relu (rectified linear unit) is a function meant to zero out negative values, whereas a sigmoid “squashes” arbitrary values into the [0, 1] interval , outputting something that can be interpreted as a probability.


## Our activation function here generalizes the linear function that would be too restrictive on its own. Without it, layer_dense would consist of two linear operations -- a dot product and an addition: output <- dot(input, W) + b
## The layer could learn only linear transformations (affine transformations) of the input data: the hypothesis space of the layer would be the set of all possible linear transformations of the input data into a 16-dimensional space. Such a hypothesis space is too restricted and wouldn’t benefit from multiple layers of representations, because a deep stack of linear layers would still implement a linear operation: adding more layers wouldn’t extend the hypothesis space.
## To get access to a much richer hypothesis space that will benefit from deep representations, we need nonlinearity, or activation function. relu is the most popular activation function in deep learning, but many other candidates exist.

## Finally, we need to choose a loss function and an optimizer. Because we're facing a binary classification problem and the output of our model is a probability (we end our model with a single-unit layer with a sigmoid activation), it’s best to use the binary_crossentropy loss. It isn’t the only viable choice: for instance, we could use mean_squared_error. But cross-entropy is usually the best choice when you’re dealing with models that output probabilities. Cross-entropy  measures the distance between probability distributions or, in this case, between the ground-truth distribution and our predictions. As for the choice of the optimizer, we’ll go with rmsprop, which is a usually a good default choice for virtually any problem.
##
## Here’s the step where we configure the model with the rmsprop optimizer and the binary_crossentropy loss function. Note that we’ll also monitor accuracy during training.
##
## Compiling the model

model |> compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy",
                 metrics = "accuracy")

## Validation sample: set aside a validation set

x_val <- x_train[seq(10000), ]
partial_x_train <- x_train[-seq(10000), ]
y_val <- y_train[seq(10000)]
partial_y_train <- y_train[-seq(10000)]

## Training our model

history <- model |> fit(
                        partial_x_train,
                        partial_y_train,
                        epochs = 20,
                        batch_size = 512,
                        validation_data = list(x_val, y_val)
                    )

## The history object has a member metrics, which is a named list containing data about everything that happened during training. Let’s look at it:

str(history$metrics)

## The metrics list contains four entries: one per metric that was being monitored during training and during validation. We’ll use the plot() method for the history object to plot the training and validation loss side by side, as well as the training and validation accuracy.

## We see that the training loss decreases with every epoch, and the training accuracy increases with every epoch. That’s what we would expect when running gradient descent optimization -- the quantity we’re trying to minimize should be less with every iteration. But that isn’t the case for the validation loss and accuracy: they seem to peak at the fourth epoch. This is an example of what we warned against earlier: a model that performs better on the training data isn’t necessarily a model that will do better on data it has never seen before. This is overfitting: after the fourth epoch, we are overoptimizing on the training data, and end up learning representations that are specific to the training data and not generaalizable to data outside of the training set.

## The plot() method for training history objects uses ggplot2 for plotting if it’s available (if it isn’t, base graphics are used). But to make our own custom plot (my preference), we extract a df and proceed.

history_df <- as.data.frame(history)
str(history_df)

## In this case, to prevent overfitting, we could stop training after six epochs.
## Let us train a new model from scratch for six epochs and then evaluate it on the test data.
##
## Retraining a model from scratch

model <- keras_model_sequential() |>
layer_dense(16, activation = "relu") |>
layer_dense(16, activation = "relu") |>
layer_dense(1, activation = "sigmoid")
model |> compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy",
                 metrics = "accuracy")

## The rmsprop optimizer is generally a good enough choice for most problems.

model |> fit(x_train, y_train, epochs = 6, batch_size = 512)
results <- model |> evaluate(x_test, y_test)

## The final results are:
results

## This fairly naive approach achieves an accuracy of 87%.
##
## Using a trained model to generate predictions on new data
##
## After having trained a model, we want to use it in a practical setting. You can generate the likelihood of reviews being positive by using the predict() method

model |> predict(x_test)

## The model is confident for some samples (0.99 or more, or 0.01 or less) but less confident for others.

##
##  Classifying newswires: A multiclass classification example
##

## The Reuters dataset
##
## The Reuters dataset, a set of short newswires and their topics, published by Reuters in 1986, is a simple, widely used toy dataset for text classification. The dataset contains 46 different topics; some topics are more represented than others, but each topic has at least 10 examples in the training set.

reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters

## Again, the argument num_words = 10000 restricts the data to the 10,000 most frequently occurring words found in the data. We have 8,982 trainin examples and 2,246 test examples:

length(train_data)

## Decoding newswires back to text

word_index <- dataset_reuters_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- as.character(word_index)
decoded_words <- train_data[[1]] |>
sapply(function(i) {
    if (i > 3) reverse_word_index[[as.character(i - 3)]]
    else "?"
})
decoded_review <- paste0(decoded_words, collapse = " ")
decoded_review

## The label associated with an example is an integer between 0 and 45—a topic index:

str(train_labels)

## Encoding the input data

vectorize_sequences <- function(sequences, dimension = 10000) {
    results <- matrix(0, nrow = length(sequences), ncol = dimension)
    for (i in seq_along(sequences))
        results[i, sequences[[i]]] <- 1
    results
}


x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

##
## One-hot encoding
##

to_one_hot <- function(labels, dimension = 46) {
    results <- matrix(0, nrow = length(labels), ncol = dimension)
    labels <- labels + 1 ## Vectorized training labels
    for(i in seq_along(labels)) {
        results[i, labels[[i]]] <- 1
    }
    results
}

y_train <- to_one_hot(train_labels)
y_test <- to_one_hot(test_labels)

##
## But the above is unnecessary in Keras:
##

y_train <- to_categorical(train_labels)
y_test <- to_categorical(test_labels)

## Building our model

## Model definition
model <- keras_model_sequential() |>
layer_dense(64, activation = "relu") |>
layer_dense(64, activation = "relu") |>
layer_dense(46, activation = "softmax")

##
## Note: the last layer has the same units as the number of classes, because we want the probability that a record belongs to each class.
## Also, we chose 64 units, but may want to have more if needed.
##
## The best loss function to use in this case is categorical_crossentropy. It mea- sures the distance between two probability distributions: here, between the probability distribution output by the model and the true distribution of the labels. By minimizing the distance between these two distributions, we train the model to output something as close as possible to the true labels.

model |> compile(optimizer = "rmsprop",
                 loss = "categorical_crossentropy",
                 metrics = "accuracy")

## Setting aside a validation set

val_indices <- 1:1000
x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]
y_val <- y_train[val_indices, ]
partial_y_train <- y_train[-val_indices, ]

## Training the model (for 20 epochs):

history <- model |> fit(
                        partial_x_train,
                        partial_y_train,
                        epochs = 20,
                        batch_size = 512,
                        validation_data = list(x_val, y_val)
                    )

data.frame(history)

## The model begins to overfit after nine epochs. Let’s train a new model from scratch for nine epochs and then evaluate it on the test set.

## Retraining a model from scratch

model <- keras_model_sequential() |>
layer_dense(64, activation = "relu") |>
layer_dense(64, activation = "relu") |>
layer_dense(46, activation = "softmax")

model |> compile(optimizer = "rmsprop",
                 loss = "categorical_crossentropy",
                 metrics = "accuracy")
model |> fit(x_train, y_train, epochs = 9, batch_size = 512)
results <- model |> evaluate(x_test, y_test)

results

## We get an accuracy of about 79%. This is a more challenging problem, not least because there are 46 classes.


predictions <- model |> predict(x_test)

## The largest entry is the predicted class

which.max(predictions[1, ])

## A different way to handle the labels and the loss
##
## Another way to encode the labels would be to preserve  their integer values, like this:

y_train <- train_labels
y_test <- test_labels

## The only thing this approach would change is the choice of the loss function. The loss function categorical_crossentropy, expects the labels to follow a categorical encoding. With integer labels, we should use sparse_categorical_crossentropy:

model |> compile(
             optimizer = "rmsprop",
             loss = "sparse_categorical_crossentropy",
             metrics = "accuracy")

## This new loss function is still mathematically the same as categorical_crossentropy; it just has a different interface.

## We now illustrate the importance of having sufficiently large intermediate layers
##
## I mentioned earlier that because the final outputs are 46-dimensional, we should avoid intermediate layers with many fewer than 46 units. Now let’s see what happens when we introduce an information bottleneck by having intermediate layers that are significantly less than 46-dimensional: for example, 4-dimensional.

## A model with an information bottleneck

model <- keras_model_sequential() |>
layer_dense(64, activation = "relu") |>
layer_dense(4, activation = "relu") |>
layer_dense(46, activation = "softmax")
model |> compile(optimizer = "rmsprop",
                 loss = "categorical_crossentropy",
                 metrics = "accuracy")
model |> fit(
             partial_x_train,
             partial_y_train,
             epochs = 20,
             batch_size = 128,
             validation_data = list(x_val, y_val)
         )

## The model now peaks at ~71.8% validation accuracy, an 8% absolute drop. This drop is mostly because we’re trying to compress a lot of information (enough information to recover the separation hyperplanes of 46 classes) into an intermediate space that is too low-dimensional. The model is able to cram most of the necessary information into these 4-dimensional representations, but not all of it.

## Try using larger or smaller layers: 32 units, 128 units, and so on.
## We used two intermediate layers before the final softmax classification layer. We could  try using a single intermediate layer, or three intermediate layers.


##
## A regression example: predicting house prices.
##
##  The Boston housing price dataset example will attempt to predict the median price of homes in a given Boston suburb in the mid-1970s, given data points about the suburb at the time, such as the crime rate, the local property tax rate, and so on. The dataset we’ll use has an interesting difference from the two previous examples. It has relatively few data points: only 506, split between 404 training samples and 102 test samples. And each feature in the input data (e.g., the crime rate) has a different scale. For instance, some values are proportions, which take values between 0 and 1, others take values between 1 and  12, others between 0 and 100, and so on.

## Loading the Boston housing dataset

boston <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% boston

## We have 404 training samples and 102 test samples, each with 13 numerical features, such as per capita crime rate, average number of rooms per dwelling, accessibility to highways, and so on. The targets are the median values of owner-occupied homes, in thousands of dollars:

str(train_targets)

## The prices are typically between $10,000 and $50,000, in the mld-1970s.

## Preparing the data
##
## It would be problematic to feed into a neural network values that all take wildly different ranges. The model might be able to automatically adapt to such heterogeneous data, but it would definitely make learning more difficult. A widespread best practice for dealing with such data is to do feature-wise normalization: for each feature in the input data (a column in the input data matrix), we subtract the mean of the feature and divide by the standard deviation, so that the feature is centered around 0 and has a unit standard deviation. This is easily done in R using the scale() function.

## Normalizing the data
mean <- apply(train_data, 2, mean)
sd <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = sd)
test_data <- scale(test_data, center = mean, scale = sd)

## Note that the quantities used for normalizing the test data are computed using the training data. 

## Building our model

## Because so few samples are available, we’ll use a very small model with two intermediate layers, each with 64 units. In general, the less training data you have, the worse  overfitting will be, and using a small model is one way to mitigate overfitting.

build_model <- function() { ## Because we need to instantiate the same model multiple times, we use a function to construct it.
    model <- keras_model_sequential() |>
    layer_dense(64, activation = "relu") |>
    layer_dense(64, activation = "relu") |>
    layer_dense(1)
    model |> compile(optimizer = "rmsprop",
                     loss = "mse",
                     metrics = "mae")
    model
}

## The model ends with a single unit and no activation (it will be a linear layer). This is a typical setup for scalar regression (a regression where you’re trying to predict a single continuous value). Applying an activation function would constrain the range the output can take: here, because the last layer is purely linear, the model is free to learn to predict values in any range. We compile the model with the mse loss function—mean squared error (MSE), the square of the difference between the predictions and the targets. This is a widely used loss function for regression problems.

##
## We are also monitoring a new metric during training: mean absolute error (MAE). It is the absolute value of the difference between the predictions and the targets. For instance, an MAE of 0.5 on this problem would mean our predictions are off by $500 on the average.

##
## Validating our approach using K-fold validation
##
##  To evaluate our model while we keep adjusting its parameters (such as the number of epochs used for training), we could split the data into a training set and a validation set, as we did in the previous examples. But because we have so few data points, the validation set would end up being very small (e.g., about 100 examples). As a consquence, the validation scores might change a lot depending on which data points we chose for validation and which we chose for training: the validation scores might have a high variance with regard to the validation split. This would prevent us from reliably evaluating our model. The best practice in such situations is to use K-fold cross-validation.

## K-fold validation
k <- 4
fold_id <- sample(rep(1:k, length.out = nrow(train_data)))
num_epochs <- 100
all_scores <- numeric()
for (i in 1:k) {
    cat("Processing fold #", i, "\n")
    val_indices <- which(fold_id == i)
    val_data <- train_data[val_indices, ]
    val_targets <- train_targets[val_indices]
    ## Prepare the validation data: data from partition #k.

    partial_train_data <- train_data[-val_indices, ]
    partial_train_targets <- train_targets[-val_indices]
    ## Prepare the training data: data from all other partitions.
   
    model <- build_model()   ## Build the Keras model (already compiled).

    model |> fit(
                 partial_train_data,
                 partial_train_targets,
                 epochs = num_epochs,
                 batch_size = 16,
                 verbose = 0
              )
    ## Train the model (in silent mode, verbose = 0).

    results <- model |>
    evaluate(val_data, val_targets, verbose = 0)
    ## Evaluate the model on  the validation data.

    all_scores[[i]] <- results[['mae']]
}

## Running this with num_epochs = 100 yields the following results:
all_scores

mean(all_scores)

## The different runs do indeed show rather different validation scores, from 2.1 to 2.4. The average (2.3) is a much more reliable metric than any single score—that’s the entire point of K-fold cross-validation. In this case, we’re off by $2,300 on average, which is significant considering that the prices range from $10,000 to $50,000. Let’s try training the model a bit longer: 500 epochs. To keep a record of how well the model does at each epoch, we’ll modify the training loop to save the per-epoch validation score log for each fold.


num_epochs <- 500
all_mae_histories <- list()
for (i in 1:k) {
    cat("Processing fold #", i, "\n")
    val_indices <- which(fold_id == i)
    val_data <- train_data[val_indices, ]
    val_targets <- train_targets[val_indices]
    ## Prepare the validation data: data from partition #k.

    ##Prepare the training data: data from all other partitions.

    partial_train_data <- train_data[-val_indices, ]
    partial_train_targets <- train_targets[-val_indices]

    ## Build the Keras model (already compiled).
    model <- build_model()
    history <- model |> fit(
                            partial_train_data, partial_train_targets,
                            validation_data = list(val_data, val_targets),
                            epochs = num_epochs, batch_size = 16, verbose = 0
                        )
    ## Train the model (in silent mode, verbose = 0).

    mae_history <- history$metrics$val_mae
    all_mae_histories[[i]] <- mae_history
}

all_mae_histories <- do.call(cbind, all_mae_histories)

## We can then compute the average of the per-epoch MAE scores for all folds.
average_mae_history <- rowMeans(all_mae_histories)

plot(average_mae_history, xlab = "epoch", type = 'l')

truncated_mae_history <- average_mae_history[-(1:10)]
plot(average_mae_history, xlab = "epoch", type = 'l',
     ylim = range(truncated_mae_history))

## As we see, validation MAE stops improving significantly after 100-140 epochs (this number includes the 10 epochs we omitted). Past that point, we start overfitting.

## Once we are finished tuning other parameters of the model (in addition to the number of epochs, we could also adjust the size of the intermediate layers), we can train a final production model on all of the training data, with the best parameters, and then look at its performance on the test data.

## Training the final model

## Get a fresh, compiled model.
model <- build_model()
## Train it on the entirety of the data.
model |> fit(train_data, train_targets,
              epochs = 120, batch_size = 16, verbose = 0)
result <- model |> evaluate(test_data, test_targets)

## We’re still off by quite a bit

## Generating predictions on new data

## When calling predict() on our binary classification model, we retrieved a scalar score between 0 and 1 for each input sample. With our multiclass classification model, we retrieved a probability distribution over all classes for each sample. Now, with this scalar regression model, predict() returns the model’s guess for the sample’s price in thousands of dollars:

predictions <- model |> predict(test_data)
predictions[1, ]

## The first house in the test set is predicted to have a price of about $8,993

