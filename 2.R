install.packages("keras")
install.packages("tensorflow")

library(tensorflow)
install_tensorflow()
library(keras)
library(tensorflow)

# Load dataset
fashion_mnist <- dataset_fashion_mnist()

x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

# Normalize (convert 0–255 → 0–1)
x_train <- x_train / 255
x_test <- x_test / 255

# Reshape for CNN (important)
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# Build CNN (6 layers + output layer)
model <- keras_model_sequential() %>%
  
  # Layer 1
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = c(28,28,1)) %>%
  
  # Layer 2
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Layer 3
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  
  # Layer 4
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Layer 5
  layer_flatten() %>%
  
  # Layer 6
  layer_dense(units = 128, activation = 'relu') %>%
  
  # Output layer
  layer_dense(units = 10, activation = 'softmax')

# Compile model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Train model
model %>% fit(
  x_train, y_train,
  epochs = 5,
  batch_size = 64,
  validation_data = list(x_test, y_test)
)

# Evaluate model
model %>% evaluate(x_test, y_test)

# Predictions for 2 images
pred <- model %>% predict(x_test)

for (i in 1:2) {
  predicted_label <- which.max(pred[i,]) - 1
  cat("Image", i,
      "Predicted:", predicted_label,
      "Actual:", y_test[i], "\n")
}
