'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import time
import numpy as np
import keras
import tensorflow as tf
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
# from keras.preprocessing.text import Tokenizer # <- Module not working on Keras 3 

import mlflow
import mlflow.keras
import mlflow.tensorflow

# ----------------------------------------------
# Setup TensorFlow to use GPU with Metal acceleration (Mac M1/M2)
print("TensorFlow version:", tf.__version__)
print("Built with Metal:", hasattr(tf, '__metal_version__'))

gpu_devices = tf.config.list_physical_devices('GPU')
print(f"\n🎯 GPU devices found: {len(gpu_devices)}")

if gpu_devices:
    print("✅ Metal acceleration is working!")
    for gpu in gpu_devices:
        print(f"   - {gpu}")
else:
    print("❌ No GPU detected - using CPU only")

# Detect and log the device being used
if gpu_devices:
    device_type = "m3_gpu"
else:
    device_type = "cpu"

# ----------------------------------------------
# Set the tracking URI to connect to your local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Create or set experiment
experiment_name = "reuters-classification-keras-mlflow"
mlflow.set_experiment(experiment_name)

print(f"✓ Experiment '{experiment_name}' is ready")
print(f"View it at: http://127.0.0.1:5000")

# ----------------------------------------------
# Enable automatic logging of TensorFlow/Keras metrics, parameters, and models
mlflow.tensorflow.autolog()

# ----------------------------------------------
# Model parameters
# ----------------------------------------------
# # Configuration 1 (Baseline - your current):
# max_words = 1000
# batch_size = 32
# epochs = 5
# learning_rate = 0.001  # Adam's default

# # Configuration 2 (Larger vocabulary, more epochs):
# max_words = 2000
# batch_size = 32
# epochs = 10
# learning_rate = 0.001

# Configuration 3 (Different batch size and learning rate):
max_words = 1000
batch_size = 64
epochs = 5
learning_rate = 0.01  # You'll need to use Adam(learning_rate=0.01) for this

# ----------------------------------------------
print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

# Log parameters
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("max_depth", max_words)
mlflow.log_param("learning_rate", "adam")
mlflow.log_param("Epochs", epochs)

mlflow.set_tag("project", "reuters_classification")
mlflow.set_tag("dataset", "reuters")
mlflow.set_tag("model_type", "MLP")
mlflow.set_tag("status", "baseline")
mlflow.set_tag("hardware", device_type)


# # The Tokenizer module is not available in Keras 3, so we will use an alternative method to vectorize the sequences
# print('Vectorizing sequence data...')
# tokenizer = Tokenizer(num_words=max_words)
# x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
# x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

# Replace the Tokenizer lines with this manual approach:
print('Vectorizing sequence data...')
def sequences_to_matrix(sequences, num_words):
    """Manual implementation of binary mode matrix conversion"""
    matrix = np.zeros((len(sequences), num_words))
    for i, seq in enumerate(sequences):
        for word_index in seq:
            if word_index < num_words:  # Only consider words within vocab size
                matrix[i, word_index] = 1
    return matrix

x_train = sequences_to_matrix(x_train, max_words)
x_test = sequences_to_matrix(x_test, max_words)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# Ensure no active run exists before starting a new one
mlflow.end_run()
with mlflow.start_run(run_name=f"dense-512-lr-{learning_rate}-epochs-{epochs}-v3") as run:

    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # mlflow.tensorflow.autolog()

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    start_time = time.time()

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)
    
    end_time = time.time()
    training_duration = end_time - start_time
    
    score = model.evaluate(x_test, y_test,
                        batch_size=batch_size, verbose=1)
    
    # After training, calculate custom metrics
    train_accuracy = history.history['accuracy'][-1]  # Final training accuracy
    val_accuracy = history.history['val_accuracy'][-1]  # Final validation accuracy

    
    # Log metrics
    mlflow.log_metric("final_train_accuracy", train_accuracy)
    mlflow.log_metric("final_val_accuracy", val_accuracy)

    mlflow.log_metric("test_loss", score[0])
    mlflow.log_metric("test_accuracy", score[1])
    mlflow.log_metric("training_time_seconds", training_duration)

    # Store run_id for later use
    run_id_v1 = run.info.run_id
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Run ID: {run_id_v1}")
    print(f"Training time: {training_duration:.2f} seconds")
    print(f"\nMetrics:")    

    print(f"  - Test loss:     {score[0]:.4f}")
    print(f"  - Test accuracy:  {score[1]:.4f}")
    print("\n✓ Model logged to MLflow")