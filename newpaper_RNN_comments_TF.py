#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:56:47 2020

@author: cmcelfresh
"""


"""Mounting and Authorizing"""

#Mount the google drive and move into the folder with the data
from google.colab import drive
drive.mount('/content/gdrive')

import os

os.chdir('/content/gdrive/My Drive/Colab Notebooks')

print("Current directory is " + os.getcwd())



#Authorize the drive to save files directy to the google drive
!pip install -U -q PyDrive #THIS COMMAND ONLY WORKS IN GOOGLE DRIVE
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive 
from google.colab import auth 
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()

drive = GoogleDrive(gauth)

import string
import random
import tensorflow as tf
import numpy as np
from os import path
import time


"""Load in the training data or generate it if it hasn't been generated"""

path_to_file = ("PaloAltoIssues_comment_text.txt")

#Check if the comment text has previously been saved
if path.exists("PaloAltoIssues_comment_text.txt"):
    #If the formatted string already exists, just load and use it
    print("PaloAltoIssues_comment_text file exists, loading it")
else:
    #If the string has not been prepared, prepare it and save it for future use
    comment_text = ""
    
    running_comment_length = 0
    
    #Select a subset of comments so the RNN can be more accurate for the topic
    #Use Palo Alto issues because there are the most comments
    df_subset = df[df["topic"] == "PALO ALTO ISSUES"].copy()
    
    
    #First get all the comments into a single string
    for i in range(int(len(df_subset))):
        for comment in df_subset["comments"].iloc[i]:
      
            comment_text+=comment.comment_content 
            comment_text+=" "
            
            #Keep a running tab of the length
            running_comment_length += len(comment.comment_content)
    
    
    #Calculate the average comment length - to be used in the trianing of the RNN
    avg_length = int(running_comment_length/df_subset["comment_nums"].sum())
    
    print("The average comment length is %.2f characters" % avg_length)
    
    #######################################Work on cleaning up the text later!!
    
    #Generate a list of the characters used
    chars = list(set(comment_text)) 
    
    chars_good = list(string.ascii_letters + string.punctuation + string.digits + "\t\n")
    
    #Find the unwated chars by the non-insection portin of the permitted characters and
    #the characters that were found in the comments
    unwanted_chars = set(chars_good)^set(chars)
    
    #Replace unwanted characters with spaces
    for i in unwanted_chars : 
        comment_text = comment_text.replace(i, ' ') 
    
    
    #Save the comment_text as a .txt file to be used later
    text_file = open("PaloAltoIssues_comment_text.txt", "w")
    text_file.write(comment_text)
    text_file.close()



# Read in the text file
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

#Shorten the total length to make text more managable
all_text = text

#Select a random slice so multiple training sessions use all the data
rand_start = random.randint(0,len(all_text)-20000000 )

text = all_text[rand_start:rand_start+20000000]


""""Prepare the text to be fed into the RNN"""

# The unique characters in the file
vocab = sorted(set(list( list(string.ascii_letters + string.punctuation + string.digits + "\t\n ") )))
vocab_size = len(vocab)

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 200
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 1000

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

"""Define and build the RNN model"""

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


#Checks to see if the training checkpoints folder exists - load training data
if os.path.exists(checkpoint_dir):
    print("Checkpoints found - loading latest checkpoint")
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
else:
    print("No checkpoints found")
    
"""Compile/train model and test it by generating text"""
    
model.compile(optimizer='adam', loss=loss)

EPOCHS=100

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 200

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))



print(generate_text(model, start_string="Test "))