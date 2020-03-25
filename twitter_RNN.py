

"""web page interactin imports"""
from bs4 import BeautifulSoup
import requests
import time


"""nlp imports"""
import string
import nltk
import re
from nltk import word_tokenize
from nltk.corpus import stopwords


"""twitter import"""
import tweepy 

"""Build the RNN model from preious data"""
import tensorflow as tf
import numpy as np
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #To shut off all tf warnings/deprecations

#Constants taken from training steps
vocab = sorted(set(list( list(string.ascii_letters + string.punctuation + string.digits + "\t\n ") )))
BATCH_SIZE = 1000
vocab_size = len(vocab)
rnn_units = 1024
embedding_dim = 256


#Building procedure for the text-generating model
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


# Recreate the mapping from the vocab character list
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


#Define the text generation model
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

"""
model = build_model(
      vocab_size = vocab_size,
      embedding_dim=embedding_dim,
      rnn_units=rnn_units,
      batch_size=BATCH_SIZE)

#Load the training data from the more recently saved checkpoint
checkpoint_dir = './training_checkpoints'
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
"""
checkpoint_dir = './training_checkpoints'

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

#Save a list of the articles that have been commented on so they aren't
#commented on twice
commented_articles_urls = [] 

#Time to make some posts!
while True:
    
    """Find the link to the first article and get the title and topic"""
    urls = []
    extracted_topics = []
    views = []
    titles =[]
        
    page_url = "https://www.paloaltoonline.com/square/"
    
    #Extract the unsegmented page info
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, 'html.parser')
        
    articles = soup.find_all('div', attrs={'style':'margin: 0 0 15px 0;'})
        
    for article in articles:
            
        #Extract the article topic and URL
        page_data = article.find_all('a')
           
        extracted_topics.append(page_data[0].text)
                
        extracted_url = "https://www.paloaltoonline.com/news" + page_data[1].get('href')[7:]
        urls.append(extracted_url)
                                    
        #Extract the title of the articles
        #text_header = article.text.split('\n')
        #titles.append(text_header[3].strip())
        titles.append(article.find_all('a')[1].get_text())
    
    """Select the most recent article that is a PALO ALTO ISSUES or AROUND TOWN type"""
    counter=0 #placeholder for the article to be commented on
    
    for topic in extracted_topics:
        if urls[counter] in commented_articles_urls:
            counter+=1
            continue
        
        if topic in ["PALO ALTO ISSUES", "AROUND TOWN"]:
            selected_url = urls[counter]
            selected_title = titles[counter]
            commented_articles_urls.append(selected_url)
            break
            
        counter+=1    
    
    
    """Extract the title and generate hashtags"""
    if counter<len(extracted_topics):
        
        #Try to extract a hashtag by removing stopwords
        nltk.download('stopwords')
        nltk.download('punkt')
        my_stopwords = set(stopwords.words('english'))
        
        #Save a save version of the title
        selected_title_raw = selected_title
        
        #Remove all of the punctuation
        selected_title = re.sub('[\W_]+', ' ', selected_title)
        
        split_title = selected_title.split()
        
        #A list to hold any approved tags
        approved_tags = ["PaloAlto"]
        
        
        #First try to extract any capitilized names they use
        names_mentioned = []
        for val in range(len(split_title)-1):
            word = split_title[val]
            next_word = split_title[val+1]
            
            if (word.title()==word and next_word.title()==next_word and (word.isalpha() and next_word.isalpha()) ):
                name = word+next_word
                name = name.translate(str.maketrans('', '', string.punctuation))
                
                if not(name=="PaloAlto"):
                    names_mentioned.append(name)
        
        #Find a non-stop-word verb and add it to the list
        filtered_words = []
        for title_word in split_title:
            if title_word not in my_stopwords:
                filtered_words.append(title_word)
        
        #Extract any adjectives and present participle verbs
        tagged_words = nltk.pos_tag(split_title)
        for word in tagged_words:
            if word[1] in ["NN","JJ", "VBG"]:
                
                #Only add the word if it doesn't have any training non-alpha characters
                if(word[0].isalpha() and len(word[0])>1): 
                    approved_tags.append(word[0].title())
        
        
        #Remove any duplicates
        approved_tags = list(set(approved_tags))
        
        max_tags = len(approved_tags)
        
        #Limit the number of possible hashtags to 3
        if max_tags>3:
            max_tags=3
     
        #Make sure that the hashtags only take up 50 characters at most
        while True:
            final_hashtags = ""
            
            num_tags = random.choice(range(max_tags))
            hashtags = random.sample(approved_tags, k = num_tags)
            
            for tag in hashtags:
                final_hashtags = final_hashtags + "#" + tag
            
            if(len(final_hashtags)<25):
                break
    
        #Add any names (of places, or organizations, or people) until 50 characters
        for name in names_mentioned:
            if(len(name) + len(final_hashtags)+1 < 50):
                final_hashtags = name + "#" + final_hashtags
    
    """Generate a comment from our trained RNN"""
    if counter<len(extracted_topics):
        
        #Use a random 3-gram slice of the title as a starting source to generate the comment
        starting_string = ""
        
        #Split the raw title into its individual strings - this includes punctuation
        raw_split_title = selected_title_raw.split()
        
        try:
            
            #Select a random starting word from the first half of the title string
            startingWord = random.randint(0,int(round(len(split_title)/2)))
            
            #Add the following words with proper capitilization
            for i in range(3):
                if i==0:
                    starting_string = starting_string + raw_split_title[startingWord].title()
                else:
                    starting_string = starting_string + " " + raw_split_title[startingWord+i]
              
            #Keep a trailing space at the end of the starting string    
            starting_string = starting_string + " "
            
            #Pick a different starting string if the randomly generated one overlaps 
            #punctuation
            unfit = False
            for value in range(len(starting_string)):
                if starting_string[value] in string.punctuation:
                    unfit=True
                    
            #If the generated starting string contains punctuation, revert to the 3-gram
            if unfit==True:
                starting_string = split_title[0] + " " + split_title[1] + " " + split_title[2] + " "
               
        #If the slice is out of range, then pick the first 3 words as the starting string
        except:
            starting_string = split_title[0] + " " + split_title[1] + " " + split_title[2] + " "
            
        generated_comment = generate_text(model, start_string=starting_string)
        #generated_comment = "Interesting article..."
        #generated_comment='Violent Committee for California, the Wells From A City Council meeting plan artists and it was.  The city is finally becoming a real estate developer that has a history of affordable housing.'
        
        acceptable_comment = False
        
        #re-generate comments until a basic structure of 3 or less sentences is met
        while not(acceptable_comment):
            
            #Don't allow any trailing "web link" text
            if ("Web link" in generated_comment) or ("Web Link" in generated_comment):
                continue
            
            #Make some rough approximations to try and keep the text to several 'complete'
            #sentences
            stop_points =[]
            for val in range(len(generated_comment)):
                if generated_comment[val] in ["?", ".", "!"]:
                    stop_points.append(val)
            
            #Select however many sentences is less than 200 characters
            if max(stop_points)<200:
                generated_comment = generated_comment[:max(stop_points)+1]    
                acceptable_comment=True
                
            else: 
                #Regenerate the comment if it doesn't fit
                generated_comment = generate_text(model, start_string=starting_string)
                #generated_comment = "Interesting article..."
    
    
    """Post a comment on twitter IF an appropriate topic was found"""
    if counter<len(extracted_topics):
        
        #Merge the comment and the link
        final_comment = generated_comment + "\n"
        
        final_comment = final_comment + final_hashtags
        
        final_comment = final_comment + " " + selected_url
        
        """Now post a response on twitter!"""
          
        # personal details  -- add your own
        consumer_key ="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        consumer_secret ="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        access_token ="XXXXXXXXXXXXXXXXX-XXXXXXXXXXXXXXXXX"
        access_token_secret ="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
          
        #Authorization of key and secret
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
          
        #Authorization of token and secret
        auth.set_access_token(access_token, access_token_secret) 
        api = tweepy.API(auth) 
          
        #Post the tweet!
        api.update_status(status =final_comment) 
        print(final_comment)
        
    
    
    """Sleep the program"""
    
    aDay = 60*60*3 #Number of seconds in a day
    
    random_sleep_time = random.randint(aDay-60*60*1.5, aDay+60*60*1.5)
    
    #Sleep the program for a random interval of time
    time.sleep(random_sleep_time)
