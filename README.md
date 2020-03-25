# ML_Scrape_TwitterBot
This project explores web scraping, data clustering/exploration, RNN text production of user comments, and dispatching a twitter bot to tweet about articles. The data was extracted in July 2019. Please note that **this project was done purely for fun and for the sake of exploring these ML tools - I do not stand by anything that the bot produces**. 
That being said, check out the results of the twitter bot at : https://twitter.com/AltoParent
<br/>

### Table of Contents
- **[1. Scraping](#1-scraping)**<br>
- **[2. Data Exploration](#2-data-exploration)**<br>
- **[3. Text Clustering](#3-text-clustering)**<br>
    * **[3.1 Unsupervised KNN Clustering](#31-unsupervised-KNN-clustering)**<br>
    * **[3.2 Supervised Text Classification](#32-supervised-text-classification)**<br>
- **[4. GRU Commenter](#4-gru-commenter)**<br>
- **[5. Twitter Bot Assembly](#5-twitter-bot-assembly)**<br>

### To-Do
- Improve RRN model with different layer and node structure
- Add seperate RNN model to comment on sports articles

## 1. Scraping
The scraping portion of the project can be found in **newspaper_scrape.py**. A unique newspaper dataset is (respectfully) scraped from the Palo Alto Online website (https://www.paloaltoonline.com/) and organized into a pandas dataframe. The data collected includes:
- Page url
- Article title
- Article topic
- Posting day
- Posting month
- Posting year
- Post time
- Author
- Views of each article
- Comment # on each article
- Comment content
- Comment likes
- Comment neighborhood
- Article text

A total of 6536 articles and their accompanying content were extracted. Dating from 2015-2019. Scaping required roughly 12 hours with ~0.8s breaks between requests. 

## 2. Data Exploration
The extended code for the data exploration **newspaper_explore.py**. General content trends were explored, including the number of total articles produced per year and the the number of articles per topic.

<details><summary>Plotting Code (Click me)</summary>
<p>


```python
total_year_articles_unsorted = Counter(df['post_year'])

total_year_articles_sorted = {}

#Sort the order of the total yearly posts
for key in sorted(total_year_articles_unsorted.keys()):
    total_year_articles_sorted[key] = total_year_articles_unsorted[key]

fig1, ax1 = plt.subplots(figsize=(5.3, 4))
ax1.bar(range(len(total_year_articles_sorted)), total_year_articles_sorted.values(), align='center')
plt.xticks(range(len(total_year_articles_sorted)), list(total_year_articles_sorted.keys()), rotation=45)
plt.title('Total Articles Per Year')
plt.ylabel("Number of Articles")
plt.show()
```

</p>
</details>

<img src="https://github.com/cameronmcelfresh/newspaper_images/blob/master/fig_1_total_articles_per_year.png" width="300">

<details><summary>Plotting Code (Click me)</summary>
<p>

```python
total_topic_articles = Counter(df['topic'])

fig2, ax2 = plt.subplots(figsize=(5.3,6))
ax2.bar(range(len(total_topic_articles)), total_topic_articles.values(), align='center')
plt.xticks(range(len(total_topic_articles)), list(total_topic_articles.keys()), rotation = 90)
fig2.subplots_adjust(bottom=0.35)
plt.title('Total Articles Per Topic')
plt.ylabel("Number of Articles")
plt.show()
```
</p>
</details>

<img src="https://github.com/cameronmcelfresh/newspaper_images/blob/master/fig_2_total_articles_per_topic.png" width="300">



<br/><br/><br/>
More concisely, the number of articles per year can be visualize as a stacked bar plot. 
<details><summary>Plotting Code (Click me)</summary>
<p>

```python
for topic_type in all_topics:
    year_dict = Counter(df[df['topic']==topic_type]['post_year']).copy()    
    
    for year in total_year_articles_sorted.keys():
        if year not in year_dict.keys():
            year_dict[year]=0
    
    yearly_topics.append(sort_year_dict(year_dict))\
    
yearly_topics_reduced = pd.DataFrame.from_dict(yearly_topics)
yearly_topics_reduced.index=all_topics #Set the indicies to be the individual topics
yearly_topics_reduced = yearly_topics_reduced.transpose()

fig3_1 = yearly_topics_reduced.plot.bar(figsize=(5.3,4), stacked=True)
plt.title("Articles Per Year")
plt.ylabel("Number of Articles")
plt.show()
```
</p>
</details>

<img src="https://github.com/cameronmcelfresh/newspaper_images/blob/master/articles_per_year.png" width="300">

<br/><br/><br/>
The views per year and comments per year trend demonstrate which topics where of most interest to the readers and how that changed over the years. 
<details><summary>Plotting Code (Click me)</summary>
<p>
   
```python
#Views Per Year - Stacked Bar Plot
yearly_views_reduced = pd.DataFrame.from_dict(views_per_topic)
yearly_views_reduced.index=all_topics #Set the indicies to be the individual topics
yearly_views_reduced = yearly_views_reduced.transpose() #Switch the columns and indicies

fig5_1 = yearly_views_reduced.plot.bar(figsize=(5.3,4), stacked=True)
plt.title("Views Per Year")
plt.ylabel("Views")
plt.show()

#Comments Per Year - Stacked Bar Plot
for topic in all_topics:    
    yearly_comments = {}

    for year in all_years:        
        yearly_comments[year] = df[(df['topic']==topic) & (df['post_year']==year)]['comment_nums'].sum()
        
    all_yearly_comments.append(copy.deepcopy(yearly_comments))
    
all_yearly_comments_reduced = pd.DataFrame.from_dict(all_yearly_comments)
all_yearly_comments_reduced.index=all_topics #Set the indicies to be the individual topics
all_yearly_comments_reduced = all_yearly_comments_reduced.transpose() #Switch the columns and indicies

fig8_1 = all_yearly_comments_reduced.plot.bar(figsize=(5.3,4),stacked=True, sort_columns=True)
plt.subplots_adjust(left=0.2)
plt.title("Comments Per Year")
plt.ylabel("Number of Comments")
plt.show()
```
</p>
</details>

<img src="https://github.com/cameronmcelfresh/newspaper_images/blob/master/views_per_year.png" width="300"> <img src="https://github.com/cameronmcelfresh/newspaper_images/blob/master/comments_per_year.png" width="300">

<br/><br/><br/>
Visualizing the comments vs. views relationship, to no surprise as an article recieves more views, it tends to have more comments (or vice versa). 
<details><summary>Plotting Code (Click me)</summary>
<p>

```python
fig11, ax11 = plt.subplots(figsize=(5.3,4))
plt.scatter(df["views"], df["comment_nums"], alpha=0.5) 
plt.xscale("log")
plt.ylabel("Number of Comments")
plt.xlabel("Number of Views")
plt.title("Views vs. Comments")
```
</p>
</details>

<img src="https://github.com/cameronmcelfresh/newspaper_images/blob/master/views_vs_comments.png" width="300">


## 3. Text Clustering
### 3.1 Unsupervised KNN Clustering
The complete code for the text clustering can be found in **newspaper_classify.py**. A subset of the topics are selected and their article text is preprocessed to train a series of multiclass classifiers. The subset of topics chosen were:
- Crimes & Incidents
- Issues Beyond Palo Alto
- Schools & Kids
- Sports

#### Preprocessing 
The article text was preprocessed by removing special characters and symbols, removing all digits and single characters, and then lemmatizing the text. The article text is then 

<details><summary>Text Clean Up, Lemmatization (Click me)</summary>
<p>
   
```python
"""Clean up the articles! - remove any of the unknown characters and special characters"""
for i in range(len(df)):    
    df["article_text"].loc[i] = re.sub(r'\W', ' ', df["article_text"].loc[i]).lower() #Remove special symbols, make lowercase
    
    df["article_text"].loc[i] = re.sub(" \d+", " ", df["article_text"].loc[i]) #Remove all digits

    df["article_text"].loc[i] = re.sub(r'\s+[a-zA-Z]\s+', ' ', df["article_text"].loc[i]) #Remove single characters
        
    df["article_text"].loc[i] = re.sub(r'\^[a-zA-Z]\s+', ' ', df["article_text"].loc[i]) #Remove single characters from start
    
    df["article_text"].loc[i] = re.sub(r'\s+', ' ', df["article_text"].loc[i]) #Change any double spaces to single spaces
    
"""Lemmatize the word - remove any trailing s's"""
from nltk.stem import WordNetLemmatizer 

stemmer = WordNetLemmatizer() 

for i in range(len(df)):
    
    clean_article = df["article_text"].loc[i].split()
    
    clean_article = [stemmer.lemmatize(word) for word in clean_article]
    
    clean_article=' '.join(clean_article)
        
    df["article_text"].loc[i]=clean_article
    
#Finally, extracy all the clean articles
articles = df[['article_text']].copy()

articles = list(articles["article_text"].values)
 ````
</p>
</details>

#### KNN Clustering 
Prior to training the KNN model we need to tokenize and vectorize the text using the tfidVectorizer. A vocabulary of 300,000 features, stop_words='english', and max_df=0.7 was used. A min_df=0.08 was selected such that if a topic-specific word showed up in less than half of a topics' articles, it would be discarded as a poor predictor. 

<details><summary>KNN Building (Click me)</summary>
<p>

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=300000,
                                 min_df=0.08, stop_words='english',
                                 use_idf=True, ngram_range=(1,2))

tfidf_matrix = tfidf_vectorizer.fit_transform(articles)

terms = tfidf_vectorizer.get_feature_names()


num_clusters = 4

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)
```
</p>
</details>

The centroid of each cluster can be explored by printing the words nearest its position - to give an idea of the types of clusters that were found.  

<details><summary>Centroid Exploration (Click me)</summary>
<p>
   
````python
clusters = list(km.labels_) #Take the cluster names

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(4):
     print("Cluster %d:" % i, end='')
     for ind in order_centroids[i, :10]:
         print(' %s ,' % terms[ind], end='')
     print()
````
</p>
</details>

Output:
```
Cluster 0: county , city , ha , stanford , according , university , people , san , new , community ,
Cluster 1: game , team , stanford , season , point , play , cardinal , second , coach , menlo ,
Cluster 2: police , man , police said , officer , incident , department , alto police , according , woman , car ,
Cluster 3: school , student , district , board , teacher , parent , high school , education , school district , high ,
```
The clusters seem to line up fairly well with the selected list of subtopics! Howver it appears that there are some overlapping key words, for instance, "stanford" shows up for both cluster 0 and cluster 1 as fairly close to the centroid.  The cluster organization would suggest that;
- Cluster 0 = Issues Beyond Palo Alto
- Cluster 1 =  Sports
- Cluster 2 = Crimes & Incidents
- Cluster 3 = Schools & Kids
 

<br/><br/><br/>
Visualize the KNN clustering by using multidimensional sclaing of the cosine_similarity metric.

<details><summary>Multidimensional Scaling Code (Click me)</summary>
<p>
   
```python
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

MDS()

#convert to 2 components as we're plotting points in a x,y plane
#"precomputed" because we provide a distance matrix that was previously computed
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

dist = 1 - cosine_similarity(tfidf_matrix) #To be used for multidimensional scaling/plotting

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

cluster_names = {0: 'county, city', 
                 1: 'school, student', 
                 2: 'police, man', 
                 3: 'game, team'}

cluster_df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))

fig, ax = plt.subplots(figsize=(17, 9)) # set size

for key in cluster_names.keys():
    plt.scatter(cluster_df[cluster_df["label"]==key]["x"].values,cluster_df[cluster_df["label"]==key]["y"].values, 
                label = cluster_names[key] )
    
ax.legend()
plt.title("K-Means Clustering of Article Text")

plt.xticks([])
plt.yticks([])
```
</p>
</details>

<img src="https://github.com/cameronmcelfresh/newspaper_images/blob/master/KNN_clustering.png" width="500">

### 3.2 Supervised Text Classification
#### Preprocessing 
Similar to KNN clustering, the article data text will need to be vectorized and frequency-transofmred before using the bag-of-words approach on the training and testing data. Likewise, the article topics labels must be encoded before training.  

<details><summary>Text Preprocessing - Supervised Classification (Click me)</summary>
<p>

```python
count_vect = CountVectorizer(ngram_range=(1,2) , max_df = 0.7, 
                             strip_accents = "ascii", max_features=300000, 
                             stop_words = "english"  )

X = count_vect.fit_transform(articles) #create a vocab list for bag of words

##Move from occurences to frequencies
tfidf_transformer = TfidfTransformer(use_idf=False)

X_tf = tfidf_transformer.fit_transform(X)

#Split the data into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(X_tf, y_final, test_size=0.20, random_state=4, shuffle=True)
```
</p>
</details>

####

Then let's test several models using GridSearchCV, and see what performs best. 

<details><summary>Model Selection (Click me)</summary>
<p>
   
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


#Multinomial naive bias classifier
mnb_clf = MultinomialNB()
mnb_parameters = {'alpha':[1e-6, 1e-3, 1], 'fit_prior':('True', 'False')}

gs_mnb = GridSearchCV(estimator=mnb_clf,
			param_grid=mnb_parameters,
			scoring='accuracy',
			cv=5)

#Logistic Regression Classifier
lg_clf = LogisticRegression(random_state=4,multi_class='auto')
lg_parameters = {'solver':('saga','lbfgs'), 'max_iter':[200,500]}

gs_lg = GridSearchCV(estimator=lg_clf,
			param_grid=lg_parameters,
			scoring='accuracy',
			cv=5)

#Random forest classifier
rf_clf = RandomForestClassifier(random_state=4)
rf_parameters = {'n_estimators':[100,200,250], 'bootstrap':("true", "False")}

gs_rf = GridSearchCV(estimator=rf_clf,
			param_grid=rf_parameters,
			scoring='accuracy',
			cv=5)

#Linear Support Vector Machine
svc_clf = LinearSVC(random_state=4)
svc_parameters = {'C':[0.001,0.01,1], 'loss':("hinge", "squared_hinge")}

gs_svc = GridSearchCV(estimator=svc_clf,
			param_grid=svc_parameters,
			scoring='accuracy',
			cv=5)

grids = [gs_mnb, gs_lg, gs_rf,gs_svc ]
grids_title = ['MultinomialNB', 'LogistricRegression', 'RandomForest', 'LinearSVC']

bestAcc=0

for model,model_name in zip(grids, grids_title):
    print(model_name + ":")
    model.fit(X_train, y_train)
    
    print("Best parameters are %s" % model.best_params_)
    
    #Calculate and return the training and test accuracy
    print("Best Training Accuracy: %.3f" % model.best_score_)
    
    y_pred = model.predict(X_test)
    
    test_acc = metrics.accuracy_score(y_test, y_pred)
    
    print("Test Accuracy: %.3f \n" % test_acc)
    
    if (test_acc>bestAcc):
        bestAcc=test_acc
        bestModel = model
        bestCLF = model_name
        
print("The best classifier was "+ bestCLF + " with an accuracy of %.3f" % bestAcc)
```
</p>
</details>

Output: 
``` 
MultinomialNB:
Best parameters are {'alpha': 0.001, 'fit_prior': 'True'}
Best Training Accuracy: 0.858
Test Accuracy: 0.863 

LogistricRegression:
Best parameters are {'max_iter': 200, 'solver': 'saga'}
Best Training Accuracy: 0.854
Test Accuracy: 0.861 

RandomForest:
Best parameters are {'bootstrap': 'true', 'n_estimators': 250}
Best Training Accuracy: 0.855
Test Accuracy: 0.846 

LinearSVC:
Best parameters are {'C': 1, 'loss': 'hinge'}
Best Training Accuracy: 0.862
Test Accuracy: 0.855 

The best classifier was MultinomialNB with an accuracy of 0.863
```

Then we can plot the confusion matrix to see where the errors are coming from. 

<img src="https://github.com/cameronmcelfresh/newspaper_images/blob/master/confusion.png" width="500">


## 4. GRU Commenter
The complete code for GRU model building and training is listed in **newspaper_RNN_comments_TF.py**. I chose to use a GRU though a LSTM could be readily implemented. It should be emphasized that there is an inherent disadvantage working with this type of comment text because there are the topic and style of writing can vary wildly from comment to comment. 

Note that the code is modified to run in a Google Collab (https://colab.research.google.com/notebooks/intro.ipynb) notebook to take advantage of the online-accesible GPUs. Simple modifications could be made to run the code on a local machine. 

Some modifications were made to fit within the Google Collab environment. The file was mounted in my google drive so local data could be used and then authorized to save checkpoints so model weights could be reloaded later on.
<details><summary>Google Drive Preparation (Click me)</summary>
<p>

```python
#Mounting and Authorizing

#Mount the google drive and move into the folder with the data
from google.colab import drive
drive.mount('/content/gdrive')

import os

os.chdir('/content/gdrive/My Drive/Colab Notebooks') #Link specific to your gdrive

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

```

</p>
</details>

The comments were initially filled were characters that quickly inflated the size of the training set (>200 unique characters) to larger than reasonably manageable. A list of standard english characters were outlined and the comment text was preprocessed to only include tha allowed characters. 
<details><summary>Text Preprocessing - RNN Comments (Click me)</summary>
<p>

```python
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
```
</p>
</details>

The raw text characters could then be mapped to a dictionary that is defined by the approved characters in our vocab list. The comment text can then be passed to the RNN in this mapped format.
<details><summary>Mapping of text to RNN Digestible Format (Click me)</summary>
<p>
	
```python
#The unique characters in the file
vocab = sorted(set(list( list(string.ascii_letters + string.punctuation + string.digits + "\t\n ") )))
vocab_size = len(vocab)

#Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

#The maximum length sentence we want for a single input in characters
seq_length = 200
examples_per_epoch = len(text)//(seq_length+1)

#Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 1000

BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
```
</p>
</details>

A RNN was designed with 1024 RNN units and an embedding dimension of 256. The RNN model build is admittedly simple, and parallels the structure in Google's Tensorflow introduction to RNN text production. Training was done for ~200 epochs prior to taking the model offline.
<details><summary>RNN Model Build and Training (Click me)</summary>
<p>
	
```python
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
```

</p>
</details>

Text generation is done by feeding a starting string to the model, and extracting the complete predicted string as the resulting text.
<details><summary>Generating Text (Click me)</summary>
<p>

```python
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
```
</p>
</details>

Input:
```python
print(generate_text(model, start_string="I think "))
```

Output:
```
I think we have to provide more and more housing and buying people out of their cars.  Why would you want to comment in support of this issue? 
```


## 5. Twitter Bot Assembly
The complete code for building the twitter bot can be found in **newspaper_RNN_comments_TF.py**. Once the RNN model was trained and saved - it was uploaded to a free virtual machine on Google cloud. The code probes the Palo Alto Online Town Square page every several hours to check if a new article has been uploaded.

Prior to posting, the page must be scraped for new article titles and topics. 
<details><summary>Collecting Real-Time Article Information (Click me)</summary>
<p>

```python
#Find the link to the first article and get the title and topic
    urls = []
    extracted_topics = []
    views = []
    titles =[]
        
    page_url = "https://www.paloaltoonline.com/square/"
    
    #Extract the movie titles and IMDB classification
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
    counter=0
    for topic in """Find the link to the first article and get the title and topic"""
    urls = []
    extracted_topics = []
    views = []
    titles =[]
        
    page_url = "https://www.paloaltoonline.com/square/"
    
    #Extract the movie titles and IMDB classification
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
    counter=0
    for topic in extracted_topics:
        if topic in ["PALO ALTO ISSUES", "AROUND TOWN"]:
            selected_url = urls[counter]
            selected_title = titles[counter]
            break
            
        counter+=1    extracted_topics:
        if topic in ["PALO ALTO ISSUES", "AROUND TOWN"]:
            selected_url = urls[counter]
            selected_title = titles[counter]
            break
            
        counter+=1    
	
```

</p>
</details>

If an appropriate article (one with the topic being from Around Town/Palo Alto Issues AND it hasn't been previously commented on) is found, the title is dissected to create hashtags that could accompany the tweet. 

<details><summary>Use Chosen Article Title to Generate Hashtags (Click me)</summary>
<p>

```python

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
      
```
</p>
</details>

Once an article is chosen, a random slice of the title can be used as the starting string for the RNN generated comment.
<details><summary>Generate and format a Message (Click me)</summary>
<p>

```python
    #Generate a comment from our trained RNN
    if counter<len(extracted_topics):
        
        #Use a random 3-gram slice of the title as a starting source to generate the comment
        starting_string = ""
        
        #Split the raw title into its individual strings - this includes punctuation
        raw_split_title = selected_title_raw.split()
        
        try:
            
            #Select a random starting word from the first half of the title string
            startingWord = random.randint(0,int(round(len(split_title)/2)))
            the
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
    
```

</p>
</details>

Finally, tweepy is used to post the message to twitter and the program pauses for a number of hours before repeating the procedure!
<details><summary>Post the Message on Twitter and sleep the program! (Click me)</summary>
<p>

```python
        
    
    """Post a comment on twitter IF an appropriate topic was found"""
    if counter<len(extracted_topics):
        
        #Merge the comment and the link
        final_comment = generated_comment + "\n"
        
        final_comment = final_comment + final_hashtags
        
        final_comment = final_comment + " " + selected_url
        
        """Now post a response on twitter!"""
          
        # personal details  -- add your own
        consumer_key ="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        consumer_secret ="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        access_token ="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        access_token_secret ="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
          
        #Authorization of key and secret
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
          
        #Authorization of token and secret
        auth.set_access_token(access_token, access_token_secret) 
        api = tweepy.API(auth) 
          
        #Post the tweet!
        api.update_status(status =final_comment) 
        print(final_comment)
        
    
    
    """Sleep the program"""
    
    aDay = 60*60*24 #Number of seconds in a day
    
    random_sleep_time = random.randint(aDay-60*60*1.5, aDay+60*60*1.5)
    
    #Sleep the program for a random interval of time
    time.sleep(random_sleep_time)
    """Sleep the program"""
    aDay = 60*60*24 #Number of seconds in a day
    
    random_sleep_time = random.randint(aDay-60*60*1.5, aDay+60*60*1.5)
    
    #Sleep the program for a random interval of time
    time.sleep(random_sleep_time)
    

```

</p>
</details>

