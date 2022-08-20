#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install jupyterthemes


# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 


# In[6]:


# Load the data
tweets_df = pd.read_csv('twitter.csv')


# In[7]:


tweets_df


# In[ ]:


tweets_df.info()


# In[ ]:


tweets_df.describe()


# In[ ]:


tweets_df['tweet']


# In[ ]:


tweets


# In[ ]:





# In[10]:


# Drop the 'id' column
tweets_df = tweets_df.drop(['id'], axis=1)
tweets_df


# In[11]:


tweets_df


# # TASK #3: PERFORM DATA EXPLORATION 

# In[ ]:


sns.heatmap(tweets_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[ ]:


# Plot the histogram
tweets_df.hist(bins = 30, figsize = (13, 5) color = 'blue')


# In[ ]:


# Plot countplot
sns.countplot(tweets_df['label'], label = 'Count')


# In[ ]:


# Let's get the length of the messages
tweets_df['length'] = tweets_df['tweet'].apply(len)


# In[12]:





# In[ ]:


tweets_df.describe()


# In[ ]:


# Let's view the shortest message 
tweets_df[tweets_df['length'] == 11]['tweet']


# In[ ]:


tweets_df[tweets_df[]]


# In[ ]:


# Let's view the message with mean length 
tweets_df[tweets_df['length'] == 84]['tweet']


# In[16]:


# Plot the histogram of the length column


# # TASK #4: PLOT THE WORDCLOUD

# In[ ]:


positive = tweets_df[tweets_df['label']==0]
positive


# In[18]:


negative = tweets_df[tweets_df['label']==1]
negative


# In[19]:


sentences = tweets_df['tweet'].tolist()
len(sentences)


# In[20]:


sentences_as_one_string =" ".join(sentences)


# In[21]:


sentences_as_one_string


# In[22]:


get_ipython().system('pip install wordcloud')


# In[23]:


from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))


# In[24]:


negative_list = negative['tweet'].tolist()
negative_as_one


# # TASK #5: PERFORM DATA CLEANING - REMOVE PUNCTUATION FROM TEXT

# In[13]:


import string
string.punctuation


# In[14]:


Test = '$I love AI & Machine learning!!'


# In[15]:


Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'


# In[28]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed


# In[29]:


# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# # TASK 6: PERFORM DATA CLEANING - REMOVE STOPWORDS

# In[8]:


import nltk # Natural Language tool kit 
nltk.download('stopwords')

# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')


# In[9]:


Test_punc_removed_join = 'I enjoy coding, programming and Artificial intelligence'
Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]


# In[32]:


Test_punc_removed_join_clean # Only important (no so common) words are left


# In[10]:


Test_punc_removed_join


# In[16]:


mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'
mini_challenge_removed = [char for char in mini_challenge if char not in string.punctuation]
mini_challenge_removed_join = ''.join(mini_challenge_removed)
mini_challenge_removed_join


# In[19]:


mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'
mini_challenge_clean = [word for word in mini_challenge.split() if word.lower() not in stopwords.words('english')]
mini_challenge_clean


# # TASK 7: PERFORM COUNT VECTORIZATION (TOKENIZATION)

# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This paper is the second paper.','And this is the third one.','Is this the first paper?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)


# In[25]:


print(vectorizer.get_feature_names())


# In[26]:


print(X.toarray())  


# In[27]:


mini_challenge = ['Hello World','Hello Hello World','Hello World world world']

# mini_challenge = ['Hello World', 'Hello Hello Hello World world', 'Hello Hello World world world World']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(mini_challenge)
print(X.toarray())


# # TASK #8: CREATE A PIPELINE TO REMOVE PUNCTUATIONS, STOPWORDS AND PERFORM COUNT VECTORIZATION

# In[28]:


# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[29]:


# Let's test the newly added function
tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)


# In[30]:


print(tweets_df_clean[5]) # show the cleaned up version


# In[31]:


print(tweets_df['tweet'][5]) # show the original version


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
tweets_countvectorizer = vectorizer.fit_transform(tweets_df['tweet'])


# In[33]:


print(vectorizer.get_feature_names())


# In[34]:


print(tweets_countvectorizer.toarray())  


# In[35]:


tweets_countvectorizer.shape


# In[36]:


X = pd.DataFrame(tweets_countvectorizer.toarray())


# In[51]:


X


# In[37]:


y = tweets_df['label']


# # TASK #10: TRAIN AND EVALUATE A NAIVE BAYES CLASSIFIER MODEL

# In[38]:


X.shape


# In[39]:


y.shape


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[41]:


from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# In[42]:


from sklearn.metrics import classification_report, confusion_matrix


# In[43]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[44]:


print(classification_report(y_test, y_predict_test))

