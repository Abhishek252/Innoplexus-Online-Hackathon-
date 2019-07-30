#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd
import html
import nltk

# In[33]:

# Load  train and test data
train = pd.read_csv('train_F3WbcTw.csv')
test = pd.read_csv('test_tOlRoBf.csv')

# print shape of train and test data
print(train.shape, test.shape)
print(train.head())


stopwords = nltk.corpus.stopwords.words('english')

ps = PorterStemmer()
wl = WordNetLemmatizer()
ss = SnowballStemmer('english')

# Clean text 

def replace_ascii(data):
    '''
    this function take data and remove html data from the text
    '''
    data = html.unescape(data)
    return data

def clean_text(data):
    '''
    This function take data and preprocess data and return preprocess text.
    '''
    data['text_new']= data['text'].str.lower()
    data['text_new'] = replace_ascii(data['text_new'])
    data['text_new']=data['text_new'].str.replace("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9])","")
    data['text_new']=data['text_new'].apply(nltk.word_tokenize)    
    data['text_new']= data['text_new'].apply(lambda x: [wl.lemmatize(y) for y in x ])
    data['text_new']= data['text_new'].apply(lambda x: list(set(x)))
    data['text_new']= data['text_new'].apply(lambda x: [y for y in x if y not in stopwords])
    data['text_new']= data['text_new'].apply(lambda x: [y for y in x if len(y)>2])
    data['text_new'] = data['text_new'].apply(lambda x: ' '.join(x))

    
clean_text(train)
clean_text(test)
train['text_new'].head()


# In[34]:

def remove_frequent_word(data):
    '''
    Remove most frequent word from the text
    '''
    freq = pd.Series(' '.join(train['text_new']).split()).value_counts()[:20]
    m_freq = list(freq.index)
    data['text_new'] = data['text_new'].apply(lambda x: " ".join(y for y in x.split() if y not in m_freq))
    
def remove_rarer_word(data):
    '''
    Remove the most rarer word from the text.
    '''
    rarer = pd.Series(' '.join(data['text_new']).split()).value_counts()[-20:]
    rarer_freq = list(rarer.index)
    data['text_new'] = data['text_new'].apply(lambda x: " ".join(y for y in x.split() if y not in rarer_freq))


# In[37]:
# Remove Frequent word
    
remove_rarer_word(train)
remove_frequent_word(test)

# Remove Test word
remove_rarer_word(train)
remove_rarer_word(test)



# In[223]:


print("Train text: ", train['text_new'].head())
print("Test text: ", test['text_new'].head())
len(train.text[2]), len(train.text_new[2])



# In[225]:


sentiment = train['sentiment'].value_counts()
print("Sentiment: ", sentiment)


# In[226]:


# Visualize the percentage of Positive , Negative and Neutral sentiments

plt.figure(figsize=(5,5))
labels = "Positive","Negative", "Neutral"
sizes = [837, 617, 3825]
colors = ['green', 'red', 'lightblue']
explode = (0,0,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True, startangle=140)
plt.axis('equal')
plt.show()


# In[227]:
print("Train shape: {}, Test shape: {}".format( train.shape, test.shape))

# In[228]:

# #### Data preparation

# Combined drug data of train and test set to convert into vector form

a = pd.DataFrame(train['drug'])
a = a.append(pd.DataFrame(test['drug']), ignore_index=True)

print("a shape: {}".format(a.shape))
print(a.head())


# In[229]:

# Convert all the text into vector form
vec = CountVectorizer(max_df=.5, min_df=1, max_features=5000, stop_words='english') 
vec_text_new = vec.fit_transform(train['text_new']) 
vec_text_new_df = pd.DataFrame(vec_text_new.todense())

print("Shape of vectorized text data".format(vec_text_new_df.shape))


# In[230]:
# this is our second column('drug), we have to convert it into vextor form.
vec_drug = vec.fit_transform(a['drug']) 
vec_drug_df = pd.DataFrame(vec_drug.todense())

print("Shape of Vectorized drug data {}".format(vec_drug_df.shape))

# In[231]:

print("print vectorized drug data: ",vec_text_new_df.head())

# In[233]:

# Split train and test data of drug columns
vec_drug_df_train = vec_drug_df[:5279]
vec_drug_df_test = vec_drug_df[5279:]

# In[235]:
print("Shape of train data ",vec_drug_df_train.shape)
print("Shape of test data ",vec_drug_df_test.shape)

# In[237]:

# concatenat text and drug data
train_df = pd.concat([vec_text_new_df, vec_drug_df_train], axis=1)



# In[238]:
# #### Test data preparation
# Transform test data to vector form
vec_text_new_test = vec.fit_transform(test['text_new']) 
test_vec_text_new_df = pd.DataFrame(vec_text_new_test.todense())
print("Shape of the data: ", test_vec_text_new_df.shape)


# In[240]:
# Change index of the test drug data
vec_drug_df_test.index = range(2924)
test_df = pd.concat([test_vec_text_new_df, vec_drug_df_test], axis=1)

# Split data into train and validation set
xtrain, xvalid, ytrain, yvalid = train_test_split(train_df, train['sentiment'], random_state=140, test_size=0.22)


# In[244]:
#Model Building
# Logistic Regression
lreg = LogisticRegression(penalty='l2', solver='newton-cg', 
                          multi_class='multinomial')
lreg.fit(xtrain, ytrain) 
prediction = lreg.predict(xvalid) 
print("Macro F1- Score is: ", f1_score(prediction, yvalid, average='macro'))
print("Accuracy score of Model is: ", accuracy_score(prediction, yvalid))

# In[176]:
# Prediction on test data

test_pred = lreg.predict(test_df) 
test['sentiment'] = test_pred
submission = test[['unique_hash','sentiment']] 
submission.to_csv('lregression_lemm_pca_drug.csv', index=False) # writing data to a CSV file
