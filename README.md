# Ciphertext-Project
MSDS692 Data Science Practicum I

## Securing Information 
Keeping information private has been a part of human history since the beginnings of society and spans every type of communication. Encoded messages have been used by oppressed social groups to unite a community,  by warring states to enact covert soldier movements (Enigma), by sociopaths in an attempt to boast their activities (Zodiac), and by the modern public performing personal/business transactions online (Hash). However, as forms of communication evolved, so has the ability to “crack” these codes. Therefore, more advanced forms of encryption are necessary in order to find the balance of protection and usability. A solution to this dilemma could be found in the maturing abilities of machine learning. Machine learning algorithms are becoming better at grouping characteristics between similar products (Recommender Systems), recognizing different faces (Image Recognition), and even making decisions in response to the current environment (Self-driving Vehicles). These capabilities are being accomplished using the available computational and cost resources. As a result, machine learning may offer the key to creating methods to protect our personal information while allowing us to take advantage of the electronic age.  

# Homomorphic Encryption
One form of protection that machine learning is assisting is in the form of homomorphic encryption. The concept of homomorphic encryption involves the ability to encode information and not require the need to decrypt it for transactional use. With this capacity, businesses and individuals could find comfort in expanding the use of more types of personal information which would also create new electronic opportunities for the public. 

This idea has been around since the release of one of the first public cryptosystems (RSA, 1977) but early attempts were not computationally viable or only allowed single-type transactions (addition or multiplication). However, the ideal goal is to embed the ability to perform all possible operations on encrypted data to eliminate the inherent security weakness of exposing the decrypted data in order to use it. This ability is known as fully homomorphic encryption (FTE) or cryptocomputing. Recently, research involving complex machine learning methods is gaining momentum in getting closer a solution. 

## Project Overview: 
As inspired by the advancements in developing a resource-efficient FHE scheme by utilizing machine learning and its ability to recognize patterns, this project will explore the use of machine learning techniques as applied to ciphertext cryptography to identify its ability to “unwind” the encryption pattern. In addition, this project will personally provide experience in text processing in both the Python and R programming environments. As a result of using multiple platforms (R and Python) as well as the computational time required to execute many of the actions, some code will not executed in this environment. Actual code can be found in other files included in this Github site. 

The project will consist of four stages (links to those pieces of the project are below): 

Data Exploration and Prep - below

First Attempt: https://github.com/ckornafel/Ciphertext-Project/blob/master/First_Attempt.html

Second Attempt: https://github.com/ckornafel/Ciphertext-Project/blob/master/Second_Attempt.html

Third Attempt: https://github.com/ckornafel/Ciphertext-Project/blob/master/Third_Attempt_and_Conclusion.html


## Project Hypothesis:
Current machine learning methods are capable of sophisticated pattern recognition (e.g. facial, handwriting, image, etc.) and can achieve a high degree of success in its predictions. This ability to recognize similar patterns in a given set of training data (e.g. images, product descriptions) could be utilized in a text-only environment. By identifying reoccurring textual “patterns” within encrypted text blocks and comparing them with known plain text examples, the machine learning model should produce adequate prediction results to match plain text sentences with their encrypted ciphertext equivalents. 

Caveat: Since cipher schemes are designed to resist decryption, I am not expecting a high success rate in prediction. 

## The Data: 
The data sets that will be used are provided by Kaggle for the closed “Cipher Challenge III” competition. The challenge contained two data sets:  a plain text training set and a ciphertext testing set. The plain text set included 108,755 lines of Shakespeare’s plays. The ciphertext set consisted of 108,755 corresponding lines; however, each line was encrypted with a possible, four layers, of known cipher techniques. For this project, I will focus only on the single-layered ciphertext (level 1) with a stretch goal of attempting an additional layer if the models proved to be successful. 

These data sets were specifically chosen due to the Shakespearean plain text. Homomorphic encryption allows for the use of encrypted data without the need to unencrypt it, and Shakespeare’s Old English prose can be viewed as a type of encryption when compared to modern English (to a small degree). Therefore, the use of this text provides an “entry-level” step into machine learning using already encrypted data. Again, if successful, including the second later encrypted data could further show machine learning’s ability to recognize patterns in known data. 

While the lines of Shakespeare text vary in length, the ciphertext contained alpha-numeric "padding" to increase each sentence/line length to the next 100 characters. This padding introduces a random element that I will need to account for. 

The Cipher text Challenge III data can be found: https://github.com/ckornafel/Ciphertext-Project/tree/master/Code


## Text Cleaning
Typically, several text "cleaning" techniques would be applied to the text prior to exploring to focus attention only on key elements. Cleaning would include the removal of punctuation, stop words, and capitalization (among other processes). However, since each character (including capitals, punctuation, etc.) may represent a crucial pattern in the encrypted text (e.g. an "a" may be translated to a "!") no cleaning will be performed. Additionally, the removal of stop words (typically the most frequent) could reduce the size of potential patterns within the text data, complicating the training/prediction process. 

# Loading and Exploring the Data
Due to the "closed" status of the Kaggle Challenge and the possibility that the data would become unobtainable, copies of each data set was added to the Git Repository. 

# Loading the Libraries
```{r include=FALSE}
library(readr)
library(stringr)  #String Manipulation Functions
library(tm) #Corpus Functions
library(wordcloud) #For Wordcloud Graphic/Plot
library(ggplot2) #Plotting Functions
library(plyr) #Functional Shortcuts
library(dplyr) #Load after plyr to prioritize on stack for speed of functions
library(gridExtra) #Display multiple ggplot on a single page
```

# Loading the Data
```{r}
#Loading the Data 
##Datasets have been copied to the website
train <- read.csv("https://raw.githubusercontent.com/ckornafel/cyphertext/master/ciphertext-challenge-iii/train.csv",
                 stringsAsFactors = FALSE)
test <- read.csv("https://github.com/ckornafel/cyphertext/raw/master/ciphertext-challenge-iii/test.csv",
                 stringsAsFactors = FALSE)

```

# Plain text Data set
I am expecting to see three columns: An ID, Plain text, and an Index Value
```{r}
head(train[order(train$index),],5)
```
As expected, the training data set contains three columns, the plain text id, the Shakespearean plain text, and the index key. 
The plain text sample shows that each line/row of text could be partial examples of play lines, including randomly included titles. It does appear that some of the text can be grouped by certain whole Shakespearean works (King Henry) and broken out by scene line.
Since the first part of King Henry IV contains 33 lines, I wanted to know what the next block of text might include. 
```{r}
train[which(train$index > 33 & train$index < 39),]
```
Westmoreland is the next section of King Henry IV after King Henry's part (scene I) - it appears that the plain text is organized as multiple plays and each line represents one line of plain text. 
```{r}
length(unique(train$text))
```
108755 unique lines of text matches that total number of rows in the training data set. Therefore, it signifies that there are no repeated lines which
could complicate dicphering the cyphertext, given that there are multiple ciphers applied to the entire set. 

# Cipher text 
```{r}
str(test)
```
The test (cyphertext) data set also contains three variables: ciphertext id, ciphertext, and difficulty. The difficulty value indicates the number
of ciphers used on the plain text. E.g. level one indicates that a single cipher was applied, level two indicates that an additional cipher was applied
to the first ciphertext, etc. 

# Splitting the Test Data set into Four Levels
```{r}
test1<- test[test$difficulty==1,]
test2<- test[test$difficulty==2,]
test3<- test[test$difficulty==3,]
test4<- test[test$difficulty==4,]
rm(test) #Conserving memory
head(test1$ciphertext) #Viewing the Level 1 CipherText
```

## Terms/Words
Both data sets contained lines of text (sentences) which could be broken down into smaller text-items. For the next part, I will be focusing on the words/terms which comprise each of the text lines. Perhaps there may be some pattern that I can use to create numeric vectors for the machine learning algorithms. 

```{r trainltrfreq, echo=FALSE,  out.width = '100%'}
knitr::include_graphics("Train_term_freq.png")
```
It appears that there are only a small subset of terms (approx 7) which appear with high frequency. One term appears over 25,000 times
but the next most frequent term drops to below 10,000 occurrences in the complete Shakespearean text. 
There are 27937 terms that appear only once in the Shakespeare text. This highlights that slightly more than half (51.33%) of the entire
set of terms are unique combinations of letters/punctuation. 

# Most Frequent Terms in the Train (Plain text) Set
```{r trainwd, echo=FALSE, out.width = '100%'}
knitr::include_graphics("train_wc.png")
```
The most frequent Shakespeare terms are similar to those found in modern English (e.g. and, the, not, etc.). Normally, in text mining exercises, 
these common (stop) words would be removed in order to focus on the more impactful words in the corpus. However, since I am working with cyphertext
every word (and punctuation) could be represented in the ciphertext and therefore needs to remain in the plain text. 

# Most Frequent Terms in the Test (Cipher text) Set
```{r testwd, echo=FALSE,  out.width = '100%'}
knitr::include_graphics("test1_wc.png")
```
The most frequent terms in the test1 data set appear to be similar in length (short) to the plain text frequent words. Although, there seem to a few words (e.g. xwd, ssiflt) that have much more frequency. This is because the training set is only a quarter of the size of the training set. There are no terms in test 3 which appear only once. The lowest frequency of occurrence are two terms which appear twice and three times in the cipher text. Given the large volume of individual number combinations, it may indicate that they represent letter pairs or phonetic sounds instead of whole terms. 

# Comparing Word Frequency between Train and Test
```{r}
WordCount <- function(x){
  return(length(unlist(strsplit(as.character(x), "\\W+"))))
  
}
train$num_term<-sapply(train$text, WordCount)
table(train$num_term)

test1$num_term<-sapply(test1$ciphertext, WordCount)
table(test1$num_term)
```
It does not appear that the count of terms for each section of Shakespeare Text correlates well with the count of terms from 
each of the cipher texts. Those term counts that do align (e.g. one instance of a 49-term length Shakespeare text) also appear in
multiple cipher texts (e.g one instance of a 49-term Test 1 text and one instance of a 49-term Test 2 text). Given the assumption 
that there is no overlap of cipher to plain texts, it would appear that spaces may not term separators in the cipher texts. 

Additionally, there are 416 instances of a single-term Shakespeare text but the smallest number of cipher text terms is seven in 
both test1 and test2 sets. Either these single-term texts are hidden in test3 and test4, or another indication that spaces are not
term separators in the cipher texts. 

# Groups of Terms
```{r}
term_freq <- train %>%
  group_by(num_term) %>%
  summarise(counts = n())

ggplot(head(term_freq,20), aes(x=num_term, y = counts ))+
  geom_bar(fill = "steelblue", stat = "identity")+
  theme_minimal()
```
It looks like the training set is mainly comprised of short (<15) worded terms. However, given that this set represents all four cipher levels and we are only focusing on one, the actual distribution of the reduced set may be smaller. 

## Characters 
While the above section explored the individual words/terms within each line of text. This next section will dive into the individual characters which make-up those words/terms. 

# A quick check to verify that the same characters are used for train and test1
```{r}
all_text_train <- paste(train$text, collapse= "")
all_text_test1 <- paste(test1$ciphertext, collapse = "")
uniq_chr_train <-as.vector(unique(strsplit(all_text_train, "")[[1]]))
uniq_chr_test1 <-as.vector(unique(strsplit(all_text_test1, "")[[1]]))

sort(uniq_chr_train) == sort(uniq_chr_train)
sort(uniq_chr_train) == sort(uniq_chr_test1)
```
It is known that the plain text and ciphertext did not contain any unique characters and were comprised of alpha letters (upper/lower case), punctuation, and spaces. This confirms that the two data sets did not include characters that were not also used in the other. Hopefully this will make it easier for the machine learning models to identify patterns. 

```{r ltrcomp, echo=FALSE, out.width = '105%'}
knitr::include_graphics("letter_comp.png")
```
The comparison of the characters highlights a potential substitution cipher being used. I've included the second layered ciphertext set for comparison. According to this chart, it could be assumed that the space character is not changed when encrypted. This means that individual words have the same separator (space) after encryption. Additionally, this plot could identify that all plain text e's are exchanged for ciphertext s's. 
Another possible assumption could be made between the level 1 and 2 encryption since the character frequencies are so similar. This could indicate that they are similar cipher techniques but with different rotating letters. 
It is also noted that punctuation does not appear in the ciphertext frequencies, this could indicate that punctuation marks are not used as substitutions for alpha letters. 

# Exploring the punctuation assumption further
The above plot highlights a unique pattern within the ciphertext that indicates that punctuation is not encrypted. 
```{r punctcomp, echo=FALSE,  out.width = '105%'}
knitr::include_graphics("punct_freq.png")
```
Given the order of frequencies for each punctuation character, it does appear that punctuation remains the same between test and train. 

#Examining Character Counts by Line
```{r chrcnts, echo=FALSE,  out.width = '105%'}
knitr::include_graphics("chr_cnts.png")
```
The above plot shows that the majority of the training (plain text) data falls below 100 characters. This indicates that the majority of the corresponding ciphertext includes over 50 additional (random) characters as "padding". These extra characters will provide a challenge for the ML models as it introduces randomness in any potential pattern for the largest portion of the data set. However, having few examples of longer character lines could help identify the padding scheme which could then be removed (and reduce the randomness). 

```{r padbox, echo=FALSE,  out.width = '105%'}
knitr::include_graphics("pad_box.png")
```
The box plot for the padded character amounts show that the majority of the padding occurs within the 0 - 100 character rows. 
This group (100) has an average of approx. 60 additional characters added to the Shakespeare text. However, it also has a range of up to 99 
additional characters - having the largest spread of padding. The 700, 900, and 1100 (largest) groups have the fewest members and consist of approx
 27, 60, and 72 (respectively) additional characters. I assume that the low number of these larger text blocks will compensate for the additional characters
when predicting the plain text. 
The above plot also indicates a large amount of outlying padding characters for the 100-char population. This could highlight a potential issue with the random-factor for the largest group and obscure predictive patterns. 

```{r setup, include=FALSE}
knitr::knit_engines$set(python=reticulate::eng_python)
```
```{r, include=FALSE}
library(reticulate)
use_python("/Users/ckornafel/anaconda3/bin/python")
```

## First Attempt
The first attempt at analyzing the text included tabulating:
* Using the entire train (108,757 rows) and test1 (27,158) including padded characters. 
* The total number of characters per line (plain text adjusted for padding)
    + This figure should help match groupings of plain text and potential cipher text matches. While the largest population for both populations is 100 characters, it could highlight patterns within the lower volume groups
* The total number of capital letters included in each line. 
* The total number of lowercase letters included in each line. 
* The total number of punctuation characters
    + It was determined that punctuation was not encrypted from the plain text.
    + However, the cipher text data includes additional punctuation as they are also included in the random character padding. 
* Generalized individual character frequencies by line that are sorted: most frequent to least frequent 
    + This included 52 additional variables 
    + Columns were reordered by row to generalize actual characters (since they would be different between the two sets)
* The data sets would be analyzed using H2O DeepLearning Forward Feeding Neural Network and Sklern's SVM
    + The actual text lines were used for the H2O model as it automatically one-hot encodes categorical variables. 
    + The actual text was removed when building the SVM model
* The index value from the Train data set was added to the test data set to be used as the response variable. 
    + The correct index of matching plain/cipher text was used to determine how successful the predictions were. 
* The train set consisted of only plain text and the test set consisted only of cipher text (as structured by Kaggle)

# Examples of Data sets
```{python}
import pandas as pd
train = pd.read_csv('pred_train.csv')
test1 = pd.read_csv('pred_test1.csv')

train.head(10)
test1.head(10)
```

I also created a smaller sample data set, as the training of certain models errored out after processing too long
```{python}
train_sample = train.sample(n=900, random_state=1)
test1_sample = test1[test1['index'].isin(train_sample['index'])]

#Vlaidating that all test classes are still in train
test1_sample['index'].isin(train['index']).value_counts()
```
So the sampled test data set matches the sampled train data set cases - validating by index


## H2O Neural Network Implementation
```{r a1h2odata, echo=FALSE,  out.width = '100%'}
knitr::include_graphics("a1h2odataload.png")
```
The two data frames were successfully parsed and converted to H2O frame types. 

# Issue #1
After attempting to run the full train set (108k+) lines in H2O, I discovered that H2O caps the number of classes must be under 1000. Since each plain text sentence was its own class, I needed to adjust and send the sampled data frames through instead

```{r a1h2odata2, echo=FALSE,  out.width = '100%'}
knitr::include_graphics("a1h2odataload2.png")
```
As we can see, the Deeplearning model failed out (due to too many classes) and smaller data frames were loaded instead. 

# Scoring History
```{r a1h2oscore1, echo=FALSE,out.width = '100%'}
knitr::include_graphics("a1h2oscore1.png")
```
As shown in the scoring history plot above, the log_loss for validation dropped steeply after the 1st epoch. I used only 10 epochs to keep the model from crashing, but it does not appear that using a larger number would have improved the results

# Variable Importance
```{r a1h2ovarimp1, echo=FALSE,  out.width = '100%'}
knitr::include_graphics("a1h2ovarimp1.png")
```
The model did produce a variable importance chart shows that the most critical variables were letter frequencies (V10 = 10th most frequent character) that measured the larger sentences. Given that the most common length was 100 characters, this would indicate that the longer (and more rare) strings were able to identify the pattern better than the shorter strings. It also shows that the count of punctuation was also important - given that it was already discovered that punctuation was not changed within the cipher used. 

```{r a1h2oerror1, echo=FALSE,  out.width = '100%'}
knitr::include_graphics("a1h2oerror1.png")
```
The final output of the model shows that it was not a very accurate and well fitted one. The MSE and RMSE were almost 1 indicating a large amount of error generated. 

# Cross Validating the Sample Model
```{r a1h2ocrossvalid1, echo=FALSE,  out.width = '100%'}
knitr::include_graphics("a1h2ocrossvalid1.png")
```                                                                                                                                                                                                                                                                                         

Performing the cross validation did not highlight any possible improvements. The mean accuracy was 0.11% and a corresponding error rate of almost 100%

After adjusting the number of hidden layers and decreasing the epoch, the MSE did not improve (and in some cases actually got worse)

# Best Model Predictions
```{r a1h2obestmodpred1, echo=FALSE,  out.width = '100%'}
knitr::include_graphics("a1h2obestmodpred.png")
```
After determining the "best" fit model, I used the sample test set for prediction. The MSE for the prediction achieved a horrible 99.785% error rate (worse than just guessing)


## SVM
When building the SVM model, the complete data set proved to be too large - so the sample data set was used instead. 
Prior to training the model, the data sets were encoded and scaled using StandardScaler() function. 
The gamma value was set to "auto" 
# The Results
```{r a1svmacc, echo=FALSE,  out.width = '100%'}
knitr::include_graphics("a1svmacc.png")
```
After fitting/training the model and then using the sample data sets to predict, the ending accuracy rate was determined to be 0.0 (I have never seen a score so low!)

## Conclusion
This attempt showed that, given the train/test data sets, H2O Neural Network and SVM were unable to predict cipher text (given plain text examples) with any useful accuracy. While the NN performed much better than the SVM, it still did not preform as expected. 

Thoughts: 
* I believe that using each plain text line as a separate class proved to be a large issue. Unfortunately, reducing the number of examples also decreased the amount of repetitive patterns that could be used for accurate predictions. 
    + There is a balance of using too many examples with too many classes and providing enough examples to adequately train the model. 
* Using a custom embedding scheme represented numerical patterns, but it appears that these measurements did not provide enough data for machine learning to recognize patterns. 
* The results of the SVM (0%) were surprising as I was expecting anything else given that there are distinct character length groups represented in the data set. Although, not a powerful measurement, I would have expected some benefit. 
* I believe that there is an issue of using only plain text in the training set as it does not provide corresponding cipher text examples for the model to compare. There was a large amount of manual data exploration performed before building the test/train sets that a few examples could have been included. 
    + I was able to identify a few large length matches which I could have added to the known "training" data. 
 

```{r, include=FALSE}
library(reticulate)
use_python("/Users/ckornafel/anaconda3/bin/python")
```

## Second Attempt
After the abysmal performance achieved using the first modified data sets (attribute counts, plain text-only training set), I decided to take a different approach. I learned that I needed fewer classes than each row and I should attempt to include cipher text examples in the training set. Another issue that I identified was the random component of the cipher text character padding. In an effort to remedy some of these issues, I began to focus on the individual words within each string as opposed to viewing the entire string. 

While I was experimenting with the original data, I was able to determine the padding scheme used for the level 1 encryption. Each cipher text line split the number of random characters pre and post the true encrypted string (i.e. the actual Shakespeare text was located in the middle of the encrypted string). I made the decision to remove the padding so that I could expose the actual encryption pattern which would hopefully improve prediction accuracy. 
Additionally, I did notice that the encryption scheme used was a rotating poly alphabetic substitution which changed after every capital letter (similar to a Caesar Shift cipher). This highlights that manual effort achieved more than using the machine learning techniques already. However, I attempted to recreate the information that I used in breaking the encryption method. 

The new approach: 
* Listed individual words of plain text with their corresponding encrypted text
    + This processed greatly reduced the amount of classes due to the repetition of words throughout the Shakespeare text. However, there were still over 1000 combination of alpha/punctuation so I did cap the training size to 1000. 
* Words were separated via space characters (as identified in the data exploration section). 
    + Since both plain and cipher texts used the same spacing scheme, it produced equal number of word pairs. 
* All punctuation was included since it these characters represented similarities between each pair. 
* To further reduce file size, the total number of occurrences for each plain/cipher pair within the data sets was added. 
* The longest word/punctuation combination was 48 characters long.
    + This produced 48 individual columns for each word - shorter words had 0's filled in for the missing characters (I was adding my own padding!)
* The majority of the columns listed individual characters were of chr-typed that would be one-hot encoded prior to training the various models. 
* The models used include: Decision Tree, SVM, KNN, and H2O's Forward Feed Neural Network. 

# Data set Sample
```{python}
import pandas as pd
ltr = pd.read_csv('/Users/ckornafel/Desktop/MSDS692 Data Science Practicum I/cipher_ltr.csv')
ltr = ltr.fillna(0) #Filling in the missing values with a 0

ltr.head()
```

# One-Hot Encoding the Sample Set
```{python}
SAMPLE_SIZE = 1000
ltr_samp = ltr.sample(n=SAMPLE_SIZE) #Obtaining a random sample of 1000 items

p_target_samp = ltr_samp['plainwd'].astype('category') #coding the response variable
ltr_samp = ltr_samp.drop('plainwd', axis = 1) #Removing response from feature set

one_hot_samp = pd.get_dummies(ltr_samp) #One hot encoding the features

one_hot_samp.head()
```

As the example shows above, each of the character variables have been expanded based on their values to 598 columns. 

# Breaking the set into Train/Test and Scaling the Values
```{python}
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#Identifying feature and target
features_samp = one_hot_samp
target_samp = p_target_samp

#Sample Size Data
x_samp_train, x_samp_test, y_samp_train, y_samp_test = train_test_split(features_samp, target_samp, random_state = 0)

#Scaling the datasets
scaler1 = StandardScaler()
scaler1.fit(x_samp_train)
x_samp_train = scaler1.transform(x_samp_train)
x_samp_test = scaler1.transform(x_samp_test)

```
```{r a2scaler, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2scaler.png")
```


While not as sophisticated as word-type embedding, the one-hot encoding of scaled letter frequencies and positions does resemble a corpus-type format. My hope is that these vector values can provide enough information for the ML model to determine the cipher. 


## Decision Tree
After training/testing a decision tree with a max depth of 100, using the full data set, I found that it took a lot of time to process (too much for RMarkdown) and did not produce any viable results. For this example, I am using the reduced sample set and allowing the model to expand to whatever depth it needs to create pure leaves.

```{r include=FALSE}
library(pander)
panderOptions('digits', 5)
panderOptions('round', 5)
panderOptions('keep.trailing.zeros', TRUE)
```

```{python}
#Decision Tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report

##Reducing the dataset size so I can increase depth
#Creating the model without max depth value - this will fully extend each leaf
s_tree_mod = DecisionTreeClassifier()
s_tree_mod.fit(x_samp_train, y_samp_train)

s_tree_pred = s_tree_mod.predict(x_samp_test)

#Classification Report
s_tree_cr = classification_report(y_samp_test, s_tree_pred, output_dict=True)
acc = s_tree_cr["accuracy"]
print( "Accuracy of Decision Tree: ", acc)
print( "Precision of Decision Tree: ", s_tree_cr["macro avg"]["precision"])
```
The decision tree produced 0.4% accuracy and a precision of 0.25% precision. So this model did not produce accurate or repeatable predictions. However, these scores are marginally better than the previous attempt. Perhaps other models will yield better results. 


## KNN
Using KNN is popular for recommender systems that group similar items together based on their attributes. Hopefully, this model can find enough similarities within the data for better predictions than the decision tree. 
```{python eval = FALSE}
#KNN
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np

#Finding the K with the lowest error
error = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_samp_train, y_samp_train)
    pred_i = knn.predict(x_samp_test)
    error.append(np.mean(pred_i != y_samp_test))
```

```{r a2knneror, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2knnerror.png")
```
The KNN error provided a few models that had a slightly less error rate than the other processes, but overall poor results. However, I went ahead and attempted some predictions

# Predictions
```{python}
#KNN
from sklearn.neighbors import KNeighborsClassifier 
NEIGHBORS = 100
s_knn_mod = KNeighborsClassifier(n_neighbors = NEIGHBORS)
s_knn_mod.fit(x_samp_train, y_samp_train)
```
```{python eval = FALSE}
#Determining Accuracy
s_knn_mod_accuracy = s_knn_mod.score(x_samp_test,y_samp_test)
#print("KNN Accuracy: ", s_knn_mod_accuracy)

#Predict
s_knn_pred = s_knn_mod.predict(x_samp_test)

knn_cr = classification_report(y_samp_test, s_knn_pred, output_dict=True)

```
```{r a2knnacc, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2knnacc.png")
```
The accuracy did improve from the Decision Tree model, from 0.4% to 0.8% but the precision of the predictions dropped considerable. This would indicate that the improved accuracy (if one could call it improved) would not be consistently produced. The F1 score indicates that the true positives were very low in the prediction results so it appears that the model was better at predicting the negative cases. The likely source would be the high number of cases within the testing frame. 

# KNN Conclusion 
Again, using a random sample of the data set, I feel, is a drawback when using this model. Using a reduced sample allows the model to fit within a acceptable period of time (before the IDE times out) and it reduces the overall classes; however, it limits the available examples that could help precision and overall accuracy. 
The accuracy score was better than expected at this stage (although still horrible) but the corresponding low Precision and F1-Score reduce the overall improvement since it implies that lucky chance played a part. 

## SVM
The next model that I used is the support vector machine model which I also used with the previous data set. The repeated measurement could highlight if the modifications I performed on the train/test sets added benefit. 

I attempted two kernel shapes with the data, linear and ovo. The One vs One (ovo) kernel is specific for multiclass classification which trains data based on the number of classes available. 

The first linear SVM model performed as well as the previous one:
```{r a2svmlinacc, echo=FALSE,  out.width = '100%'}
knitr::include_graphics("a2svmlinacc.png")
```

However, the One vs. One model showed a little promise: 
```{r a2svmovoacc, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2svmovoacc.png")
```

Perhaps with some fine tuning the model would improve its performance

A grid search was performed using the kernels: linear, rbf, poly, and ovo with C values ranging from 1 to 10. However, the infrequent classes (there was a class with only one member) created issues with cross validating more than cv = 2. 
```{python eval=FALSE}
svc= SVC()
param = {'kernel':('linear', 'rbf', 'poly'), 'C': [1,10]}
grid = GridSearchCV(svc,param, cv=2)
```

Unfortunately, the limited grid search showed that the linear kernel and a C value of 1 was the best fit. Since the OvO kernel already yielded the "best" results, I used that model for prediction. 

```{r a2svmovopred, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2svmovopred.png")
```
We can see that the accuracy remained the same with predictions. However, the precision and F1-Scores were pretty bad. 

## SVM Conclusion 
I feel that using a reduced sample data set also played a role in the poor performance. Perhaps if I manipulated the data set to include only the most frequent cases it would have performed better. 
The SVM OvO model did perform better than the previous attempt at using SVM, which would indicate that the kernel shape is important for this type of fit. 

## H2O Forward Feed Neural Network
I used the H2O NN with the previous data set and only had to reduce the data size to accommodate its 1000 case maximum. Therefore, it was able to utilize more data in determining and recognizing text patterns. For this attempt (focusing on words instead of the entire string), the number of cases has already been reduced. This is further reduced to create the train/validation sets - so no data is "technically" omitted. Additionally, NN's require a lot of training data to generate results, so this adjustment should yield better performance. 
```{r a2sh2odataload, echo=FALSE, out.width = '70%'}
knitr::include_graphics("a2h2odataload.png")
```

As we can see, the data set that was loaded into H2O was successfully parsed and contained 1501 rows with 49 columns. At this stage, the V1.. variables are stored as factors and will automatically be one-hot-encoded. 

The H2O model constructed:
```{r a2sh2omod1, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2h2omod1.png")
```
The model was constructed with a small hidden layer to ensure completion and reduce running time, as well as using only 10 epochs. 

However, the results were very similar to the previous attempt
```{r a2sh2omod1res, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2h2omod1res.png")
```

This model was then cross-fold validated in an attempt to improve its performance. However, the number of cross-folds had to remain within the number of available cases and was set to nfolds = 3. This lower number was not expected to greatly improve results - which came true

```{r a2h2ocvmse, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2h2ocvmse.png")
```
The cross-validation only yielded a reduction of 0.0012 in MSE. Therefore, I decided to perform a grid search to discover any parameter adjustments that could help improve results. 

```{r a2h2ogridparam, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2h2ogridparam.png")
```

For the grid search, I used multiple hidden layers (although still on the small side to accommodate computer resources). I also introduced several L1 values which helps improve generalization (given the low frequency of multiple cases). 

```{r a2h2ogridout, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2h2ogridout.png")
```

The results of the grid search yielded models which continually increased its logloss values. As a result, I focused only on the first model with the lowest logloss. 

```{r a2h2ogridlogloss, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2h2ogridlogloss.png")
```
We can see from the above plot that the performance of the "best" model increased for every epoch prior to 40. Given that I was only using 10 epochs for the initial model, it could indicate that this value needs increasing

The MSE did not show  improvement for any of the possible models from the grid search. Therefore, I used the web interface to develop a more complex model (using many of the defaults). One of the benefits of using the web portal is that the models build much faster and with fewer computer resources. This allowed me to increase the size of the network 
```{r a2h2owebmod, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2h2owebmod.png")
```
As the above model shows, I was able to increase the hidden layer to 200, 200 nodes.

```{r a2h2owebper, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a2h2owebper.png")
```

The web-created model showed considerable improvement from the smaller network. This infers that large hidden layers are most likely necessary for this type of endeavor. 

## H2O Conclusion
Again, the neural network showed the most promise with this task when compared with the other models used in this iteration of the project. However, it was necessary to use the web portal to expand the neural network and capitalize on the power of hidden layers. 
I still feel that the lack of samples did have a negative impact on the overall performance. 


## Third and (hopefully) Final Attempt

After the abysmal results achieved during the past two attempts, I needed to reframe the problem and see if I could solve for some of the issues that I learned along the way...

Issues: 
* Too many classes
    + The problem with a rotating cipher is that each substitution changes after a certain textual event (for this one it changes after every capital letter). This produces that each cipher string/line can be unique, even if the plain text is the same - just at a different position.
* Too much / Not Enough data
    + Many of the models which used the full data set to train error ed out or failed to complete. Given the limited resources available for me, I need to reduce the volume of data that I use. However, reducing the number of examples creates an accuracy issue in that there are not enough samples to train adequately. Therefore, a balance/compromise needed to be found. 
* Textual Embedding are crucial but difficult to create
    + Because all the models require numeric representation of the text, it is crucial that I find as many meaningful measurements that can be applied to the text blocks. 
    + The H2O NN's provided variable importance when it generates a model. From this information, I was able to manually decipher the text, but also highlight the important features to include. 
    + The custom embedding that I used previously were very limited (single integer representations) which limit the prediction abilities. 

# Possible Solution: Create a language translation model
* Languages contain a large amount of independent cases (e.g. typically there are only a few exact choices for each word in another language)
  + However, there are also occurrences of "one to many" or "many to one" word/phrase correlations between different languages. This would accommodate the changes in cipher substitution.
* If I focus on the most frequent words/phrases within each (plain/cipher) then it may be possible to "translate" some of the text in each line to the degree that the remaining encrypted text can be extrapolated. 
* There are pre-defined word embedding that can be used to increase the numeric data "behind the scenes". 
    + Although I specifically chose the Shakespeare text because of its resemblance to encrypted English, there may be enough common words to exact benefit from pre-made embedding. 
    
## Tensorflow's LSTM Network
I decided to use a different Neural Network structure as H2Os Forward Feed produced the most favorable results but did not perform to expectation. Therefore, this next process will use Tensorflow with Python's Keras's front end, training a Long term Shor term Memory Network. As for the data, I removed the cipher text padding and aligned the plain text string with the "true" cipher text corresponding string. I also took the opportunity to remove punctuation as it was not encrypted (and increased the repetitive cases). 

Modified LSTM process from: https://stackabuse.com/python-for-nlp-neural-machine-translation-with-seq2seq-in-keras/, Usman Malik, 2019 - FRENCH LANGUAGE TRANSLATION

Attempting to create a translation dictionary of plain text and cipher text. This example will be using known plain text/cipher text relationships - however a similar
data set could be executed given the relative frequencies of each term in the text and matching those that are close. While the former method would likely reduce the accuracy of 
the "translation", it may offer enough correct terms to predict corresponding text relationships using by measuring their similarities. 

The first attempt used the entire, matched, data set (27k + lines). Unfortunately, the system timed out prior to completion. 

However.... 

```{r a3ltsmfull, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a3ltsmfull.png")
```
Before the build timed out, it registered an accuracy rate of over 89% on the first epoch! 

I reduced the size of the data set, knowing that it would also impact its final ability to successfully predict some of the text. 
```{r a3ltsmdata, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a3ltsmdata.png")
```
The reduced data set consisted of 9203 unique plain text words and almost triple that in the cipher text. The large increase indicates that the cipher text includes multiple substitutions for similar words. 
```{r a3ltsmlen, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a3lstmlen.png")
```
After the first few failed attempts at model completion, I discovered that this type of iteration performs better with string lengths of less than 50 characters. Therefore, I again modified the data set and re-ran the model. 

This instance also used GloVe's prebuilt 100-vector length, word embedding which was applied to the plain text examples (matched when available). 
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf]

# GloVe's Word Embedding
```{r a3ltsemb, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a3ltsmemb.png")
```
The 100 vector word embedding for a single plain text tokenized word. Using this information GREATLY increases the amount of numeric measurement that the network can use to fit the model. 

# Neural Network Configuration
```{r a3ltsmmap, echo=FALSE, out.width = '100%'}
knitr::include_graphics("ciphermod3.png")
```
The map of the network shows two embedding layers and two instances of LSTM. I am also using the prescribed number of nodes (256). This shows the combination of the input layer (plain text) joining with the second input layer (cipher text) and flowing into a dense network layer. 

```{r a3ltsmprog, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a3ltsmprog.png")
```
The model did take quite a while to cycle through the 10 training epochs, but it did complete. However, as a result of limiting the data set the accuracy rate achieved was only 75.6%. Not as good as the full-set model, but a marked improvement from the 0.8% accuracy rates from the previous attempts. 

# Prediction
Now that the LSTM model has been trained, it is time to see if it can successfully "translate" plain text into the correct cipher text

```{r a3ltsmpred, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a3ltsmpred.png")
```

The prediction output did not match the plain text string at all. However, given the repeated word (ihd), I feel that the issue lies within my code and not with the abilities of the model. 



## H2O Gradient Boost Machine with T
Given the possible success with the LSTM I decided to try H2O's Gradient Boost Machine using their new word2vec embedding function. The example of using this functionality was provided by H2O's website: https://github.com/h2oai/h2o-3/blob/master/h2o-r/demos/rdemo.word2vec.craigslistjobtitles.R (sebhrusen, 2019). 

I believe that some of the issues that I had with Tensorflows LSTM Network was a result of using the entire plain text/cipher text strings. Therefore, I modified the data set to capture only the first word of the plain text and filtered for specific Shakespeare characters. While this change will not identify specific lines of text, it will narrow down the section/block based on the speaker of the line. 
For this exercise I am using known text pairs so that I can determine the accuracy of the translation/decryption. However, if the pairs were unknown, the data set could be approximated by filtering for all capital plain and cipher words at the beginning of each line. 
Collapsing the plain text by only certain first-words does increase the repeated patterns that the machine can use for prediction. It also increases the variation of encrypted options so that searches can be performed on multiple cipher-words. 

# Modifying the Data set
```{r warning=FALSE}
library(utils)
library(stringr)
library(tidyr)
library(readr)
corp <-read_csv("/Users/ckornafel/Desktop/MSDS692 Data Science Practicum I/corp.csv", col_types = cols())

#Removing punctuation
corp <-as.data.frame(sapply(corp, function(x) as.character(gsub('[[:punct:]]+', " ",x))))

#Extracting the first two terms
corp$plain_fw <- word(corp$plain, 1,2, sep = " ")

#Reordering the dataframe columns (putting response in front) and dropping plain text column
corp <- corp[,c(3,2)]

names <- c("KING HENRY", "GLOUCESTER", "HAMLET", "BRUTUS", "QUEEN MARGARET", "MARK ANTHONY", "PORTIA", "FALSTAFF", "DUKE VINCENTIO", "KING LEAR",
           "PROSPERO", "TITUS ANDRONICUS", "IMOGEN", "ROSALIND", "MACBETH", "HELENA", "CORIOLANUS", "BIRON", "PRINCE HENRY")

#Filtering for the parts above
corp_names <- subset(corp, plain_fw %in% names) #sorting by the reduced plaintext terms

head(corp_names)
```
As the above example shows, the data set was reduced to a single plain text class word and the remaining cipher text string was included. The later will be processed through the word2vec function to identify word embedding

The GBM model was created using the base parameters and trained using a 80/20 split
```{r eval = FALSE}
#Tokenize the ciphertext
c_words <- tokenize(cipher_corp$cipher)

#Use the H2O word to vector function for word embeddings
w2v_model <- h2o.word2vec(c_words, sent_sample_rate = 0, epochs = 10) 

#Transforming the prediction into vectors to use in the GMB model
cipher_vecs <- h2o.transform(w2v_model, c_words, aggregate_method = "AVERAGE")


valid_cipher <- ! is.na(cipher_vecs$C1) #Checking for valid characters
data <- h2o.cbind(cipher_corp[valid_cipher, "plain"], cipher_vecs[valid_cipher, ])
data.split <- h2o.splitFrame(data, ratios = 0.8) #splitting the set into train and test

#Creating the GBM model 
gbm <- h2o.gbm(x = names(cipher_vecs), y = "plain",
                     training_frame = data.split[[1]], validation_frame = data.split[[2]])

#Make Predictions using a line from Duke V
deciphert("HKJP AXMNIDSTS  Cx exhdykjg ned ftat xay  jgpr", w2v_model, gbm)
```

# General (default) GBM Model Parameters
```{r a3h2ogbm, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a3h2ogbm.png")
```

# General (default) GBM Model Parameters
```{r a3h2ogbmacc, echo=FALSE, out.width = '70%'}
knitr::include_graphics("a3h2ogbmacc.png")
```

With the updated data set and the vectorized cipher tokens, the GBM model performed very well with a low MSE rate. This is the best performance noted to date. 

# Testing the Prediction Capability
I randomly selected a cipher string that corresponded with Duke Vincentio and attempted to see if the GBM model could successfully identify the speaker/part of the Shakespeare Play
```{r a3h2opred, echo=FALSE, out.width = '100%'}
knitr::include_graphics("a3h2opred.png")
```
As the output shows, the model did identify the correct plain text speaker/part of the play corresponding to the cipher text entered. Additionally, the model provided the top six possible matches with a numeric indication of their likelihood.

# H2O Conclusion
The final implementation of H2O GBM using the word2vec function required very little code yet produced the best results from this project. I do realize; however, that most of the benefit was derived from the greatly reduced data set (focusing only on certain first plain text words). This leads me to believe that multi-classification, using neural networks is a potential tool for cipher text encryption. However, it would require more computational resources than an average college student has access to. 


## Overall Thoughts
Many of the machine learning models were not able to successfully identify patterns within the encrypted text, given the resources available. However, there were some “glimmers” of hope with the use of neural networks. I believe that increasing the size of the network (adding more hidden layers) and expanding the data used could result in better performance. Building a viable fully homomorphic encryption scheme, in which the machine recognizes encoded data and is able to process it without decryption would require large, complex neural networks. 

It was shown that the type of data is very important when it comes to building a successful machine learning model. I found that modifying the text into a custom embedding scheme took a lot of effort but yielded very little results. Additionally, the fact that I manually uncovered the type of cipher being used for the challenge well before achieving any form of automated success, demonstrates that current (inexpensive) machine learning techniques are not cost effective for this type of project. However, the process did produce useful information, e.g. variable importance, which did assist with determining which character types I should focus on.
