import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# 1. Read data
data = pd.read_csv("spam.csv", encoding='latin-1')

#--------------
# GUI
st.title("Data Science Project")
st.write("## Ham vs Spam")

# Upload file
uploaded_file = st.file_uploader('Choose a file' , type = ['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file , encoding = 'latin-1')
    data.to_csv('spam_new.csv', index = False)

# 2. Data pre-processing
source = data['v2']
target = data['v1']
# ham = 0, spam = 1
target = target.replace("ham", 0)
target = target.replace("spam", 1)

text_data = np.array(source)

# phan nay se duoc luu va load o phan predict cung voi model
count = CountVectorizer(max_features=6000)
count.fit(text_data)
bag_of_words = count.transform(text_data)

X = bag_of_words.toarray()

y = np.array(target)

# 3. Build model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

clf = MultinomialNB()
model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#4. Evaluate model
score_train = model.score(X_train,y_train)
score_test = model.score(X_test,y_test)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cr = classification_report(y_test, y_pred)

y_prob = model.predict_proba(X_test)
roc = roc_auc_score(y_test, y_prob[:, 1])

#5. Save models
# luu model classication
pkl_filename = "ham_spam_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(model, file)
  
# luu model CountVectorizer (count)
pkl_count = "count_model.pkl"  
with open(pkl_count, 'wb') as file:  
    pickle.dump(count, file)


#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    ham_spam_model = pickle.load(file)
# doc model count len
with open(pkl_count, 'rb') as file:  
    count_model = pickle.load(file)

# GUI

menu = ['Business Objective' , 'Build Project' ,'New Prediction']
choice = st.sidebar.selectbox('Menu' , menu)
if choice == 'Business Objective':
    st.subheader('Business Objective')
    st.write("""
    ###### Classifying spam and ham message is one of the most common natural language processing tasks""")
    st.write("""###### => Problem/ Requirement: Use ML algo in python for ham and spam messages """ )
    st.image('ham_spam.jpg')

elif choice == 'Build Project':
    st.subheader('Build Project')
    st.write("##### 1. Some data")
    st.dataframe(data[['v2' , 'v1']].head(3))
    st.dataframe(data[['v2' , 'v1']].tail(3))

    st.write('##### 2. Visualize Ham and Spam')
    fig1 = sns.countplot(data = data[['v1']] , x =  'v1')
    st.pyplot(fig1.figure)

    st.write('##### 3. Build model...')

    st.write('##### 4. Evaluation')
    st.code('Score train: ' + str(round(score_train, 2)) + ' vs Score Test:' + str(round(score_test , 2)))
    st.code('Accuracy:' + str(round(acc , 2)))
    st.write('##### Confusion matrix:')
    st.code(cm)
    st.write('##### Classification report:')
    st.code(cr)
    st.code('RoC AUC score:' + str(round(roc , 2)))

    # calculate roc curve
    st.write('##### ROC curve')
    fpr , tpr , thresholds = roc_curve(y_test , y_prob[:  , 1])
    fig , ax = plt.subplots()
    ax.plot([0,1] , [0,1] , linestyle = '--')
    ax.plot(fpr , tpr , marker = '.')
    st.pyplot(fig)

    st.write('##### 5. Summary: This model is good enough for Ham vs Spam classification')

elif choice == 'New Prediction':
    st.subheader('Select data')
    flag = False
    lines = None
    type  = st.radio('Upload data or Input data?' , options = ('Upload' ,'Input'))
    if type == 'Upload':
        # Upload file
        uploaded_file_1 = st.file_uploader('Choose a file' , type = ['txt' , 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1 ,header = None)
            st.dataframe(lines)
            # st.write(lines.columns)
            lines = lines[0]
            flag =True
    
    if type =='Input':
        email = st.text_area(label = 'Input your content:')
        if email != '':
            lines = np.array([email])
            flag = True
    
    if flag:
        st.write('Content:')
        if len(lines) > 0:
            st.code(lines)
            x_new = count_model.transform(lines)
            y_pred_new =ham_spam_model.predict(x_new)
            st.code('New predictions (0: Ham , 1: Spam): ' + str(y_pred_new))