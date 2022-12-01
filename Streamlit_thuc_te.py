# import thu vien

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import thu vien
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
import streamlit as st
import plotly.express as px


# 1. Read data
df = pd.read_csv('e-commerce.csv')

# GUI
st.title('Data Science Project')
st.write('## Customer_Segment')

# Upload file
uploaded_file = st.file_uploader('Choose a file', type = ['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file )
    df.to_csv('e-commerce_new.csv', index = False)

# 2.Data Pre-processing

string_to_date = lambda x: datetime.strptime(x , '%d/%m/%Y')

# convert invoiceDate from object to datetime format
df['day'] = df['day'].apply(string_to_date)
df['day'] = df['day'].astype('datetime64[ns]')

# drop NA values

df = df.dropna()

# Lets take a look at the data we will need to manipulate
print('Transactions timeframe from {} to {}'.format(df['day'].min(), df['day'].max()))
print('{:,} transactions don\'t have a customer id'.format(df[df.customer_id.isnull()].shape[0]))
print('{:,} uniques customer_id'.format(len(df.customer_id.unique())))

# Create RFM analysis for each customer

# RFM _ BUOC QUAN TRONG DE TAO RA 3 CHI SO R , F ,M
# Convert string to date, get max date of dataframe
max_date  = df['day'].max().date()

Recency = lambda x: (max_date - x.max().date()).days
Frequency = lambda x: len(x.unique())
Monetary = lambda x: round(sum(x) , 2)

df_RFM = df.groupby('customer_id').agg({'day' : Recency,
              'order_id': Frequency,
              'gross_sales' :Monetary }).reset_index()

# Rename the columns of DF
df_RFM.columns = ['customer_id' ,'Recency', 'Frequency', 'Monetary']
# Descending Sorting _ sap xep giam dan theo spending_money
df_RFM = df_RFM.sort_values('Monetary' , ascending= False)

# Visualization

plt.figure(figsize= (8, 10))
plt.subplot(3,1,1)
sns.distplot(df_RFM['Recency'])
plt.subplot(3,1,2)
sns.distplot(df_RFM['Frequency']) # Plot distribution of F
plt.subplot(3,1,3)
sns.distplot(df_RFM['Monetary']) # Plot distribution of M
# plt.show()

#Create labels for Recency , Frequency ,Monetary
r_labels = range(4, 0 , -1) # so ngay tinh tu lan cuoi mua hang lon thi
                          # gan nho, nguoc lai thi dan lon
f_labels = range(1,5)
m_labels = range(1,5)

# Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(df_RFM['Recency'].rank(method = 'first') , q = 4 ,labels = r_labels)
f_groups = pd.qcut(df_RFM['Frequency'].rank(method = 'first') , q = 4 ,labels = f_labels)
m_groups = pd.qcut(df_RFM['Monetary'].rank(method = 'first') , q = 4 ,labels = m_labels)

# Create new columns R , F ,M
df_RFM = df_RFM.assign(R = r_groups.values, F = f_groups.values ,M =m_groups.values)
df_RFM.head()

## Concat RFM quartile values to create RFM Segments

def join_rfm(x):
    return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))

df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm , axis = 1)

## Count num of unique segments
rfm_count_unique = df_RFM.groupby('RFM_Segment')['RFM_Segment'].nunique()
print(rfm_count_unique[-5:])

## Calculate RFM score and level
# Calculate RFM_Score
df_RFM['RFM_Score'] = df_RFM[['R', 'F', 'M']].sum(axis =1 )
df_RFM.head()

# 3.## Manual Segmentation

def rfm_level(df):
    if (df['R'] == 4 and df['F'] == 4 and df['M'] == 4):
        return 'STARS'
    elif (df['R'] == 4 and df['F'] == 1 and df['M'] == 1):
        return 'NEW'
    else:
        if df['M'] == 4:
            return 'BIG SPENDER'
        elif df['F'] == 4:
            return 'LOYAL'
        elif df['R'] ==4:
            return 'ACTIVE'
        elif df['R'] == 1:
            return 'LOST'
        elif df['M'] == 1:
            return 'LIGHT'
        return 'REGULAR'

#Create a new column RFM_Level
df_RFM['RFM_Level'] = df_RFM.apply(rfm_level, axis =1)

# NUMber of segments

df_RFM['RFM_Level'].value_counts()

## Calculate mean values for each segment

# Calculate average values for each RFM_Level and return a size of each segment
rfm_agg = df_RFM.groupby('RFM_Level').agg({
     'Recency' : 'mean',
     'Frequency' : 'mean',
     'Monetary' : ['mean', 'count']}).round(0)

rfm_agg.columns = rfm_agg.columns.droplevel()
rfm_agg.columns = ['RecencyMean' , 'FrequencyMean', 'MonetaryMean' ,'Count']
rfm_agg['Percent'] = round((rfm_agg['Count']  / rfm_agg.Count.sum())* 100, 2)

# Reset the index
rfm_agg = rfm_agg.reset_index()

#4. VIsualization 

## * TRee Map
# Create our plot an resize it\
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(14, 10)
colors_dict = {'ACTIVE' :'yellow' , 'BIG SPENDER': 'royalblue',
      'LIGHT': 'cyan', 'LOST':'red', 'LOYAL': 'purple','POTENTIAL':'green',
      'STARS' : 'gold'}

squarify.plot(sizes = rfm_agg['Count'],
           text_kwargs= {'fontsize': 12, 'weight':'bold','fontname':'sans serif'},
            color = colors_dict.values(),
            label = ['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} \
-                customers ({}%)'.format(*rfm_agg.iloc[i]) for i in range(0, len(rfm_agg))], alpha = 0.5)

plt.title('Customers Segments', fontsize = 26 , fontweight  = 'bold')
plt.axis('off')

plt.savefig('RFM Segments.png')
# plt.show()

## * Scatter Plot(RFM)


fig = px.scatter(rfm_agg, x= 'RecencyMean', y = 'MonetaryMean', size = 'FrequencyMean',
      color = 'RFM_Level', hover_name ='RFM_Level', size_max = 100)
# fig.show()

## * 3D Scatter Plot(RFM)

fig = px.scatter_3d(df_RFM , x = 'Recency', y = 'Frequency', z = 'Monetary',
         color = 'RFM_Level', opacity = 0.5,
         color_discrete_map= colors_dict)

fig.update_traces(marker = dict(size = 5),
      selector= dict(mode  = 'markers'))

# fig.show()

#################### GUI ###################

menu = ['Business Objective', 'Build Project', 'Searching']

choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':
    st.subheader('Business Objective')
    st.write('''
    ###### Market segmentation is the process of grouping consumers based on meaningful similarities ''')
    st.image('R-project-customer-segmentation.png')

if choice == 'Build Project':
    st.subheader('Build Project')
    st.write('##### 1. Some data')
    st.dataframe(df.head())

    st.write('###### 2. Calculate/ Build model...')

    st.write('###### 3. Plot')

    ## 1. TREE MAP

    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(14, 10)
    colors_dict = {'ACTIVE' :'yellow' , 'BIG SPENDER': 'royalblue',
        'LIGHT': 'cyan', 'LOST':'red', 'LOYAL': 'purple','POTENTIAL':'green',
        'STARS' : 'gold'}

    squarify.plot(sizes = rfm_agg['Count'],
            text_kwargs= {'fontsize': 12, 'weight':'bold','fontname':'sans serif'},
                color = colors_dict.values(),
                label = ['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} \
    -                customers ({}%)'.format(*rfm_agg.iloc[i]) for i in range(0, len(rfm_agg))], alpha = 0.5)

    plt.title('Customers Segments', fontsize = 26 , fontweight  = 'bold')
    plt.axis('off')

    plt.savefig('RFM Segments.png')
    st.pyplot(fig)

    ## 2. 3D Scatter Plot(RFM)

    fig = px.scatter_3d(df_RFM , x = 'Recency', y = 'Frequency', z = 'Monetary',
         color = 'RFM_Level', opacity = 0.5,
         color_discrete_map= colors_dict)

    fig.update_traces(marker = dict(size = 5),
      selector= dict(mode  = 'markers'))

    st.plotly_chart(fig)

    ## 3.Scatter Plot(RFM
 
    fig = px.scatter(rfm_agg, x= 'RecencyMean', y = 'MonetaryMean', size = 'FrequencyMean',
    color = 'RFM_Level', hover_name ='RFM_Level', size_max = 100)
    # fig.savefig('pic_1.png')
    st.plotly_chart(fig)

if choice == 'Searching':
    st.subheader('searching')
    type = st.number_input(label = 'Input customer id:', step = 1 )
    if type != '':
        result = df_RFM.loc[df_RFM['customer_id'] == type]
        st.dataframe(result)