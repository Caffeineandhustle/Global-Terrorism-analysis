#!/usr/bin/env python
# coding: utf-8

# In[396]:


#Import all the necessary libraries

import pandas as pd #data processing
import numpy as np #linear algebra
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns #Data visualization
import math
from mpl_toolkits.basemap import Basemap
import warnings
warnings.filterwarnings('ignore')


# In[326]:


#Read Data
gtdata=pd.read_csv("/Users/priyeshkucchu/Desktop/globalterrorismdb_0718dist.csv",encoding='ISO-8859-1')


# In[327]:


#Show top data rows
gtdata.head()


# In[333]:


#Renaming the columns
gtdata.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state',                       'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',                       'nwound':'Wounded','summary':'Summary',                          'gname':'Group','targtype1_txt':'Target_type',                          'weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)


# In[334]:


#Read data with new column names
gtdata=gtdata[['Year','Month','Day','Country','state','Region','city',               'latitude','longitude','AttackType','Target','Killed','Wounded',
               'Summary','Group','Target_type','Weapon_type','Motive']]
gtdata.head()


# In[335]:


#Show information about the data 
gtdata.info()


# In[336]:


#Show the columns in the data

gtdata.columns


# In[337]:


#See the no of rows and columns in the data 
gtdata.shape


# In[338]:


#See the null values

gtdata.isnull().sum()


# In[340]:


#See the unique values in Country column
gtdata.Country.unique()


# In[341]:


#Show the data types of all the columns
gtdata.dtypes


# In[342]:


#Show unique value in the success column
gtdata1.success.unique()


# In[31]:


#Show unique value in the success column
gtdata1.motive.unique()


# # Data Visualization

# In[59]:


#Terrorist attacks by Year
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.countplot(x="Year",data=gtdata)
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=45)
plt.title("Terrorist attacks by Year")
plt.tight_layout()


# In[64]:


#Terrorist attacks by Month
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
sns.countplot(x="Month",data=gtdata)
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=45)
plt.title("Terrorist attacks by Month")
plt.tight_layout()


# In[106]:


#Top 20 countries with highest crime rate
top_countries=gtdata1.country_txt.value_counts(dropna=True)
t=top_countries.head(20)


# In[107]:


sns.set(font_scale=1.5)
plt.figure(figsize=(16,10))
sns.barplot(x=t.index,y=t.values,data=gtdata1)
plt.xlabel('Countries')
plt.ylabel('')
plt.xticks(rotation=45)
plt.title("Top 20 countries with highest crime rate",fontsize=20)
plt.tight_layout()


# In[114]:


#Top 20 cities with highest crime rate
top_cities=gtdata1.city.value_counts(dropna=True)
t=top_cities.head(20)
t


# In[118]:


#Visualize
sns.set(font_scale=1.5)
plt.figure(figsize=(16,10))
sns.barplot(x=t.index,y=t.values,data=gtdata1)
plt.xlabel('City')
plt.ylabel('')
plt.xticks(rotation=45)
plt.title("Top 20 cities with highest crime rate",fontsize=20)
plt.tight_layout()


# In[108]:


#Successful attacks by year
sns.set(font_scale=1.5)
plt.figure(figsize=(18,10))
sns.countplot(x="Year",hue="Success",data=gtdata1)
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title("Successfull attacks by year")
plt.tight_layout()


# In[109]:


#Show unique values of column suicide
gtdata1.suicide.unique()


# In[119]:


#Successfull suicides by year
sns.set(font_scale=1.5)
plt.figure(figsize=(18,10))
sns.countplot(x='iyear',hue="suicide",data=gtdata1)
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title("Successfull suicides by year")
plt.tight_layout()


# In[125]:


#Successfull ransoms by year
sns.set(font_scale=1.5)
plt.figure(figsize=(18,10))
sns.countplot(x='iyear',hue="ransom",data=gtdata)
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title("Successfull ransoms by year")
plt.tight_layout()


# In[153]:


#Show unique values of attacktype column
gtdata.attacktype1_txt.unique()


# In[136]:


#Top attacks 
cnt=gtdata.attacktype1_txt.value_counts()
c=cnt.head(10)
c


# In[147]:



sns.set(font_scale=1.5)
plt.figure(figsize=(12,10))
sns.barplot(x=c.values,y=c.index,data=gtdata)
plt.xlabel('')
plt.ylabel('Attack Type 1')
plt.xticks(rotation=45)
plt.title("Top attacks",fontsize=20)
plt.tight_layout()


# In[155]:


cnt=gtdata.attacktype2_txt.value_counts()
c=cnt.head(10)
sns.set(font_scale=1.5)
plt.figure(figsize=(12,10))
sns.barplot(x=c.values,y=c.index,data=gtdata)
plt.xlabel('')
plt.ylabel('Attack Type 2')
plt.xticks(rotation=45)
plt.title("Top attacks",fontsize=20)
plt.tight_layout()


# In[160]:


#Data reporting sources unique values : From where the terror attacks have been reported
gtdata.dbsource.unique()


# In[170]:


#Top Data reporting sources
k=gtdata.dbsource.value_counts()
db=k.head(5)


# In[179]:



sns.set(font_scale=1.5)
plt.figure(figsize=(10,7))
sns.barplot(x=db.values,y=db.index,data=gtdata)
plt.xlabel("Count")
plt.ylabel("Data base sources")
plt.xticks(rotation=45)
plt.title("Crime database sources ")
plt.tight_layout()


# In[188]:


#Terrorist attacks on the map

def mapWorld(col,size,label,metr=100,colmap='hot',ds=gtdata,scat=False):

    datat=ds
    m=Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,             llcrnrlon=-150,urcrnrlon=180,resolution='c',height=100,width=200)
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-120,91.,30.))
    m.drawmeridians(np.arange(-120,90.,60.))
    lat=datat['latitude'].values
    lon=datat['longitude'].values
    a_1=datat[col].values
    if size:
        a_2 = datat[size2].values
    else: a_2 = 1
    if scat:
        m.scatter(lon, lat, latlon=True,c=a_1,s=metr*a_2,edgecolors='black',cmap=colmap,alpha=1)
    else:
        m.scatter(lon, lat, latlon=True,c=a_1,s=metr*a_2,cmap=colmap,alpha=1)
sns.set(font_scale=1.5)
plt.figure(figsize=(15,15))
plt.title('Terrorist attacks', fontsize=20)
mapWorld(col='targtype1', size=False,label='',metr=10,colmap='viridis',ds=gtdata)


# In[191]:


#How terrorism has changed over the years
fig= plt.figure(figsize=(20,15))
ax1= fig.add_subplot(3,2,1)
ax1.set_title("1970-1979")
mapWorld(col='targtype1',size=False,ds=gtdata[(gtdata['iyear']>=1970) & (gtdata['iyear']<1980)],        label='',metr=10,colmap='viridis',scat=True)

ax2= fig.add_subplot(3,2,2)
ax2.set_title("1980-1989")
mapWorld(col='targtype1',size=False,ds=gtdata[(gtdata['iyear']>=1980) & (gtdata['iyear']<1990)],        label='',metr=10,colmap='viridis',scat=True)
ax3= fig.add_subplot(3,2,3)
ax3.set_title("1990-1999")
mapWorld(col='targtype1',size=False,ds=gtdata[(gtdata['iyear']>=1990) & (gtdata['iyear']<2000)],        label='',metr=10,colmap='viridis',scat=True)
ax4= fig.add_subplot(3,2,4)
ax4.set_title("2000-2009")
mapWorld(col='targtype1',size=False,ds=gtdata[(gtdata['iyear']>=2000) & (gtdata['iyear']<2010)],        label='',metr=10,colmap='viridis',scat=True)
ax5= fig.add_subplot(3,2,1)
ax5.set_title("2010-2017")
mapWorld(col='targtype1',size=False,ds=gtdata[(gtdata['iyear']>=2010)],        label='',metr=10,colmap='viridis',scat=True)


# In[194]:


#"Sucessful/Unsucessful terror crime"
sns.set(font_scale=1.5)
plt.figure(figsize=(20,20))
plt.title("Sucessful/Unsucessful terror crime",fontsize=20)
mapWorld(col='success',size=False,label='',metr=10,colmap='viridis',ds=gtdata)


# In[196]:


#How terrorism has changed by region/years

def plot_by_years(kind='region_txt',big=(30,20)):
    sns.set(style="white",font_scale=2.5)
    fig = plt.figure(figsize=big)
    ax1 = fig.add_subplot(3,2,1)
    ax1.set_title('2010-2017')
    ax1.set_ylabel('');
    gtdata[gtdata['iyear']>=2010]['eventid'].groupby(gtdata[kind]).count().plot(kind='barh');
    ax1.set_ylabel('');
    ax2 = fig.add_subplot(3,2,2)
    ax2.set_title('2000-2009')
    gtdata[(gtdata['iyear']>=2000) & (gtdata['iyear']<2010)]['eventid'].groupby(gtdata[kind]).count().plot(kind='barh');
    ax2.set_ylabel('');
    ax3 = fig.add_subplot(3,2,3)
    ax3.set_title('1990-1999')
    gtdata[(gtdata['iyear']>=1990) & (gtdata['iyear']<2000)]['eventid'].groupby(gtdata[kind]).count().plot(kind='barh');
    ax3.set_ylabel('');
    ax4 = fig.add_subplot(3,2,4)
    ax4.set_title('1980-1989')
    gtdata[(gtdata['iyear']>=1980) & (gtdata['iyear']<1990)]['eventid'].groupby(gtdata[kind]).count().plot(kind='barh');
    ax4.set_ylabel('');
    ax4 = fig.add_subplot(3,2,5)
    ax4.set_title('1970-1979')
    gtdata[(gtdata['iyear']>=1970) & (gtdata['iyear']<1980)]['eventid'].groupby(gtdata[kind]).count().plot(kind='barh');
    plt.tight_layout()
    plt.ylabel('');
plot_by_years(kind='region_txt')


# In[197]:


#How terrorism has changed by attack type/years
plot_by_years(kind="attacktype1_txt");


# In[199]:


#How terrorism has changed by target type/years
plot_by_years(kind="targtype1_txt",big=(30,30));


# In[281]:


#Group with the most attacks
u=gtdata.gname.value_counts()
t1=u.head()
print( "Group with the most attacks:", u.index[1],"and the count is :", u.values[1])


# In[282]:


sns.set(font_scale=1.5)
plt.figure(figsize=(10,8))
sns.barplot(x=t1.values,y=t1.index,data=gtdata)
plt.xlabel("Count")
plt.ylabel("Most attacks")
plt.title("Group with the most attacks:")
plt.xticks(rotation=45)
plt.tight_layout()


# In[252]:


# Most no of attacks
print("Most Attack Types:", gtdata['attacktype1_txt'].value_counts().idxmax())


# # Data visualization using Plotly

# In[276]:


#Top 40 Worst Terror Attacks in History from 1982 to 2016
import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
colorscale = [[0, '#edf8fb'], [.3, '#00BFFF'],  [.6, '#8856a7'],  [1, '#810f7c']]
heatmap=go.Heatmap(z=heat.as_matrix(), x=heat.columns, y= heat.index, colorscale=colorscale)
data=[heatmap]
layout=go.Layout(title="Top 40 Worst Terror Attacks in History from 1982 to 2016",xaxis=dict(ticks='',nticks=20),                yaxis=dict(ticks=''))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig, filename='heatmap',show_link=False)


# In[278]:


#Top countries affected by Attacks

gtdata.country_txt.value_counts()[:15]


# In[297]:


sns.set(font_scale=1.5)
plt.figure(figsize=(18,10))
sns.barplot(gtdata['Country'].value_counts()[:15].index,gtdata['Country'].value_counts()[:15].values,           palette='Blues_d')
plt.xlabel("Countries")
plt.ylabel("Count")
plt.title("Top countries affected by Attacks")
plt.xticks(rotation=45)
plt.tight_layout()


# In[343]:


# History of worst terror attacks


gtdata['Wounded']=gtdata['Wounded'].fillna(0).astype(int)
gtdata['Killed']=gtdata['Killed'].fillna(0).astype(int)
gtdata['Casualities']=gtdata['Wounded']+ gtdata['Killed']
gtdata1=gtdata.sort_values(by='Casualities',ascending=False)[:40]
heat= gtdata1.pivot_table(index='Country',columns='Year',values='Casualities')
heat.fillna(0,inplace=True)



# In[365]:


#Countries with most no of terror attacks ( not inlcuding Unknown)
gtdata_bubble=gtdata[(gtdata['Group']!='Unknown')& (gtdata['Casualities']>50)]
gtdata_bubble.sort_values(by='Casualities',ascending=False).head()


# In[345]:


gtdata_bubble=gtdata_bubble.sort_values(['Region','Country'])


# In[349]:


gtdata_bubble=gtdata_bubble.drop(['latitude','longitude','Target','Summary','Motive'],axis=1)


# In[350]:


gtdata_bubble=gtdata_bubble.dropna(subset=['city'])


# In[352]:


gtdata_bubble.isnull().sum()


# In[355]:


#Top Five country those have suffered most attacks
gtdata_bubble.Country.value_counts().head()


# # Bubble Plot
# 
# 

# In[398]:



hover_text=[]
for i,row in gtdata_bubble.iterrows():
    hover_text.append(('City: {city}<br>'+
                      'Group: {group}<br>'+
                      'Casualities: {casualities}<br>'+
                      'Year: {year}').format(city=row['city'],group=row['Group'],casualities=row['Casualities'],
                                            year=row['Year']))
gtdata_bubble['text']=hover_text


# In[375]:


trace0=go.Scatter(

    x=gtdata_bubble['Year'] [gtdata_bubble['Country']=='Iraq'],
    y=gtdata_bubble['Casualities'] [gtdata_bubble['Country']=='Iraq'],
    mode='markers',
    name='Iraq',
    text=gtdata_bubble['text'][gtdata_bubble['Country']=='Iraq'],
    marker=dict(
    
        symbol='circle',
        sizemode='area',
        size= gtdata_bubble['Casualities'][gtdata_bubble['Country']=='Iraq'],
        line=dict(width=2),
    )
    
)
trace1=go.Scatter(

    x=gtdata_bubble['Year'] [gtdata_bubble['Country']=='Afghanistan'],
    y=gtdata_bubble['Casualities'] [gtdata_bubble['Country']=='Afghanistan'],
    mode='markers',
    name='Afghanistan ',
    text=gtdata_bubble['text'][gtdata_bubble['Country']=='Afghanistan'],
    marker=dict(
    
        symbol='circle',
        sizemode='area',
        size= gtdata_bubble['Casualities'][gtdata_bubble['Country']=='Afghanistan'],
        line=dict(width=2),
    )
    
)
trace2=go.Scatter(

    x=gtdata_bubble['Year'] [gtdata_bubble['Country']=='Pakistan'],
    y=gtdata_bubble['Casualities'] [gtdata_bubble['Country']=='Pakistan'],
    mode='markers',
    name='Pakistan',
    text=gtdata_bubble['text'][gtdata_bubble['Country']=='Pakistan'],
    marker=dict(
    
        symbol='circle',
        sizemode='area',
        size= gtdata_bubble['Casualities'][gtdata_bubble['Country']=='Pakistan'],
        line=dict(width=2),
    )
    
)
trace3=go.Scatter(

    x=gtdata_bubble['Year'] [gtdata_bubble['Country']=='Nigeria'],
    y=gtdata_bubble['Casualities'] [gtdata_bubble['Country']=='Nigeria'],
    mode='markers',
    name='Nigeria',
    text=gtdata_bubble['text'][gtdata_bubble['Country']=='Nigeria'],
    marker=dict(
    
        symbol='circle',
        sizemode='area',
        size= gtdata_bubble['Casualities'][gtdata_bubble['Country']=='Nigeria'],
        line=dict(width=2),
    )
    
)


# In[376]:


data=[trace0,trace1,trace2,trace3]
layout=go.Layout(
         title = 'Top 4 countries',
         xaxis = dict(
             title = 'Year',
             
             range = [1976,2016],
             tickmode = 'auto',
             nticks = 30,
             showline = True,
             showgrid = False
             ),
         yaxis = dict(
             title = 'Casualities',
             type = 'log',
             range = [1.8,3.6],
             tickmode = 'auto',
             nticks = 40,
             showline = True,
             showgrid = False),
         paper_bgcolor='rgb(243, 243, 243)',
         plot_bgcolor='rgb(243, 243, 243)',
         )


# In[378]:


fig=go.Figure(data=data, layout=layout)
py.iplot(fig,filename='Terrorism')


# In[379]:


# Which groups have attacked the most
gtdata.Group.value_counts()[1:15]


# In[380]:


gtdata_terror=gtdata[gtdata.Group.isin(['Taliban','Islamic State of Iraq and the Levant (ISIL)','Shining Path'])]


# In[381]:


# In which country 'Taliban','Islamic State of Iraq and the Levant (ISIL)','Shining Path' has attacked the most
gtdata_terror.Country.unique()


# In[408]:


# Map countries with terror attacks using folium

gtdata_Group = gtdata.dropna(subset=['latitude','longitude'])


# In[400]:


gtdata_Group = gtdata_Group.drop_duplicates(subset=['Country','Group'])


# In[401]:


terrorist_groups = gtdata.Group.value_counts()[1:8].index.tolist()


# In[402]:



gtdata_Group = gtdata_Group.loc[gtdata_Group.Group.isin(terrorist_groups)]


# In[403]:


gtdata_Group.Group.unique()


# In[405]:


m1 = folium.Map(location=[20, 0], tiles="Stamenterrain", zoom_start=2)
marker_cluster = MarkerCluster(
    name='clustered icons',
    overlay=True,
    control=False,
    icon_create_function=None
)
for i in range(0,len(gtdata_Group)):
    marker=folium.Marker([gtdata_Group.iloc[i]['latitude'],gtdata_Group.iloc[i]['longitude']]) 
    popup='Group:{}<br>Country:{}'.format(gtdata_Group.iloc[i]['Group'],
                                          gtdata_Group.iloc[i]['Country'])
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)
marker_cluster.add_to(m1)
folium.TileLayer('openstreetmap').add_to(m1)
folium.TileLayer('Mapbox Bright').add_to(m1)
folium.TileLayer('cartodbdark_matter').add_to(m1)
folium.TileLayer('stamentoner').add_to(m1)
folium.LayerControl().add_to(m1)
m1.save('Terrorist_Organizations_in_Country_cluster.html')


# In[407]:


m1

