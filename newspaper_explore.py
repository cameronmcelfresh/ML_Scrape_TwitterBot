
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import Counter
import copy
import operator


"""The Comment class needs to be present in order for Spyder to recognize the Comment
class being imported in the newspaper data file"""
class Comment:
    def __init__(self, neighborhood, comment_likes, comment_content):

        self.neighborhood = neighborhood
        self.comment_likes = comment_likes
        self.comment_content = comment_content
    

"""Function to sort dictionaries by the yearly keys"""
def sort_year_dict(unsorted_dict):
    
    sorted_dict = {}
    
    for key in sorted(unsorted_dict.keys()):
        sorted_dict[key] = unsorted_dict[key]
    
    return sorted_dict
    

"""Make a copy of the original dataframe for safe keeping"""
df_orig = df.copy()

"""Initial data exploration"""

#First lets remove any mislabeled or Na values
print("Counting Na-labeled values in the dataframe")
print(df.isna().sum())

#Check to see if there are any topics that are largely underrepresented
print("Counting the number of each topic article in the dataframe")
print(df['topic'].value_counts())

#Let's remove the lowest 4 categories because of how few articles there are
Topics2Remove = ['HISTORIC PHOTOS', 'MOVIES', 'BOOKS', 'RESTAURANTS']

for aTopic in Topics2Remove:
    dropIndex = df[df['topic'] == aTopic].index
    df.drop(index = dropIndex, axis = 0, inplace=True) #Drop all rows with that given topic
    
print("Topic counts after removing topics; historic photos, movies, books, and restraunts")
print(df['topic'].value_counts())


"""Total yearly articles"""
total_year_articles_unsorted = Counter(df['post_year'])

total_year_articles_sorted = {}

#Sort the order of the total yearly posts
for key in sorted(total_year_articles_unsorted.keys()):
    total_year_articles_sorted[key] = total_year_articles_unsorted[key]

fig1, ax1 = plt.subplots(figsize=(5.3, 4))
ax1.bar(range(len(total_year_articles_sorted)), total_year_articles_sorted.values(), align='center')
plt.xticks(range(len(total_year_articles_sorted)), list(total_year_articles_sorted.keys()), rotation=90)
plt.title('Total Articles Per Year')
plt.ylabel("Number of Articles")
plt.show()


"""Total topic articles"""
total_topic_articles = Counter(df['topic'])

fig2, ax2 = plt.subplots(figsize=(5.3,6))
ax2.bar(range(len(total_topic_articles)), total_topic_articles.values(), align='center')
plt.xticks(range(len(total_topic_articles)), list(total_topic_articles.keys()), rotation = 90)
fig2.subplots_adjust(bottom=0.35)
plt.title('Total Articles Per Topic')
plt.ylabel("Number of Articles")
plt.show()

"""Number of articles per topic broken down by year"""
yearly_topics = []

all_topics = list(total_topic_articles.keys())

#Lets sort the topics by most to least total views
temp_views = sorted(total_topic_articles.values())
temp_views.reverse() #Reverse the list to be most to least
i = 0
for view_num in temp_views: #Sort the list by most views to least
    for k in total_topic_articles.keys():
        if total_topic_articles[k]==view_num:
            all_topics[i] =k
            i=i+1

for topic_type in all_topics:
    
    year_dict = Counter(df[df['topic']==topic_type]['post_year']).copy()
    
    for year in total_year_articles_sorted.keys():
        if year not in year_dict.keys():
            year_dict[year]=0
    
    yearly_topics.append(sort_year_dict(year_dict))
  
#Find appropriate maximum value to scale all plots
max_val=0
for year in yearly_topics:
    if max_val<max(year.values()):
        max_val=max(year.values())
        

#Plot figure with subplots
fig3 = plt.figure(figsize=(16,8))
num = 1
for year_dict in yearly_topics:
    plt.subplot(2,3,num)
    plt.bar(range(len(year_dict)), year_dict.values(), align='center')
    plt.xticks(range(len(yearly_topics[0])), list(yearly_topics[0].keys()), rotation=45) #Make sure it is consistent across all years
    plt.ylim(0, max_val+20)
    plt.title(all_topics[num-1])
    num+=1

fig3.suptitle('Articles Per Year', fontsize=25)
plt.subplots_adjust(hspace=0.4,wspace=0.25)
plt.show()

#Now do it in stacked barplot form
yearly_topics_reduced = pd.DataFrame.from_dict(yearly_topics)
yearly_topics_reduced.index=all_topics #Set the indicies to be the individual topics
yearly_topics_reduced = yearly_topics_reduced.transpose() #Transpossse to switch the columns and indicies

fig3_1 = yearly_topics_reduced.plot.bar(figsize=(5.3,4), stacked=True)
plt.title("Articles Per Year")
plt.ylabel("Number of Articles")
plt.show()

    
"""Total yearly views"""
all_years = list(total_year_articles_sorted.keys())

total_views = {}

for year in all_years:
    total_views[year] = sum(df[df['post_year']==year]['views'])
    
fig4, ax4 = plt.subplots(figsize=(5.3,4))
ax4.bar(range(len(total_views)), total_views.values(), align='center')
plt.xticks(range(len(total_views)), list(total_views.keys()), rotation = 45)
plt.title('Views Per Year')
plt.ylabel("Views")
plt.show()

"""Views per year segmented into individual topics"""
views_per_topic = []

for topic in all_topics:
    yearly_topic_views = {}
    
    for year in all_years:
        yearly_topic_views[year] = sum(df[(df['post_year']==year) &(df['topic']==topic)]['views'])
        
    views_per_topic.append(copy.deepcopy(yearly_topic_views))

max_topic_val = 0
#Find maximum for scale
for val in views_per_topic:
    if max_topic_val<max(val.values()):
        max_topic_val=max(val.values())
        

#Plot figure with subplots
fig5 = plt.figure(figsize=(16,8))
num = 1
for topic in views_per_topic:
    plt.subplot(2,3,num)
    plt.bar(range(len(topic)), topic.values(), align='center')
    plt.xticks(range(len(views_per_topic[0])), list(views_per_topic[0].keys()), rotation=45) #Make sure it is consistent across all years
    plt.ylim(0, max_topic_val+200)
    plt.title(all_topics[num-1])
    num+=1

fig5.suptitle('Views Per Year', fontsize=25)
plt.subplots_adjust(hspace=0.4,wspace=0.25)
plt.ylabel("Views")
plt.show()

##Now plot it with a stacked barplot!!
yearly_views_reduced = pd.DataFrame.from_dict(views_per_topic)
yearly_views_reduced.index=all_topics #Set the indicies to be the individual topics
yearly_views_reduced = yearly_views_reduced.transpose() #Transposssssse to switch the cooooolumns and indicies

fig5_1 = yearly_views_reduced.plot.bar(figsize=(5.3,4), stacked=True)
plt.title("Views Per Year")
plt.ylabel("Views")
plt.show()


"""Average views per article - arranged by topic and year"""
fig6 = plt.figure(figsize=(16,8))
num=1

for topic in all_topics:
    
    yearly_views = []
    yearly_std = []
    
    for year in all_years:
        
        years_data = df[(df['topic']==topic) & (df['post_year']==year)]['views'].copy()
        
        if math.isnan(years_data.mean()):
            yearly_views.append(0)
        else:
            yearly_views.append(years_data.mean())

        if math.isnan(years_data.std()):
            yearly_std.append(0)
        else:
            yearly_std.append(years_data.std())
        
    
    plt.subplot(2,3,num)
    plt.bar(range(len(yearly_views)), yearly_views, yerr = yearly_std, align='center')
    plt.xticks(range(len(views_per_topic[0])), all_years, rotation=45) #Make sure it is consistent across all years
    plt.ylim(0, 80000)
    plt.title(all_topics[num-1])
    num+=1    

fig6.suptitle('Average Views Per Article Per Year', fontsize=25)
plt.subplots_adjust(hspace=0.4,wspace=0.25)
plt.ylabel("Avg. Views Per Article")
plt.show()



"""Topic comments per year"""
fig8 = plt.figure(figsize=(16,8))
num=1

all_yearly_comments =[]

for topic in all_topics:
    
    yearly_comments = {}

    for year in all_years:
        
        yearly_comments[year] = df[(df['topic']==topic) & (df['post_year']==year)]['comment_nums'].sum()
        
    all_yearly_comments.append(copy.deepcopy(yearly_comments))
        
    plt.subplot(2,3,num)
    plt.bar(range(len(yearly_comments)), yearly_comments.values(), align='center')
    plt.xticks(range(len(views_per_topic[0])), all_years, rotation=45) #Make sure it is consistent across all years
    plt.ylim(0, 13000)
    plt.title(all_topics[num-1])
    num+=1    

fig8.suptitle('Comments Per Year', fontsize=25)
plt.subplots_adjust(hspace=0.4,wspace=0.25)
plt.ylabel("Comments Per Year")
plt.show()

##Now plot it with a stacked barplot!!
all_yearly_comments_reduced = pd.DataFrame.from_dict(all_yearly_comments)
all_yearly_comments_reduced.index=all_topics #Set the indicies to be the individual topics
all_yearly_comments_reduced = all_yearly_comments_reduced.transpose() #Transposssssse to switch the cooooolumns and indicies

fig8_1 = all_yearly_comments_reduced.plot.bar(figsize=(5.3,4),stacked=True, sort_columns=True)
plt.subplots_adjust(left=0.2)
plt.title("Comments Per Year")
plt.ylabel("Number of Comments")
plt.show()

"""Topic comments per year"""
fig9 = plt.figure(figsize=(16,8))
num=1

for topic in all_topics:
    
    yearly_comments = {}
    yearly_comments_std = []

    for year in all_years:
        
        yearly_comments[year] = df[(df['topic']==topic) & (df['post_year']==year)]['comment_nums'].mean()
        yearly_comments_std.append(df[(df['topic']==topic) & (df['post_year']==year)]['comment_nums'].std())

    plt.subplot(2,3,num)
    plt.bar(range(len(yearly_comments)), yearly_comments.values(), yerr = yearly_comments_std, align='center')
    plt.xticks(range(len(views_per_topic[0])), all_years, rotation=45) #Make sure it is consistent across all years
    plt.ylim(0, 80)
    plt.title(all_topics[num-1])
    num+=1    

fig9.suptitle('Average Comments Per Article Per Year', fontsize=25)
plt.subplots_adjust(hspace=0.4,wspace=0.25)
plt.show()

"""Total comments by neighborhood"""
#Extract all neighborhood titles

all_neighborhoods = []
all_neighborhood_comments = {}

for comments in df['comments']:
    for comment in comments:
        if comment.neighborhood.split("\n")[0] not in all_neighborhoods:
            all_neighborhoods.append(comment.neighborhood.split("\n")[0])
            
        if comment.neighborhood.split("\n")[0] in all_neighborhood_comments.keys():
            all_neighborhood_comments[comment.neighborhood.split("\n")[0]] = all_neighborhood_comments[comment.neighborhood.split("\n")[0]] + 1
            
        if comment.neighborhood.split("\n")[0] not in all_neighborhood_comments.keys():
            all_neighborhood_comments[comment.neighborhood.split("\n")[0]] = 1

comment_place_holder = sorted(all_neighborhood_comments.items(), key=operator.itemgetter(1))

all_neighborhood_comments_sorted = {}

#Sort the comments, but only interested in neighborhoods that have commented over 100
#times in the last 5 years

for neighborhoods in comment_place_holder:
    if len(neighborhoods[0])>3 and neighborhoods[1]>1000:
        all_neighborhood_comments_sorted[neighborhoods[0][13:]]= neighborhoods[1]

#Make the last 2 bars red
colors = ['cadetblue']*len(all_neighborhood_comments_sorted)
colors[len(all_neighborhood_comments_sorted)-1]='red'
colors[len(all_neighborhood_comments_sorted)-2]='red'


fig10, ax10 = plt.subplots(figsize=(10.6,8))
ax10.bar(range(len(all_neighborhood_comments_sorted)), all_neighborhood_comments_sorted.values(), align='center', color = colors)
plt.xticks(range(len(all_neighborhood_comments_sorted)), list(all_neighborhood_comments_sorted.keys()),rotation=90, fontSize=10)
fig10.subplots_adjust(bottom=0.3)
fig10.subplots_adjust(left=0.15)
plt.title('Total Comments By Neighborhood')
plt.ylabel("Total Comments")
plt.show()


"""Views vs. Comments"""
fig11, ax11 = plt.subplots(figsize=(5.3,4))
plt.scatter(df["views"], df["comment_nums"], alpha=0.5) #Maybe change this to by color?
plt.xscale("log")
plt.ylabel("Number of Comments")
plt.xlabel("Number of Views")
plt.title("Views vs. Comments")

