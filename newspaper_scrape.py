
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import numpy as np
import math

#Create a comment class to act as a container for all the comment info
class Comment:
    def __init__(self, neighborhood, comment_likes, comment_content):

        self.neighborhood = neighborhood
        self.comment_likes = comment_likes
        self.comment_content = comment_content

base_page_url = "https://www.paloaltoonline.com/square/"

urls = []
topics = []
views = []
titles =[]
num_list = np.linspace(0,499,500)*20

#Print progress
print("Finding articles...")

for num in num_list:
    
    page_val = math.floor(num)
    
    page_url = base_page_url + "?sort=&cat=&search=&s=" + str(page_val)

    #Unsegmented page text
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    articles = soup.find_all('div', attrs={'style':'margin: 0 0 15px 0;'})
    
    for article in articles:
        
        #Extract the article topic and URL
        page_data = article.find_all('a')

        #Extract the article topic
        topics.append(page_data[0].text)
        
        extracted_url = "https://www.paloaltoonline.com/news" + page_data[1].get('href')[7:]
        urls.append(extracted_url)
        
        
        #Extract the number of views of the article      
        post_view_data  = article.find('span',attrs={'class':'grey'}).text
        post_view_data.strip()
        post_view_list = post_view_data.split()
        view_num = post_view_list[-2]
        views.append(int(view_num))
        
        #Extract the title of the article
        text_header = article.text.split('\n')
        titles.append(text_header[3].strip())
        
    time.sleep(0.8)
        
print(str(len(urls)) + " articles found...\n")

#Create a dataframe to hold the info extracted from the page
df = pd.DataFrame(columns = ['url','title','topic','post_day','post_month','post_year','post_time',
                             'author','views','comment_nums','comments','article_text'])
#locator number
loc_num = 0
    
for url in urls:
    
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    #Check to ensure the story is in the database, otherwise continue to next url
    if soup.p.text == 'The story could not be found in the database.':
        loc_num+=1
        continue
    
    #Extract post_date and post time
    date_info_split = soup.find('span', attrs={'class':'grey'}).text[9:].strip().split(',')
    
    post_day = date_info_split[0]
    post_month = date_info_split[1].split()[0]
    post_year = date_info_split[2].strip()
    post_time = date_info_split[3]
    
    #Extract the Author Name
    author = soup.find('a', attrs={'class':'grey'}).text.strip()
    
    #Extract comments and comment conents/info
    comments = soup.find_all('div', attrs={'id':'comment'})
    
    comment_nums = int(len(comments))
    
    #Container to hold all comments for a post
    post_comments = []
    
    for comment in comments[:-1]:
        
        #Check to see if there are any user-defined comments, otherwise exit
        if comment.text == 'There are no comments yet. Please share yours below.':
            break
        
        neighborhood = comment.find_all('span')[0].text.split('\t')[7]

        if str(type(comment.find('div', attrs = {'class':'grey'})))=="<class 'NoneType'>":
            comment_likes=0
        else:
            try:
                comment_likes = int(comment.find('div', attrs = {'class':'grey'}).text.strip().split()[0])
            except:
                comment_likes=0
                        
        comment_content = comment.p.text.strip()
        
        post_comments.append(Comment(neighborhood.split("\n")[0], int(comment_likes), comment_content))
        
    #Extract article content
    article_text = soup.find('div', attrs={'class':'story'}).text[:-150]
    
    #Add all contents to the dataframe
    df.loc[loc_num] = [urls[loc_num]] + [titles[loc_num]] + [topics[loc_num]] + [post_day] + [post_month] + [post_year] + [post_time] + [author] + [views[loc_num]] + [comment_nums] + [post_comments] + [article_text]
    
    loc_num+=1
    
    #Print progress
    if len(df)%10==0:
        print("Content of " +str(len(df)) + " articles scraped...\n")
    
    #Pause as to not request at too fast of a rate
    time.sleep(0.8)
    
    
