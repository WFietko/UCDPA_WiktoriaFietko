# Importing packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

# Importing CSV file into pandas DataFrame
movies = pd.read_csv("blockbusters.csv")
print(movies)

# Understanding the data
print(movies.head())
print(movies.shape)
print(movies.info())

# Changing the display of values in the 'worldwide_gross' column.
# Removing of '$' and '.' signs attached to the values in the column.
movies['worldwide_gross'] = movies['worldwide_gross'].str.replace(',', '').str.replace('$', '')


# Custom function to round gross values to mln. Looping and iterrows.
def round_to_mln():
    movies.worldwide_gross = pd.to_numeric(movies.worldwide_gross)
    for i, row in movies.iterrows():
        worldwide_gross = movies.worldwide_gross[i]
        worldwide_gross = worldwide_gross / 1000000
        movies.worldwide_gross[i] = worldwide_gross


round_to_mln()

# Checked if it worked
print(movies.info())
print(movies.head())

# Cleaning the dataset
print(movies.isna().sum())
# There are missing values in genre_2 and genre_3, however this might be because some movies do not have a sub-genre.
# Replacing NaN with None
movies = movies.fillna("None")

# Checking if still there are missing values.
print(movies.isnull().sum())

# Checking for duplicates in the DataFrame.
duplicate_movies = movies.duplicated().any()
print(duplicate_movies)

# What are the top 10 movies of all time that earned the most?

# Sorting and slicing values to see top 10 movies of all time, based on the worldwide gross.
top10 = movies[['title', 'worldwide_gross']].sort_values(by='worldwide_gross', ascending=False)
top10_titles = top10.iloc[:10]
print(top10_titles)

# Creating a horizontal bar plot.
plt.figure(figsize=(10, 5))
sns.set_style('whitegrid')
sns.set_palette('PRGn')
sns.barplot(x='worldwide_gross',
            y='title',
            data=top10_titles)
plt.xlabel('Worldwide gross in mln of $', size=12)
plt.ylabel('Title', size=12)
plt.title("Top 10 movies of all time", size=18)
plt.tight_layout()
plt.savefig('fig1.png')
plt.show()

# Is there a correlation between the total revenue a film has achieved and its iMDB ranking?
correlation = movies['worldwide_gross'].corr(movies['imdb_rating'])
print(correlation)

# Creating a scatter plot to compare two variables.
plt.figure(figsize=(15, 10))
sns.set_style('whitegrid')
fig = sns.regplot(x='imdb_rating', y='worldwide_gross', data=movies, scatter_kws={'color': '#550527', 'alpha': 0.4},
                  line_kws={'color': '#688E26', 'alpha': 0.4})
plt.xlabel('iMDB rating', fontsize=12)
plt.ylabel('Worldwide gross in mln of $', fontsize=12)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('Correlation between Worldwide gross and iMDB rating', fontsize=18)
plt.savefig('fig2.png')
plt.show()

# Does maturity rating have an influence on worldwide gross?

# Creating a boxplot to show if maturity ratings have an influence on worldwide gross.
plt.figure(figsize=(10, 10))
sns.set_theme(style='whitegrid')
custom_palette_1 = ['#802392', '#995FA3', '#9A98B5', '#A0B9C6']
sns.boxplot(x='rating', y='worldwide_gross', data=movies, linewidth=2.5, palette=custom_palette_1)
plt.xlabel('Movie rating', size=12)
plt.ylabel('Worldwide gross in mln of $', size=12)
plt.title('Does maturity rating have an influence on worldwide gross?', size=18)
plt.savefig('fig3.png')
plt.show()


# Which studio has the most hits in its portfolio?

# Counting how many studios are on the list.
nr_studios = len(movies.studio.unique())
print(nr_studios)

# Counting how much each studio earned in average.
avr_gross_studio = movies.groupby(['studio']).mean()['worldwide_gross'].sort_values(ascending=False)
print(avr_gross_studio)
# Walt Disney Pictures is the studio with the highest gross.

# Counting how many movies were produced by each studio.
total_movies_studio = movies.groupby(['studio']).count()['title'].sort_values(ascending=False)
print(total_movies_studio)
# Warner Bros produced the most hits.

# Pie chart to visualize which studio produced the most hits.

# Creating a dictionary from an existing Series.
dic = total_movies_studio.to_dict()

# Grouping together all elements in the dictionary whose values are less than 9.
newdic = {}
for key, group in itertools.groupby(dic, lambda k: 'Other' if (dic[k] < 9) else k):
    newdic[key] = sum([dic[k] for k in list(group)])

# Pie chart
labels = newdic.keys()
sizes = newdic.values()
explode = (.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,)  # to expand the 1st slice - slice that is the greatest
fig, ax = plt.subplots(figsize=(8, 8.5))
custom_palette_2 = ["#001219", "#005f73", "#0a9396", "#48AD8B", "#94D2BD", "#DABF6C", "#D2B04B", "#ee9b00", "#ca6702",
                    "#bb3e03", "#ae2012", "#9b2226"]
ax.pie(sizes, explode=explode, colors=custom_palette_2, autopct='%1.1f%%', startangle=0,
       wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, textprops={'color': "w"})
ax.legend(labels, loc='upper left')
ax.axis('equal')
plt.tight_layout()
plt.savefig('fig4.png')
plt.show()


# Which genre is the most profitable?

# Calculating how many movies are in each main genre.
sum_title_genre = movies.groupby('Main_Genre').count()['title'].sort_values(ascending=False)
print(sum_title_genre.head())

# Calculating the total profit of each main genre.
sum_gross_genre = movies.groupby('Main_Genre').sum()['worldwide_gross'].reset_index().sort_values(by='worldwide_gross', ascending=False)
print(sum_gross_genre.head())

# Numpy array - calculating the average profit per main genre.
np_sum_title_genre = np.array(sum_title_genre)
np_sum_gross_genre = np.array(sum_gross_genre['worldwide_gross'])
np_index = np.array(sum_title_genre.index)
np_avr_gross_genre = (np_sum_gross_genre / np_sum_title_genre)
print(np_avr_gross_genre)

# Forming a new array by stacking the given arrays and transpose it,
# in order to get the right shape of the data for further analysis.
np_avr_genre = np.vstack((np_index, np_avr_gross_genre)).transpose()

# Converting the array to pandas DataFrame and sorting values.The end result should be two columns with sorted values.
avr_genre = pd.DataFrame({'Main Genre': np_avr_genre[:, 0], 'Average Gross': np_avr_genre[:, 1]})
avr_genre = avr_genre.sort_values(['Average Gross'], ascending=False).reset_index(drop=True)
print(avr_genre)

# Creating a barplot to visualize the results.
plt.figure(figsize=(10, 7))
sns.set_style('whitegrid')
sns.set_palette('PRGn')
sns.barplot(y='Main Genre',
            x='Average Gross',
            data=avr_genre,
            estimator=sum,
            ci=None)
plt.xlabel('Average gross in mln of $', size=12)
plt.ylabel('Genre', size=12)
plt.title('The most profitable genre', size=18)
plt.tight_layout()
plt.savefig('fig5.png')
plt.show()

# Are movies getting longer?

# Histogram about distribution of movie running times.
median_length = movies['length'].median()
print(median_length)
color_hist = '#A23B72'
color_line = '#2E86AB'

plt.hist(movies['length'], bins=20, color=color_hist, edgecolor='black')
plt.axvline(median_length, color=color_line, label='Median Length')
plt.legend()
plt.title('Distribution of movie running times')
plt.xlabel('Length of movie in min')
plt.ylabel('Nr of movies')
plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.7)
plt.tight_layout()
plt.savefig('fig6.png')
plt.show()


# Linear chart of how the length of movies has changed over the years.
plt.figure(figsize=(10, 10))
sns.set_style('whitegrid')
fig2 = sns.lineplot(x='year', y='length', data=movies)
fig2.set(xlabel='Year', ylabel='Length')
fig2.set_title('How the length of movies has changed over the years', size=18)
plt.savefig('fig7.png')
plt.show()


# Merge DataFrames - this code is not related to the main dataset,
# however it was performed to complete one of the tasks in the assessment.
# In the example below, I have demonstrated how to merge two dataframes using the merge function.

# Creating two random DataFrames
df1 = pd.DataFrame({
    "Country": ['Poland', 'Ireland', 'Spain'],
    "Capital": ['Warsaw', 'Dublin', 'Madrid']
})

df2 = pd.DataFrame({
    "Country": ['France', 'Spain', 'Poland'],
    "Currency": ['Euro', 'Euro', 'Zloty']
})

# Merging DataFrames on column 'Country'.
df3 = pd.merge(df1, df2, on='Country')
print(df3)