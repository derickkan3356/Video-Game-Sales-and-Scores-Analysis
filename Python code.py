# general
import pandas as pd
import numpy as np
import re
import ast
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# library for wiki data
import requests
from bs4 import BeautifulSoup

# visualization library
import seaborn as sns 
import matplotlib.pyplot as plt
import folium
from folium.features import GeoJson
from folium import Choropleth
import geopandas as gpd

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

# Print versions
import sys
print("Your Environment:")
print("Python version:", sys.version)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("scipy version:", __import__('scipy').__version__)
print("requests version:", requests.__version__)
print("beautifulsoup4 version:", __import__('bs4').__version__)
print("seaborn version:", sns.__version__)
print("matplotlib version:", __import__('matplotlib').__version__)
print("folium version:", __import__('folium').__version__)
print("geopandas version:", gpd.__version__)

print("--------------------------\nTo ensure full re-runnability, please maintain consistent versions:")
print("pandas version: 2.1.2")
print("numpy version: 1.26.1")
print("scipy version: 1.11.3")
print("requests version: 2.31.0")
print("beautifulsoup4 version: 4.12.2")
print("seaborn version: 0.13.0")
print("matplotlib version: 3.8.1")
print("folium version: 0.15.0")
print("geopandas version: 1.0.0")

data_IGN_original = pd.read_csv("./data/IGN_data.csv")
data_sale_original = pd.read_csv("./data/Video_Games.csv")

def data_cleaning(data_IGN_input, data_sale_input):
    """We remove the columns that are not related to this project purpose, or duplicated along datasets.
    We also perform cleaning on the game name column of both datasets, which is highly relevant to the datasets merging performance.
    Some columns contains strings that represent lists (e.g."['Atlus', 'Nordcurrent']"). We convert these strings into actual lists."""

    data_IGN, data_sale = data_IGN_input.copy(), data_sale_input.copy()

    # adjust released_date to year format for matching Sale dataset format
    data_IGN['Year_of_Release'] = pd.DatetimeIndex(data_IGN['released_date']).year

    # dropping rows with missing Value of Game and Score
    data_IGN.dropna(subset=['game','score'],how = 'any', inplace = True)
    data_sale.dropna(subset=['Name', 'Global_Sales'], inplace = True)

    # removing the columns that are beyond the project scope
    IGN_Useless_columns = ["Unnamed: 0","developers", "franchises", "features",\
                                            "released_date", "score_text", "esrb_info"]
    data_IGN.drop(columns=IGN_Useless_columns, inplace = True)

    Sale_Useless_columns = ["Genre", "Publisher", \
                                "Critic_Count" , "User_Count", "Developer", "Rating"]
    data_sale.drop(columns=Sale_Useless_columns, inplace = True)

    # remove non-alphabetic characters from game name
    data_IGN['game'] = data_IGN['game'].str.replace('[\!\?\.\:\'\-\~\,\"]', '', regex=True)
    data_sale['Name'] = data_sale['Name'].str.replace('[\!\?\.\:\'\-\~\,\"]', '', regex=True)

    # replace double space to single space
    data_IGN['game'] = data_IGN['game'].str.replace('  ', ' ',)
    data_sale['Name'] = data_sale['Name'].str.replace('  ', ' ',)

    # lower case game names
    data_IGN["game"] = data_IGN["game"].str.strip().str.lower()
    data_sale["Name"] = data_sale["Name"].str.strip().str.lower()

    # convert the columns from string representation of lists to actual lists
    def safe_literal_eval(val):
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return val
        return val
        
    for col in ['publishers', 'platform', 'genres']:
        data_IGN[col] = data_IGN[col].apply(safe_literal_eval)

    # convert string column in sale data into numeric
    data_sale['User_Score'] = data_sale['User_Score'].replace('tbd', np.nan)
    data_sale['User_Score'] = pd.to_numeric(data_sale['User_Score'])

    # ensure 'Score' columns are numeric
    for df in [data_IGN_input, data_sale_input]:
        score_columns = [col for col in df.columns if 'Score' in col]
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # scale critic score to 10-point scale
    data_sale['Critic_Score'] = data_sale['Critic_Score'].apply(lambda x: x / 10)

    return data_IGN, data_sale

# function to flatten lists and get unique values
def unique_values(x):
    flattened_list = []
    for sublist in x:
        if isinstance(sublist, list):
            flattened_list.extend(sublist)
        else:
            flattened_list.append(sublist)
    return list(set(flattened_list))

# function to determine the aggregation method based on the column type
def get_agg_func(column_name, column_data):
    if column_name == 'Year_of_Release':
        return 'min'
    elif 'score' in column_name.lower():
        return 'mean'
    elif np.issubdtype(column_data.dtype, np.number):
        return 'sum'
    elif isinstance(column_data.iloc[0], list):
        return unique_values  # return all unique for list of string
    else:
        return lambda x: ', '.join(map(str, x.unique())) # return all unique for string
    
def combine_platform(data_IGN_input, data_sale_input):
    """Some games are published on multiple platform and duplicate, stating with [bracket] or (parenthesis).
    We will combine these game into single row by
        - taking the minimum on the year of release
        - take mean on score columns
        - averaging other numeric columns
        - appending the unique value on the string / list of strings columns
    This aggregation logic will be re-use through out the entire project."""

    data_IGN = data_IGN_input.copy()
    data_sale = data_sale_input.copy()

    # remove bracketed content from game names
    data_IGN["game"] = data_IGN["game"].apply(lambda name: re.sub(r"(\[.*?\]|\(.*?\))", "", name))
    data_sale["Name"] = data_sale["Name"].apply(lambda name: re.sub(r"(\[.*?\]|\(.*?\))", "", name))

    # make sure it is stripped
    data_IGN['game'] = data_IGN['game'].str.strip()
    data_sale['Name'] = data_sale['Name'].str.strip()

    # sort data by Year_of_Release before grouping to ensure 'first' picks the earliest year's score
    data_IGN.sort_values(by='Year_of_Release', inplace=True)
    data_sale.sort_values(by='Year_of_Release', inplace=True)

    # create aggregation dictionaries for each columns, except the group by column
    agg_funcs_IGN = {col: get_agg_func(col, data_IGN[col]) for col in data_IGN.columns if col != 'game'}
    agg_funcs_sale = {col: get_agg_func(col, data_sale[col]) for col in data_sale.columns if col != 'Name'}

    # aggregate data
    data_IGN_aggregated = data_IGN.groupby('game').agg(agg_funcs_IGN).reset_index()
    data_sale_aggregated = data_sale.groupby('Name').agg(agg_funcs_sale).reset_index()

    return data_IGN_aggregated, data_sale_aggregated

def mapping_game_name(IGN_name_input, Sale_name):
    """Sale dataset use '/' to separate alternative version of a game in single row,
    but IGN dataset separate alternative version into multiple rows.
    From example 'Pokemon Red/Pokemon Blue', we can see game with alternative versions is split by a Slash '/'.
    However, we cannot split the game name by a 'slash' since some origin game containing this symbol.
    So, we check if a game name (with slash symbol) exist in both dataset,
    which mean that slash symbol is not an presentation of alternative versions, and then apply mapping to game name"""

    mapping = {} # mapping_set

    # remove "Version" term
    IGN_name = IGN_name_input.apply(lambda name: re.sub(r"\ version$", "", name))

    # create a name set prepared for quick check
    ign_game_set = set(IGN_name)

    # only split if the game does not exist in the IGN set
    for game in Sale_name:
        if '/' in game and game not in ign_game_set:
            parts = game.split('/')
            for part in parts:
                mapping[part.strip()] = game

    return IGN_name.map(mapping).fillna(IGN_name) # return apply mapping

def combine_mapped_game(df_input):
    """And to prevent duplication, we will group them as one, base on mapped game name."""
    df = df_input.copy()

    # we are no longer use "Name" column.
    df.drop(columns="game", inplace = True)

    # create aggregation dictionaries for each columns
    agg_funcs = {col: get_agg_func(col, df[col]) for col in df.columns if col != 'mapped_game'}
    
    # perform the groupby and aggregation
    df = df.groupby('mapped_game').agg(agg_funcs).reset_index()

    return df

def fail_to_merge(data_IGN, data_sale, sim_threshold):
    """Function to check the game name with high cosine similarity.
    Can be used to identify which game exists in both datasets but fail to merge together."""

    # outer merge two df to get those fail marge game name
    outer_merge = pd.merge(data_IGN, data_sale, left_on='mapped_game', right_on='Name', how='outer')
    fail_merge = outer_merge[(outer_merge['mapped_game'].isnull()) | (outer_merge['Name'].isnull())]

    fail_IGN =  fail_merge['mapped_game'].dropna().to_list()
    fail_sale = fail_merge['Name'].dropna().to_list()

    # convert the game name to TF-IDF vectors
    vectorizer = TfidfVectorizer().fit(fail_IGN + fail_sale)
    tfidf_matrix1 = vectorizer.transform(fail_IGN)
    tfidf_matrix2 = vectorizer.transform(fail_sale)

    # calculate cosine similarity between the two game name
    cosine_sim = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

    # find pairs with cosine similarity greater than a threshold
    result = []
    for i in range(len(fail_IGN)):
        for j in range(len(fail_sale)):
            if cosine_sim[i, j] > sim_threshold:
                result.append((fail_IGN[i], fail_sale[j], cosine_sim[i, j]))

    result_df = pd.DataFrame(result, columns=['IGN dataset name', 'sale dataset name', 'cos similarity'])

    return result_df

def remove_unnecessary_columns(df_input):
    """
    1. We observed that there are discrepancies in the year of release between the datasets.
        We have decided to always take the earlier date because it likely represents the initial release of the game.
        This date is often the most significant for sales spikes and initial reviews.
    2. We also noticed that most columns from the IGN dataset provide more detailed information, including publisher, platform, and genres.
        Therefore, we will retain these columns from the IGN dataset and remove the corresponding columns from the sales dataset.
        Additionally, as the game names are already consistent after merging, we will simply keep the game name from the IGN dataset.
    3. We will rename some columns for better readability.
    """

    df = df_input.copy()

    # get earlier release year
    df['Year_of_Release'] = df[['Year_of_Release_x', 'Year_of_Release_y']].apply(lambda x: np.nanmin(x), axis=1)
    df.drop(columns=['Year_of_Release_x', 'Year_of_Release_y'], inplace=True)

    # remove unnecessary columns from sale dataset
    df.drop(columns=['Name', 'Platform'], inplace=True)

    # convert float to int
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)

    # rename columns
    rename_dict = {
        'mapped_game': 'Game',
        'publishers': 'Publishers',
        'platform': 'Platforms',
        'genres': 'Genres_IGN',
        'score': 'IGN_Score',
        'esrb': 'ESRB_Rating'
    }
    df.rename(columns=rename_dict, inplace=True)

    return df

def agg_by_publisher_releaseYear(df_input):
    """When aggregating game data by publisher and release year, we must account for games with multiple publishers.
    Simply exploding the list of publishers into duplicate rows can inflate sales and count values,
    as each publisher row would duplicate these values.
    To address this, we divide the sum of sales and count values by the number of publishers before exploding,
    ensuring accurate reflection of each publisher's contribution.
    For score-related columns, we simply use mean for aggregation without dividing the number of publishers before exploding."""

    df = df_input.copy()
    df = df.dropna(subset=['Publishers'])

    # drop columns that are not in use in publisher aggregation
    df = df.drop(columns=['Game', 'Platforms', 'Genres_IGN', 'ESRB_Rating'])

    # divide sale and count columns by the number of publishers
    columns_to_divide = [col for col in df.columns if ('Sales' in col) or ('Count' in col)]
    df[columns_to_divide] = df.apply(
        lambda row: row[columns_to_divide] / len(row['Publishers']), axis=1
    )

    # explode the 'Publishers' column to break lists into multiple rows, and Standardize Publisher name's style
    df = df.explode('Publishers')
    df = df.dropna(subset=['Publishers'])
    df["Publishers"] = df["Publishers"].apply(lambda publisher: publisher.strip().lower())
    # remove double space
    df["Publishers"] = df["Publishers"].replace(r'\s+', ' ', regex=True)

    # we manually checked that microsoft has different name that can be combine
    df['Publishers'] = df['Publishers'].str.replace('microsoft game studios', 'microsoft')

    # create a dictionary of aggregation functions for each column for IGN_data
    agg_funcs = {col: get_agg_func(col, df[col]) for col in df.columns if not col in ['Publishers', 'Year_of_Release']}

    df = df.groupby(['Publishers', 'Year_of_Release']).agg(agg_funcs).reset_index()

    return df

def get_country_of_publishers():
    """We extract a list of country with publisher from wiki.
    We also perform come cleaning on publisher name as this can increase the merging performance."""

    url = "https://en.wikipedia.org/wiki/List_of_video_game_publishers"

    # send a GET request to the URL
    response = requests.get(url)

    # parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')

    # find the table containing the list of publishers
    table = soup.find('table', {'class': 'wikitable sortable'})

    publishers = []
    countries = []

    for row in table.find_all('tr')[1:]:  # skip the header row
        columns = row.find_all('td')
        if len(columns) > 1:
            publisher = columns[0].text.strip()
            country = columns[1].text.strip()
            publishers.append(publisher)
            countries.append(country)

    df = pd.DataFrame({
        'Publishers': publishers,
        'City': countries
    })

    # standardize Publisher name's style
    df["Publishers"] = df["Publishers"].apply(lambda publisher: publisher.strip().lower()).astype(str)

    # grabbing the Country from string data
    df["City"] = df["City"].str.split(',').tolist()
    df["Country"] = df["City"].apply(lambda x: x[-1])
    df.drop(columns=["City"], inplace=True)

    # clean country name
    replacements = {
        'California United States': 'United States of America',
        'California': 'United States of America',
        'Washington': 'United States of America',
        'Illinois': 'United States of America',
        'Texas': 'United States of America',
        'New York City': 'United States of America',
        'Florida': 'United States of America',
        'United States': 'United States of America',
        'United Kingdoms': 'United Kingdom',
        'England': 'United Kingdom',
        'Czech Republic': 'Czechia',
        'The Netherlands': 'Netherlands',
        'Amsterdam': 'Netherlands',
        'Republic of Korea': 'South Korea',
        'Korea': 'South Korea'
    }
    df['Country'] = df['Country'].str.strip()
    df['Country'] = df['Country'].replace(replacements)

    return df

def agg_by_country(df_input):
    df = df_input.copy()

    df = df.drop(columns=['Year_of_Release', 'Publishers'])

    agg_funcs = {col: get_agg_func(col, df[col]) for col in df.columns if col != 'Country'}

    df = df.groupby('Country').agg(agg_funcs).reset_index()

    return df

def data_occurrence(df):
    """Check for the occurrence of each release year"""

    year_counts = df.copy()['Year_of_Release'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    year_counts.plot(kind='bar', color='dodgerblue')
    year_list = sorted(df.copy()['Year_of_Release'].unique())
    index_of_1996 = year_list.index(1996) - 0.5  # adjusting to place the line between 1995 and 1996

    # Adding line the indicate before and after IGN started
    plt.axvline(x=index_of_1996, color='red', linestyle='--', label='IGN Active Review Starts') 
    plt.title('Number of Games Released Each Year', fontsize=14)
    plt.xlabel('Year of Release', fontsize=12)
    plt.ylabel('Number of Games', fontsize=12)
    plt.xticks(rotation=45)  # rotate the x-axis labels for better readability
    plt.tight_layout()
    plt.show()

def Correlation_score_sales(df):
    correlation_df = df.copy()
    min_year_in_data = int(correlation_df['Year_of_Release'].min())
    max_year = int(correlation_df['Year_of_Release'].max())

    # create bin edges that start from the minimum year up to before 1996, 
    # and then after 2015
    if min_year_in_data < 1996:
        bin_edges = [min_year_in_data, 1996] + list(range(2001, 2021, 5)) 
    else:
        bin_edges = list(range(1996, 2021, 5))

    # ensure the last bin captures everything after 2015 up to the maximum year
    bin_edges.append(max_year + 1)

    # define labels for these group
    bin_labels = ['before 1996'] if min_year_in_data < 1996 else []
    bin_labels += [f'{start}-{start+4}' for start in range(1996, 2016, 5)]
    if max_year >= 2016:
        bin_labels.append('after 2015')

    # bin_edges = sorted(set(bin_edges))
    correlation_df['5-Year Bin'] = pd.cut(correlation_df['Year_of_Release'], \
                                            bins=bin_edges, labels=bin_labels, right=False)
    plt.figure(figsize=(10, 6))
    # create the FacetGrid
    g = sns.FacetGrid(correlation_df, col="5-Year Bin", col_wrap=3, height=3)

    # function for calculating for correlation coefficient and 
    def scatter_with_regression(x, y, **kwargs):
        corr, _ = pearsonr(x, y)
        label = f'r = {corr:.3f}'
        sns.regplot(x=x, y=y, order=3, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, **kwargs)
        plt.gca().text(0.05, 0.95, label, transform=plt.gca().transAxes, \
                        ha='left', va='top', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # map the custom function to the grid
    g.map(scatter_with_regression, "IGN_Score", "Global_Sales")

    # set the titles and labels
    g.fig.suptitle('IGN Score vs. Global Sale for Different Years', y=1.02)
    g.set_titles('Year {col_name}')
    g.set(ylim=(0, 30))

    # remove subplot label and name a new one.
    g.set_axis_labels(" ", " ")
    # X-axis label
    g.fig.text(0.5, 0.04, 'IGN Score', ha='center', va='center', fontsize=16) 
    # Y-axis label
    g.fig.text(0.04, 0.5, 'Global Sales(million)', \
                ha='center', va='center', rotation='vertical', fontsize=16)  
    g.fig.subplots_adjust(left = 0.1, bottom = 0.1)

    plt.show()

def Confidence_Interval(group):
    group = group.dropna()
    n = len(group)
    mean = np.mean(group)
    std = np.std(group, ddof=1)

    # take confidence level as 0.95 
    ci_lower = mean - 1.96 * std/np.sqrt(n)
    ci_upper = mean + 1.96 * std/np.sqrt(n)
    
    return pd.DataFrame({
        'Mean': [mean],
        'CI Lower Bound': [ci_lower],
        'CI Upper Bound': [ci_upper]
    })

def year_to_group(year):
    # aggregate every 5 years before 1996
    if year < 1978:
        return "before 1978"
    elif year < 1996:
        # starting from 1995, create groups backwards in 5-year intervals
        max_year = 1995
        while year <= max_year:
            if year > max_year - 5:
                return f"{max_year-4}-{max_year}"
            max_year -= 5
    else:
        return str(year)

def uncertainty_chart(df_input, IGN_color, user_color, critic_color, area_alpha, point_alpha):
    df = df_input.copy()    
    
    # apply the grouping function
    df['Year_Group'] = df['Year_of_Release'].apply(year_to_group)

    # apply confidence interval
    Uncertainty_IGN = df.groupby('Year_Group')['IGN_Score'].apply(Confidence_Interval).reset_index()
    df.dropna(subset = ['User_Score','Critic_Score'], inplace = True)
    Uncertainty_User = df.groupby('Year_Group')['User_Score'].apply(Confidence_Interval).reset_index()
    Uncertainty_Critic = df.groupby('Year_Group')['Critic_Score'].apply(Confidence_Interval).reset_index()

    # convert year groups to numeric values for plotting
    year_labels = sorted(set(df['Year_Group']))
    year_numeric = {label: i for i, label in enumerate(year_labels)}
    Uncertainty_IGN['Year_Numeric'] = Uncertainty_IGN['Year_Group'].map(year_numeric)
    Uncertainty_User['Year_Numeric'] = Uncertainty_User['Year_Group'].map(year_numeric)
    Uncertainty_Critic['Year_Numeric'] = Uncertainty_Critic['Year_Group'].map(year_numeric)
    df['Year_Numeric'] = df['Year_Group'].map(year_numeric)

    plt.figure(figsize=(10, 5))

    # line plot
    plt.plot(Uncertainty_IGN['Year_Numeric'], Uncertainty_IGN['Mean'], label='Mean IGN Score', color=IGN_color)
    plt.plot(Uncertainty_User['Year_Numeric'], Uncertainty_User['Mean'], label='Mean User Score', color=user_color)
    plt.plot(Uncertainty_Critic['Year_Numeric'], Uncertainty_Critic['Mean'], label='Mean Critic Score', color=critic_color)

    # CI area plot
    plt.fill_between(Uncertainty_IGN['Year_Numeric'], Uncertainty_IGN['CI Lower Bound'],\
                        Uncertainty_IGN['CI Upper Bound'], color=IGN_color, alpha=area_alpha)
    plt.fill_between(Uncertainty_User['Year_Numeric'], Uncertainty_User['CI Lower Bound'], \
                        Uncertainty_User['CI Upper Bound'], color=user_color, alpha=area_alpha)
    plt.fill_between(Uncertainty_Critic['Year_Numeric'], Uncertainty_Critic['CI Lower Bound'], \
                        Uncertainty_Critic['CI Upper Bound'], color=critic_color, alpha=area_alpha)

    # discrete point
    plt.scatter(df['Year_Numeric'], df['IGN_Score'], color=IGN_color, s=0.001, alpha=point_alpha)
    plt.scatter(df['Year_Numeric'], df['User_Score'], color=user_color, s=0.001, alpha=point_alpha)
    plt.scatter(df['Year_Numeric'], df['Critic_Score'], color=critic_color, s=0.001, alpha=point_alpha)

    plt.xlabel('Year of Release')
    plt.ylabel('Scores')
    plt.title('Uncertainty of IGN, User and Critic Scores Over Time')
    plt.xticks(ticks=list(year_numeric.values()), labels=list(year_numeric.keys()), rotation=90)
    plt.legend(loc='lower right')

    # add vertical line to highlight the lack of data issue
    plt.axvline(x=(year_numeric['1991-1995'] + year_numeric['1996']) / 2, color='black', linestyle='--')

    plt.show()

def one_hot_encoding(df):
    one_hot_encode_df = df.copy()
    one_hot_encode_df['Genres_IGN_str'] = one_hot_encode_df['Genres_IGN'].apply(lambda x: ','.join(x))
    genres_encoded = one_hot_encode_df['Genres_IGN_str'].str.get_dummies(sep=',')

    one_hot_encode_df = one_hot_encode_df.join(genres_encoded)
    columns_to_drop = ['Game', 'Publishers', 'Platforms', 'Genres_IGN', 'ESRB_Rating', 'NA_Sales', 'EU_Sales', 'JP_Sales',
                        'Other_Sales','Critic_Score', 'User_Score', 'Year_of_Release', 'Genres_IGN_str'] 
    genre_df = one_hot_encode_df.drop(columns=columns_to_drop)
    genre_columns = genre_df.drop(columns=['IGN_Score', 'Global_Sales']).columns

    return genre_columns, genre_df

def Top_x_genres(genre_df,genre_columns, x):
    genre_frequencies = genre_df[genre_columns].sum().sort_values(ascending=False)
    top_x_genres = genre_frequencies.head(x)

    genre_stats = {}

    for genre in top_x_genres.index:
        genre_data = genre_df[genre_df[genre] == 1]  
        sales_stats = Confidence_Interval(genre_data['Global_Sales'])
        score_stats = Confidence_Interval(genre_data['IGN_Score'])
        
        genre_stats[genre] = {
            'Sales Mean': sales_stats['Mean'].values[0],
            'Sales CI Lower': sales_stats['CI Lower Bound'].values[0],
            'Sales CI Upper': sales_stats['CI Upper Bound'].values[0],
            'Score Mean': score_stats['Mean'].values[0],
            'Score CI Lower': score_stats['CI Lower Bound'].values[0],
            'Score CI Upper': score_stats['CI Upper Bound'].values[0]
        }

    genre_stats_df = pd.DataFrame(genre_stats).T
    return genre_frequencies, genre_stats_df

def plot_top_x_genres(genre_frequencies, x=25):
    """For Genre, a game could contain multi-genre.
    Therefore, we are going to convert genre labels from a list format into a one-hot encoded format,
    allowing for numerical analysis and aggregation based on genre.
    Using the one-hot encoded data, we calculate the frequency of each genre and select the top 25 genres for deeper analysis. 
    For these genres, we compute average sales and scores, along with their 95% confidence intervals.
    And base on the data, we will plot an error bars to represent confidence intervals of average sale and score data. """

    top_genres = genre_frequencies.head(x)
    plt.figure(figsize=(8, 6))
    bar_plot = sns.barplot(x=top_genres.values, y=top_genres.index, color='dodgerblue')
    bar_plot.set_title(f'Top {x} Most Frequent Genres')
    bar_plot.set_xlabel('Frequency')
    bar_plot.set_ylabel('Genres')
    plt.show()

def plot_combined_error_bars(genre_stats_df):
    genre_stats_df = genre_stats_df.sort_values(by='Sales Mean', ascending=False)
    genres = genre_stats_df.index
    x = len(genre_stats_df)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Sales
    x_sales = genre_stats_df['Sales Mean']
    lower_errors_sales = x_sales - genre_stats_df['Sales CI Lower']
    upper_errors_sales = genre_stats_df['Sales CI Upper'] - x_sales
    errors_sales = [lower_errors_sales, upper_errors_sales]

    ax1.errorbar(x_sales, genres, xerr=errors_sales, fmt='o', label='Sales', color='dodgerblue', ecolor='dodgerblue', alpha=0.5,
                 capsize=3, elinewidth=2, markeredgewidth=2, markersize=8)
    ax1.set_xlabel('Average Sales (in millions)', fontsize=12, fontweight='bold', color='dodgerblue')
    ax1.set_ylabel('Genre', fontsize=12)
    ax1.set_title(f'Average Sales and Scores by Top {x} Genres with 95% Confidence Intervals')

    # Scores
    x_scores = genre_stats_df['Score Mean']
    lower_errors_scores = x_scores - genre_stats_df['Score CI Lower']
    upper_errors_scores = genre_stats_df['Score CI Upper'] - x_scores
    errors_scores = [lower_errors_scores, upper_errors_scores]

    ax2 = ax1.twiny()
    ax2.errorbar(x_scores, genres, xerr=errors_scores, fmt='^', label='Scores', color='crimson', ecolor='crimson', alpha=0.5,
                 capsize=3, elinewidth=2, markeredgewidth=2, markersize=8)
    ax2.set_xlabel('Average Scores', fontsize=12, fontweight='bold', color='crimson')
    ax2.set_xlim(5, 10)
    
    ax1.tick_params(axis='x', colors='dodgerblue')
    ax2.tick_params(axis='x', colors='crimson')

    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')

    ax1.invert_yaxis()
    plt.tight_layout()
    plt.show()

def folium_heatmap(df_input, columns, output_file='heatmap.html'):
    df = df_input.copy()

    # load world geometries and merge with data
    world = gpd.read_file("./data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
    world = world.rename(columns={'NAME': 'Country'})
    merged = world.merge(df, how='inner', on='Country')
    
    # create GeoJSON for Folium
    geo_data = merged[['Country', 'geometry']].to_json()

    # create a Folium map
    m = folium.Map(location=[0, 0], zoom_start=1)

    # add Choropleth layer for each column
    for column in columns:
        Choropleth(
            geo_data=geo_data,
            name=column,
            data=merged,
            columns=['Country', column],
            key_on='feature.properties.Country',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f'{column} by Country'
        ).add_to(m)

    # add a layer control panel to the map
    folium.LayerControl().add_to(m)

    return m

def grouped_bar_chart(df_input, top):
    """Sales data are cumulative. We can not directly compare the sales over the release date,
    as the later released game will have fewer sales compared to the earlier one.
    Normalize Sales by Time Since Release may not be a good solution because,
    in reality, video game sales often follow an exponential decay model, and it is tricky to estimate the decay rate.
    To due with this issue, we group the sales data by release year, and compare the top n publishers within the same period of release year."""

    df = df_input.copy()
    sales_regions = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales']

    # 5-year groups, group all before 1995 as one due to lack of data
    df['Year_Group'] = df['Year_of_Release'].apply(lambda x: f'{(x//5)*5}-{(x//5)*5 + 4}' if x >= 1995 else '1975-1994')
    

    # identify top n publishers by total global sales
    top_publishers = df.groupby('Publishers')[sales_regions[0]].sum().nlargest(top).index

    # color palette
    color_palette = sns.color_palette("tab10", n_colors=top)

    # create a dictionary mapping publishers to colors
    color_map = dict(zip(top_publishers, color_palette))

    # create a 2 by 2 grid layout for the subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axs = axs.flatten()

    for i, sales_region in enumerate(sales_regions):
        # filter data for top n publishers
        df_top_publishers = df[df['Publishers'].isin(top_publishers)].copy()

        # group by year group and publisher, then sum sales
        grouped_data = df_top_publishers.groupby(['Year_Group', 'Publishers'])[sales_region].sum().unstack()

        # normalize the sales within each 5-year period
        normalized_data = grouped_data.div(grouped_data.sum(axis=1), axis=0)

        # plot grouped bar chart with consistent colors
        normalized_data.plot(kind='bar', stacked=True, ax=axs[i], \
                                color=[color_map[col] for col in normalized_data.columns])

        axs[i].set_xlabel('Year of Release')
        axs[i].set_ylabel('Normalized Sales')
        axs[i].set_title(sales_region.replace('_', ' '))
        if i == 0:
            axs[i].legend(title='Publishers')
        else:
            axs[i].get_legend().remove()
        axs[i].tick_params(axis='x', rotation=45)

        # add vertical line to highlight the lack of data issue
        year_groups = list(normalized_data.index)
        idx_1994 = year_groups.index('1975-1994')
        idx_1995 = year_groups.index('1995-1999')
        axs[i].axvline(x=(idx_1994 + idx_1995) / 2, color='black', linestyle='--')


        # hide y-axis for right side charts
        if i % 2 != 0:
            axs[i].set_ylabel('')
            axs[i].set_yticklabels([])

    fig.suptitle(f'Normalized comparison of Sales for Top {top} Publishers Across Different Regions', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

# 1. Data preprocessing
data_IGN_clean, data_sale_clean = data_cleaning(data_IGN_original, data_sale_original)

data_IGN_single_platform, data_sale_single_platform = combine_platform(data_IGN_clean, data_sale_clean)

data_IGN_single_platform['mapped_game'] = mapping_game_name(data_IGN_single_platform["game"], \
                                                            data_sale_single_platform["Name"])

data_IGN_single_platform['mapped_game'] = data_IGN_single_platform['mapped_game'].str.strip().str.replace('  ', ' ',)

data_IGN_mapped = combine_mapped_game(data_IGN_single_platform)

# 2. Merging data
merged_df = pd.merge(
    data_IGN_mapped,
    data_sale_single_platform,
    left_on='mapped_game',
    right_on='Name',
    how='inner'
)

merged_clean_df = remove_unnecessary_columns(merged_df)

# 3. Aggregation
publisher_df = agg_by_publisher_releaseYear(merged_clean_df)

data_publisher_country = get_country_of_publishers()

publisher_country_merge = pd.merge(publisher_df, data_publisher_country, how='inner')

country_df = agg_by_country(publisher_country_merge)

# 4. Analysis
plt.ion() # Enable interactive mode

data_occurrence(merged_clean_df)

Correlation_score_sales(merged_clean_df)

uncertainty_chart(merged_clean_df, 'red', 'green', 'blue', 0.3, 1)

genre_columns, genre_df = one_hot_encoding(merged_clean_df)
genre_frequencies, genre_stats_df = Top_x_genres(genre_df,genre_columns, x = 25)
plot_top_x_genres(genre_frequencies, x=25)
plot_combined_error_bars(genre_stats_df)

columns = ['IGN_Score', 'User_Score', 'Critic_Score', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales']
folium_heatmap(country_df, columns).show_in_browser()

grouped_bar_chart(publisher_df, 10)

plt.ioff() # Disable interactive mode

input("\nPress Enter to close all plots...")