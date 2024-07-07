# Video Game Sales and Scores Analysis

## Overview
This repository contains the project report and Jupyter notebooks for our analysis of video game sales and scores. The project aims to explore the correlation between video game sales and scores, identify patterns under different publishers and countries, and provide insights into the gaming industry's dynamics. This analysis focuses primarily on data manipulation and visualization to uncover meaningful insights. Although the current scope does not include machine learning or other advanced tools, the project has the potential for future enhancements using these techniques.

## Data Source
We utilized three datasets with one supporting data for this project:

<img width="580" alt="image" src="https://github.com/derickkan3356/Video-Game-Sales-and-Scores-Analysis/assets/137174558/c45a8ca0-8a37-4780-8e75-9ba47e509c2c">

1. **Video Game Sales Dataset:**
   - Provides information on sales performance and popularity of various video games.
   - Source: [Kaggle](https://www.kaggle.com/datasets/ibriiee/video-games-sales-dataset-2022-updated-extra-feat)

2. **IGN Scores Dataset:**
   - Contains details of all the games reviewed on the IGN website.
   - Source: [Kaggle](https://www.kaggle.com/datasets/advancedforestry/ign-scores-dataset)

3. **Publisher's Country List:**
   - A list of video game publishers and their corresponding countries.
   - Obtained by web scraping Wikipedia using BeautifulSoup.
   - Source: [Wikipedia](https://en.wikipedia.org/wiki/List_of_video_game_publishers)

4. **Country Geometry Data:**
   - Geometry data for country names to support geographic visualizations.
   - Folder: `data/ne_110m_admin_0_countries/`
   - Source: [Natural Earth](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/)

## Manipulation Methodology

<img width="269" alt="image" src="https://github.com/derickkan3356/Video-Game-Sales-and-Scores-Analysis/assets/137174558/d732dc4e-b489-40b9-a349-031609dfda52">


1. **Data Cleaning:**
   - Standardize date formats and drop rows with missing values in key columns (game name, IGN score).
   - Remove irrelevant columns and ensure consistent game names by normalizing character formatting.
   - Convert data types as needed and standardize scores to a 10-point scale.

2. **Prepare Game Names for Merging:**
   - Aggregate game data across different platforms by:
     - Taking the minimum year of release.
     - Summing the sales figures.
     - Averaging the scores.
     - Combining platform lists.
   - Handle games with alternative versions by:
     - Identifying and merging alternative versions from the sales dataset.
     - Ensuring consistency and preventing duplication.

3. **Merge Data by Game Name:**
   - Combine sales and IGN datasets, ensuring only unique and accurately matched rows remain.
   - Retain relevant columns, preferring IGN data where it offers more detail.
   - Verify merging quality and manually adjust game names if necessary.

4. **Aggregate by Publisher:**
   - For games with multiple publishers, adjust sales figures to avoid inflation by dividing total sales by the number of publishers before exploding the list into separate rows.

5. **Aggregate by Country:**
   - Use web scraping to obtain a list of publishers and their countries.
   - Normalize country names and merge with publisher data to perform aggregation by country.

## Analysis

### Correlation of IGN score and global sales
Considering IGN was created in 1996, and the limited number of games data before 1996, we grouped the game data from before 1996 into one category for creating a more robust and statistical analysis (figure 1b). From 1996 onward, the data is segmented into 5-year intervals to observe trends over time.

![image](https://github.com/derickkan3356/Video-Game-Sales-and-Scores-Analysis/assets/137174558/c6a2f224-bb51-481d-a147-c438bc9085dd)

![image](https://github.com/derickkan3356/Video-Game-Sales-and-Scores-Analysis/assets/137174558/0fb118ae-0472-4390-9820-16bdb1703623)

**Consistent Positive Correlation**

The correlation coefficients (denoted as 𝑟) across different time periods range with weak positive correlation. This suggests that while there is some association between game scores and sales, it's not particularly strong but this is consistent across all time periods as a positive proportion.

**Correlation Strength Over Time**

From the years before 2006, the correlation is modest but keeping an increasing on the coefficient continuously, which indicated the impact of critical reviews on sales became slightly more pronounced.

But in period 2006-2010, the correlation drops again among periods, it emphasizes some games that despite lower scores, achieved significant sales, which might be presented by outliers influencing the data.

And for the years after 2011, there is a noticeable increase in correlation, likely indicating a consolidation of market trends where highly-rated games are strongly aligned.

The correlation between IGN scores and global sales keeping a positive Correlation and exhibits modest fluctuations across different periods. Although the correlation dropped slightly in 2006-2010, it stabilizes with a stronger relationship from 2011 onwards and peaking in latest years in dataset. This trend suggests an increasing in correlation of game reviews on sales by years.

### Score trending and uncertainty analysis

![image](https://github.com/derickkan3356/Video-Game-Sales-and-Scores-Analysis/assets/137174558/bd343d89-86ab-432a-9b2f-c79b929ce2dc)

**Chart Overview**

The visualization presented is a comprehensive analysis of the uncertainty of IGN, User, and Critic scores on video games over time. The chart displays the mean scores for each category, with the shaded areas representing the 95% confidence intervals and points illustrating the distribution of scores for each year. Due to the limited data availability before 1996, the periods before 1996 were grouped into 5-year intervals to enhance visualization clarity.

**Public vs. Critic Perception**

Despite the challenges posed by limited data before 1996, the overall differences between the three types of scores (IGN, User, and Critic) are minimal, about 1 point of score. However, a notable trend is observed where user scores exhibit a slight downward trajectory, whereas IGN and Critic scores show a slight upward trend. The intersection around the year 2010 indicates a shift where public ratings began to be generally lower than those of IGN and other critics. This suggests a divergence in perception between the public and professional reviewers starting around 2010. It questions about the factors influencing public perception versus professional reviews. Potential reasons for this divergence could include changes in the gaming industry, differences in expectations between the public and critics, or the impact of social media and online reviews.

**Stability and Variability of Scores**

The period between 2000 and 2010 shows remarkable stability in scores, as indicated by relatively narrow confidence intervals. However, post-2010, the scores become less stable, with wider confidence intervals. This increase in variability could reflect a period of rapid change in the industry, with the introduction of new gaming platforms, genres, and shifts in consumer preferences. The pre-1996 period is characterized by high instability, likely due to the sparse data available during those years. An important observation is the positive correlation between the amount of data and the variation in scores. This is evident from the discrete point distribution, where years with more data points show more stable scores, while years with fewer data points exhibit greater variability. This correlation suggests that the observed variations might be influenced by the volume of available data rather than solely reflecting the true variability in game scores. This is a crucial consideration for us when interpreting the uncertainty analysis.

### Game Genre Performance on Sales and Scores

![image](https://github.com/derickkan3356/Video-Game-Sales-and-Scores-Analysis/assets/137174558/e75d3dc4-c80a-45bd-a948-28aeaa71544b)

**Correlation Overview**

From an analysis by game genre, despite expectations that higher scores might correlate with higher sales, the plot does not distinctly show this relationship across genres. Each genre exhibits varied patterns, with some high-scoring games not necessarily translating to high sales, and vice versa. This suggests that factors beyond individual game ratings, including genre-specific market, publisher loyalty or brand recognition, play significant roles in determining sales outcomes.

**Genre-specific Performance**

Indicating the width of the confidence intervals for both sales and scores showing that the performance of games can significantly vary within genres. Some genres, such as First Person or Compilation, contain a wide confidence interval for sales or scores. This suggests significant variability within these genres, which could be attributed to the diverse types of games grouped under a single genre label.
This result indicate some genres' capacity to include both high-performing blockbusters and lower-tier games, which can vary significantly in quality and market reception. 

**Market niche games**

At the same time, there Genres with narrower audience appeal, such as Puzzle and Trivia, tend to have lower sales volumes but can achieve moderate to high scores, suggesting that niche games can be well-received critically despite lower sales figures. Therefore, understanding the dynamics of each genre could be crucial for predicting sales outcomes or market rating.

### Geographic Heatmap Analysis of Video Game Scores and Sales

<img width="624" alt="image" src="https://github.com/derickkan3356/Video-Game-Sales-and-Scores-Analysis/assets/137174558/b23b8d1d-4bb0-4c51-a043-248bce7118c2">

**Chart Overview**

The visualization consists of two sets of geographic heatmaps. The three heatmaps on the left-hand side represent the distribution of scores from IGN, other critics, and users. The three heatmaps on the right-hand side depict the sales distribution of publishers for North America, Europe, and Japan. In these heatmaps, red indicates the highest amount, yellow indicates the smallest amount, and black indicates no data.

**Score Distribution**

For the score distribution, the patterns reveal that publishers from Europe and Japan receive consistent ratings across IGN, other critics, and user reviews, indicating a general consensus on game quality from these regions. This consistency suggests that games from Europe and Japan are perceived to maintain a high standard, leading to similar evaluations from both professional critics and general users. IGN and user scores are notably more positive towards North American publishers, suggesting regional favorability or higher appreciation for games from these countries. Conversely, critic and user scores tend to be more negative towards Russian publishers, which could be due to regional biases, perceived quality differences, or historical context influencing the reception of Russian games. 

**Sales Distribution**

Each sales heatmap represent the sales distribution in each region, while the map colors indicate the country of the video game publishers. A significant finding is the concentrated dominance of Japanese publishers in the Japan sales map, illustrating a strong preference for local games likely driven by cultural affinity and language barriers, especially in earlier years when foreign games lacked Japanese language support. In contrast, while North American and European sales show a concentration in publishers from France, they still exhibit a more even distribution among international publishers compared to Japan. This even distribution suggests that North American and European gamers are more open to games from various countries, reflecting a globalized market with diverse preferences, unlike the localized preference seen in Japan.

### Sales proportion of top 10 publishers across different regions

![image](https://github.com/derickkan3356/Video-Game-Sales-and-Scores-Analysis/assets/137174558/15e97605-5506-4fbc-a1e5-4efe24d9fbeb)

**Chart Overview**

This visualization presents a normalized comparison of sales for the top 10 publishers across different regions. The sales data are cumulative, and direct comparison over release dates is not feasible due to the inherent advantage earlier releases have in accumulating sales. To address this, the sales data are grouped and normalized within periods of release years, allowing for comparison of the sales proportion of the top publishers within each period. The vertical line in the charts indicates a period with potential data insufficiency, marking the data before this line as possibly unreliable.

**General Trends**

The normalized sales data reveal regional preferences and market dynamics among the top video game publishers. In Japan, the sales focus predominantly on Japanese publishers such as Sony, Sega, Nintendo, and Capcom. These publishers also contribute significantly to sales in other regions, but their dominance is less pronounced compared to Japan. This finding aligns with the earlier heatmap analysis, highlighting the strong local preference for Japanese publishers in their home market.
The sales figures in Europe and North America closely resemble the global sales distribution. This similarity suggests two key implications:
1.The preferences for publishers are similar among NA and EU markets, indicating a shared taste in video game publishers.
2.The NA and EU markets are substantial enough to significantly influence the global market trends, reflecting their large consumer base and market size.

**Publisher-Specific Trends**

Nintendo: Despite the unreliable data period, Nintendo shows significant growth in the Japanese market. However, its market share varies across time in other regions.
Capcom, Sega, and Sony: These publishers exhibit a decreasing trend in market share across all regions. This decline suggests that their dominance is waning, potentially due to increased competition or shifts in market preferences.
Microsoft and Electronic Arts: These publishers demonstrate an increasing trend in market share across all regions. Their growth indicates a rising preference for their games, potentially driven by successful franchises or effective market strategies.
The trends indicate that Japanese publishers are experiencing a decreasing market share globally. This shift suggests that foreign publishers are gradually replacing Japanese publishers, reflecting a broader trend of globalization and changing market dynamics in the video game industry.

## Contribution
This project was a collaborative effort by **Derick Ka Lok Kan** and **Hinson Cheuk Hin Kwan**

| Task                                                                 | Derick Ka Lok Kan | Hinson Cheuk Hin Kwan |
|----------------------------------------------------------------------|-------------------|-----------------------|
| Exploring dataset                                                    | ✓                 | ✓                     |
| Data cleaning                                                        | ✓                 | ✓                     |
| Data merging                                                         | ✓                 | ✓                     |
| Data aggregation                                                     | ✓                 | ✓                     |
| Correlation of IGN score and global sales                            |                   | ✓                     |
| Score trending and uncertainty analysis                              | ✓                 |                       |
| Game Genre Performance on Sales and Scores                           |                   | ✓                     |
| Geographic Heatmap Analysis of Video Game Scores and Sales           | ✓                 |                       |
| Sales proportion of top 10 publishers across different regions       | ✓                 |                       |
| Code review                                                          | ✓                 | ✓                     |
| Report design                                                        | ✓                 | ✓                    |
