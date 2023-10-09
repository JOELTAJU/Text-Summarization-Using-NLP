import importlib

def program_one():
    print("Executing program one...")
        # -*- coding: utf-8 -*-
    import numpy as np
    import pandas as pd
    import warnings
    import re
    import nltk
    from nltk import word_tokenize
    from nltk.tokenize import sent_tokenize
    from textblob import TextBlob
    import string
    from string import punctuation
    from nltk.corpus import stopwords
    from statistics import mean
    from heapq import nlargest
    from wordcloud import WordCloud

    stop_words = set(stopwords.words('english'))
    punctuation = punctuation + '\n' + '—' + '“' + ',' + '”' + '‘' + '-' + '’'
    warnings.filterwarnings('ignore')

    # Importing the dataset
    df_1 = pd.read_csv("articles1.csv")
    df_2 = pd.read_csv("articles2.csv")
    df_3 = pd.read_csv("articles3.csv")

    # Checking if the columns are same or not
    (df_1.columns == df_2.columns)

    # Checking if the columns are same or not
    (df_2.columns == df_3.columns)

    # Making one Dataframe by appending all of them for the further process
    d = [df_1, df_2, df_3]
    df = pd.concat(d, keys=['x', 'y', 'z'])
    df.rename(columns={'content': 'article'}, inplace=True)
    (df.head())

    # Shape of the dataset
    ("The shape of the dataset : ", df.shape)

    # Dropping the unnecessary columns
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.head()

    # Countplot shows the distribution of Publication

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = [15, 8]
    sns.set(font_scale=1.2, style="darkgrid")
    sns_year = sns.countplot(data=df, x="publication", color="darkcyan")
    plt.xticks(rotation=45)
    sns_year.set(xlabel="Publication", ylabel="Count", title="Distribution of Publication")
    # plt.show()


    # Replacing the unnecessary row value of year with it's actual values
    df['year'] = df['year'].replace(
        "https://www.washingtonpost.com/outlook/tale-of-a-woman-who-died-and-a-woman-who-killed-in-the-northern-ireland-conflict/2019/03/08/59e75dd4-2ecd-11e9-8ad3-9a5b113ecd3c_story.html",
        2019)

    # Years
    (df['year'].value_counts())

    # Countplot shows the distribution of the articles according to the year


    plt.rcParams['figure.figsize'] = [15, 8]
    sns.set(font_scale=1.2, style='whitegrid')
    sns_year = sns.countplot(data=df, x="year", color='darkcyan')
    sns_year.set(xlabel="Year", ylabel="Count", title="Distribution of the articles according to the year")
    # plt.show()


    # Authors
    df['author'].value_counts()

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['figure.figsize'] = [15, 15]
    sns.set(font_scale=1, style='whitegrid')

    df_author = df.author.value_counts().head(80)

    sns.barplot(x=df_author.values, y=df_author.index)
    plt.xlabel("Count",fontsize=12)
    plt.ylabel("Author")
    plt.title("The Most Frequent Authors")
    sns.despine(left=True)
    plt.yticks(fontsize=8)

    plt.grid(True,axis='both')  # Enable grid lines

    sns.set_style('ticks')  # Set the style to 'ticks'
    plt.xticks(rotation=0, ha='center')  # Adjust rotation and alignment of x-axis labels

    plt.tight_layout()
    #plt.show()


    # Changing the value "The Associated Press" to "Associated Press"
    df['author'] = df['author'].replace("The Associated Press", "Associated Press")


    contractions_dict = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "doesn’t": "does not",
    "don't": "do not",
    "don’t": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y’all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "ain’t": "am not",
    "aren’t": "are not",
    "can’t": "cannot",
    "can’t’ve": "cannot have",
    "’cause": "because",
    "could’ve": "could have",
    "couldn’t": "could not",
    "couldn’t’ve": "could not have",
    "didn’t": "did not",
    "doesn’t": "does not",
    "don’t": "do not",
    "don’t": "do not",
    "hadn’t": "had not",
    "hadn’t’ve": "had not have",
    "hasn’t": "has not",
    "haven’t": "have not",
    "he’d": "he had",
    "he’d’ve": "he would have",
    "he’ll": "he will",
    "he’ll’ve": "he will have",
    "he’s": "he is",
    "how’d": "how did",
    "how’d’y": "how do you",
    "how’ll": "how will",
    "how’s": "how is",
    "i’d": "i would",
    "i’d’ve": "i would have",
    "i’ll": "i will",
    "i’ll’ve": "i will have",
    "i’m": "i am",
    "i’ve": "i have",
    "isn’t": "is not",
    "it’d": "it would",
    "it’d’ve": "it would have",
    "it’ll": "it will",
    "it’ll’ve": "it will have",
    "it’s": "it is",
    "let’s": "let us",
    "ma’am": "madam",
    "mayn’t": "may not",
    "might’ve": "might have",
    "mightn’t": "might not",
    "mightn’t’ve": "might not have",
    "must’ve": "must have",
    "mustn’t": "must not",
    "mustn’t’ve": "must not have",
    "needn’t": "need not",
    "needn’t’ve": "need not have",
    "o’clock": "of the clock",
    "oughtn’t": "ought not",
    "oughtn’t’ve": "ought not have",
    "shan’t": "shall not",
    "sha’n’t": "shall not",
    "shan’t’ve": "shall not have",
    "she’d": "she would",
    "she’d’ve": "she would have",
    "she’ll": "she will",
    "she’ll’ve": "she will have",
    "she’s": "she is",
    "should’ve": "should have",
    "shouldn’t": "should not",
    "shouldn’t’ve": "should not have",
    "so’ve": "so have",
    "so’s": "so is",
    "that’d": "that would",
    "that’d’ve": "that would have",
    "that’s": "that is",
    "there’d": "there would",
    "there’d’ve": "there would have",
    "there’s": "there is",
    "they’d": "they would",
    "they’d’ve": "they would have",
    "they’ll": "they will",
    "they’ll’ve": "they will have",
    "they’re": "they are",
    "they’ve": "they have",
    "to’ve": "to have",
    "wasn’t": "was not",
    "we’d": "we would",
    "we’d’ve": "we would have",
    "we’ll": "we will",
    "we’ll’ve": "we will have",
    "we’re": "we are",
    "we’ve": "we have",
    "weren’t": "were not",
    "what’ll": "what will",
    "what’ll’ve": "what will have",
    "what’re": "what are",
    "what’s": "what is",
    "what’ve": "what have",
    "when’s": "when is",
    "when’ve": "when have",
    "where’d": "where did",
    "where’s": "where is",
    "where’ve": "where have",
    "who’ll": "who will",
    "who’ll’ve": "who will have",
    "who’s": "who is",
    "who’ve": "who have",
    "why’s": "why is",
    "why’ve": "why have",
    "will’ve": "will have",
    "won’t": "will not",
    "won’t’ve": "will not have",
    "would’ve": "would have",
    "wouldn’t": "would not",
    "wouldn’t’ve": "would not have",
    "y’all": "you all",
    "y’all": "you all",
    "y’all’d": "you all would",
    "y’all’d’ve": "you all would have",
    "y’all’re": "you all are",
    "y’all’ve": "you all have",
    "you’d": "you would",
    "you’d’ve": "you would have",
    "you’ll": "you will",
    "you’ll’ve": "you will have",
    "you’re": "you are",
    "you’re": "you are",
    "you’ve": "you have",
    }
    import re
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    # Function to clean the html from the article
    def cleanhtml(raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext

    # Function expand the contractions if there's any
    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)

    # Function to preprocess the articles
    def preprocessing(article):
        global article_sent
        
        # Converting to lowercase
        article = article.str.lower()
        
        # Removing the HTML
        article = article.apply(lambda x: cleanhtml(x))
        
        # Removing the email ids
        article = article.apply(lambda x: re.sub('\S+@\S+','', x))
        
        # Removing The URLS
        article = article.apply(lambda x: re.sub("((http\://|https\://|ftp\://)|(www.))+(([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(/[a-zA-Z0-9%:/-_\?\.'~]*)?",'', x))
        
        # Removing the '\xa0'
        article = article.apply(lambda x: x.replace("\xa0", " "))
        
        # Removing the contractions
        article = article.apply(lambda x: expand_contractions(x))
        
        # Stripping the possessives
        article = article.apply(lambda x: x.replace("'s", ''))
        article = article.apply(lambda x: x.replace('’s', ''))
        article = article.apply(lambda x: x.replace("\'s", ''))
        article = article.apply(lambda x: x.replace("\’s", ''))
        
        # Removing the Trailing and leading whitespace and double spaces
        article = article.apply(lambda x: re.sub(' +', ' ',x))
        
        # Copying the article for the sentence tokenization
        article_sent = article.copy()
        
        # Removing punctuations from the article
        article = article.apply(lambda x: ''.join(word for word in x if word not in punctuation))
        
        # Removing the Trailing and leading whitespace and double spaces again as removing punctuation might
        # Lead to a white space
        article = article.apply(lambda x: re.sub(' +', ' ',x))
        
        # Removing the Stopwords
        article = article.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
        
        return article

    # Function to normalize the word frequency which is used in the function word_frequency
    def normalize(li_word):
        global normalized_freq
        normalized_freq = []
        for dictionary in li_word:
            max_frequency = max(dictionary.values())
            for word in dictionary.keys():
                dictionary[word] = dictionary[word]/max_frequency
            normalized_freq.append(dictionary)
        return normalized_freq

    # Function to calculate the word frequency
    def word_frequency(article_word):
        word_frequency = {}
        li_word = []
        for sentence in article_word:
            for word in word_tokenize(sentence):
                if word not in word_frequency.keys():
                    word_frequency[word] = 1
                else:
                    word_frequency[word] += 1
            li_word.append(word_frequency)
            word_frequency = {}
        normalize(li_word)
        return normalized_freq

    # Function to Score the sentence which is called in the function sent_token
    def sentence_score(li):
        global sentence_score_list
        sentence_score = {}
        sentence_score_list = []
        for list_, dictionary in zip(li, normalized_freq):
            for sent in list_:
                for word in word_tokenize(sent):
                    if word in dictionary.keys():
                        if sent not in sentence_score.keys():
                            sentence_score[sent] = dictionary[word]
                        else:
                            sentence_score[sent] += dictionary[word]
            sentence_score_list.append(sentence_score)
            sentence_score = {}
        return sentence_score_list

    # Function to tokenize the sentence
    def sent_token(article_sent):
        sentence_list = []
        sent_token = []
        for sent in article_sent:
            token = sent_tokenize(sent)
            for sentence in token:
                token_2 = ''.join(word for word in sentence if word not in punctuation)
                token_2 = re.sub(' +', ' ',token_2)
                sent_token.append(token_2)
            sentence_list.append(sent_token)
            sent_token = []
        sentence_score(sentence_list)
        return sentence_score_list

    # Function which generates the summary of the articles (This uses the 20% of the sentences with the highest score)
    def summary(sentence_score_OwO):
        summary_list = []
        for summ in sentence_score_OwO:
            select_length = int(len(summ)*0.25)
            summary_ = nlargest(select_length, summ, key = summ.get)
            summary_list.append(".".join(summary_))
        return summary_list


    # Functions to change the article string (if passed) to change it to generate a pandas series
    def make_series(art):
        global dataframe
        data_dict = {'article' : [art]}
        dataframe = pd.DataFrame(data_dict)['article']
        return dataframe

    # Function which is to be called to generate the summary which in further calls other functions alltogether
    def article_summarize(artefact):
        
        if type(artefact) != pd.Series:
            artefact = make_series(artefact)
        
        df = preprocessing(artefact)
        
        word_normalization = word_frequency(df)
        
        sentence_score_OwO = sent_token(article_sent)
        
        summarized_article = summary(sentence_score_OwO)
        
        return summarized_article


    # Generating the Word Cloud of the article using the preprocessing and make_series function mentioned below
    def word_cloud(art):
        art_ = make_series(art)
        OwO = preprocessing(art_)
        wordcloud_ = WordCloud(height = 500, width = 1000, background_color = 'white').generate(art)
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud_, interpolation='bilinear')
        plt.axis('off');
    # Generating the summaries for the first 100 articles
    summaries = article_summarize(df['article'][0:100])

    print ("The Actual length of the article is : ", len(df['article'][0]))
    print(df['article'][0])


    print ("The length of the summarized article is : ", len(summaries[0]))
    summaries[0]
    print("The summary of the article is:", summaries[0])




def program_two():
    print("Executing program two...")
    from rouge import Rouge

    # Assuming you have a list of reference summaries and generated summaries
    reference_summaries = ["Congressional Republicans have a new fear when it comes to their health care lawsuit against the Obama administration: They might win. The incoming Trump administration could choose to no longer defend the executive branch against the suit, which challenges the administration’s authority to spend billions of dollars on health insurance subsidies for   and   Americans, handing House Republicans a big victory on    issues. But a sudden loss of the disputed subsidies could conceivably cause the health care program to implode, leaving millions of people without access to health insurance before Republicans have prepared a replacement. That could lead to chaos in the insurance market and spur a political backlash just as Republicans gain full control of the government. To stave off that outcome, Republicans could find themselves in the awkward position of appropriating huge sums to temporarily prop up the Obama health care law, angering conservative voters who have been demanding an end to the law for years. In another twist, Donald J. Trump’s administration, worried about preserving executive branch prerogatives, could choose to fight its Republican allies in the House on some central questions in the dispute. Eager to avoid an ugly political pileup, Republicans on Capitol Hill and the Trump transition team are gaming out how to handle the lawsuit, which, after the election, has been put in limbo until at least late February by the United States Court of Appeals for the District of Columbia Circuit. They are not yet ready to divulge their strategy. “Given that this pending litigation involves the Obama administration and Congress, it would be inappropriate to comment,” said Phillip J. Blando, a spokesman for the Trump transition effort. “Upon taking office, the Trump administration will evaluate this case and all related aspects of the Affordable Care Act. ” In a potentially   decision in 2015, Judge Rosemary M. Collyer ruled that House Republicans had the standing to sue the executive branch over a spending dispute and that the Obama administration had been distributing the health insurance subsidies, in violation of the Constitution, without approval from Congress. The Justice Department, confident that Judge Collyer’s decision would be reversed, quickly appealed, and the subsidies have remained in place during the appeal. In successfully seeking a temporary halt in the proceedings after Mr. Trump won, House Republicans last month told the court that they “and the  ’s transition team currently are discussing potential options for resolution of this matter, to take effect after the  ’s inauguration on Jan. 20, 2017. ” The suspension of the case, House lawyers said, will “provide the   and his future administration time to consider whether to continue prosecuting or to otherwise resolve this appeal. ” Republican leadership officials in the House acknowledge the possibility of “cascading effects” if the   payments, which have totaled an estimated $13 billion, are suddenly stopped. Insurers that receive the subsidies in exchange for paying    costs such as deductibles and   for eligible consumers could race to drop coverage since they would be losing money. Over all, the loss of the subsidies could destabilize the entire program and cause a lack of confidence that leads other insurers to seek a quick exit as well. Anticipating that the Trump administration might not be inclined to mount a vigorous fight against the House Republicans given the  ’s dim view of the health care law, a team of lawyers this month sought to intervene in the case on behalf of two participants in the health care program. In their request, the lawyers predicted that a deal between House Republicans and the new administration to dismiss or settle the case “will produce devastating consequences for the individuals who receive these reductions, as well as for the nation’s health insurance and health care systems generally. ” No matter what happens, House Republicans say, they want to prevail on two overarching concepts: the congressional power of the purse, and the right of Congress to sue the executive branch if it violates the Constitution regarding that spending power. House Republicans contend that Congress never appropriated the money for the subsidies, as required by the Constitution. In the suit, which was initially championed by John A. Boehner, the House speaker at the time, and later in House committee reports, Republicans asserted that the administration, desperate for the funding, had required the Treasury Department to provide it despite widespread internal skepticism that the spending was proper. The White House said that the spending was a permanent part of the law passed in 2010, and that no annual appropriation was required  —   even though the administration initially sought one. Just as important to House Republicans, Judge Collyer found that Congress had the standing to sue the White House on this issue  —   a ruling that many legal experts said was flawed  —   and they want that precedent to be set to restore congressional leverage over the executive branch. But on spending power and standing, the Trump administration may come under pressure from advocates of presidential authority to fight the House no matter their shared views on health care, since those precedents could have broad repercussions. It is a complicated set of dynamics illustrating how a quick legal victory for the House in the Trump era might come with costs that Republicans never anticipated when they took on the Obama White House."]  # List of reference summaries as strings
    generated_summaries = ["anticipating that the trump administration might not be inclined to mount a vigorous fight against the house republicans given the dim view of the health care law a team of lawyers this month sought to intervene in the case on behalf of two participants in the health care program.the incoming trump administration could choose to no longer defend the executive branch against the suit which challenges the administration authority to spend billions of dollars on health insurance subsidies for and americans handing house republicans a big victory on issues. in a potentially decision in 2015 judge rosemary m collyer ruled that house republicans had the standing to sue the executive branch over a spending dispute and that the obama administration had been distributing the health insurance subsidies in violation of the constitution without approval from congress.in their request the lawyers predicted that a deal between house republicans and the new administration to dismiss or settle the case will produce devastating consequences for the individuals who receive these reductions as well as for the nation health insurance and health care systems generally.just as important to house republicans judge collyer found that congress had the standing to sue the white house on this issue a ruling that many legal experts said was flawed and they want that precedent to be set to restore congressional leverage over the executive branch.but on spending power and standing the trump administration may come under pressure from advocates of presidential authority to fight the house no matter their shared views on health care since those precedents could have broad repercussions"]  # List of generated summaries as strings

    # Replace special character '—' (U+2014) with a regular hyphen '-'
    reference_summaries = [summary.replace('—', '-') for summary in reference_summaries]

    # Initialize the Rouge scorer
    rouge_scorer = Rouge()

    # Calculate ROUGE scores
    scores = rouge_scorer.get_scores(generated_summaries, reference_summaries, avg=True)

    # Print the ROUGE scores
    print("ROUGE-1: {:.2f}".format(scores['rouge-1']['f']*100))
    print("ROUGE-2: {:.2f}".format(scores['rouge-2']['f']*100))
    print("ROUGE-L: {:.2f}".format(scores['rouge-l']['f']*100))


# Prompt the user for input
user_input = input("Enter 'program_one' to execute program one, 'program_two' to execute program two: ")

# Execute the corresponding program based on user input
switch = {
    'program_one': program_one,
    'program_two': program_two
}

selected_program = switch.get(user_input)
if selected_program:
    selected_program()
else:
    print("Invalid input.")
