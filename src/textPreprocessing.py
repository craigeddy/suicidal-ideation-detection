import pandas as pd
import csv
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import nltk


class PreProcesssingReddit:
    def __init__(self):
        pd.options.display.max_colwidth = 100
        pass

    def normalizeCase(self, filename, outputCSV):
        # mostFreqWordsToRemove = ['realli', 'get', 'realli', 'peopl', 'tri', 'even', 'one', 'see', 'say', 'anyth',]
        process = pd.read_csv(filename)
        print("Original: \n", process["selftext"].head(n=3))

        process['selftext'] = process['selftext'].str.lower()
        print("Lower:\n ",process["selftext"].head(n=3))

        process['selftext'] = process['selftext'].str.replace('[^\w\s]',' ')
        print("Remove Punctuations:\n ", process["selftext"].head(n=3))

        process['selftext'] = process['selftext'].str.replace('\\n', ' ')
        print("Remove New Lines:\n ", process["selftext"].head(n=3))





        stop = stopwords.words('english')
        englishWords = set(words.words())
        process['selftext'] = process['selftext'].apply(
            lambda x: " ".join(x for x in str(x).strip().split() if (x in englishWords or not x.isalpha()) and len(x) >= 3 and x not in stop))
        print("Remove Stopwords:\n ", process["selftext"].head(n=3))

        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        process['selftext'] = process['selftext'].apply(
            lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
        print("Lemmatize:\n ", process["selftext"].head(n=3))
        process['selftext'] = process['selftext'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
        print("Stemmatize:\n ", process["selftext"].head(n=3))

        MostFreq = pd.Series(' '.join(process['selftext']).split()).value_counts()[:3500]
        # LeastFreq = pd.Series(' '.join(process['selftext']).split()).value_counts()#[-16200:]
        # self.frequentWords(LeastFreq)
        print(process.shape)
        print(process["selftext"].head())
        print(MostFreq)
        # print(LeastFreq)
        process['selftext'] = process['selftext'].apply(
            lambda x: " ".join(x for x in str(x).strip().split() if x in MostFreq))
        LeastFreq = pd.Series(' '.join(process['selftext']).split()).value_counts()[-2000:]

        print(process["selftext"].head())
        print(LeastFreq)

        header = ['author', 'subreddit', 'created_utc', 'score', 'selftext']
        process.to_csv(outputCSV, columns=header)
        pass

    def frequentWords(self, freqWords):
        freqWords.to_csv('freqWordsInTS2.csv')


def main():
    p = PreProcesssingReddit()
    p.normalizeCase('generalIssuesTS2.csv', 'cleanedGeneralIssuesTS2.csv')
    # string = "I'm not trying to be dramatic, just.... realistic. I'm 24, I've had 2 suicide attempts so far, and I spent a week in a psych ward earlier this year. My suicidal thoughts are constantly on a loop, always in the background of whatever I'm doing--sometimes I can drown them out with hobbies or work but they never really go away. I try to be gratefhl for all the blessings in my life, eat healthy, reach out to people for help, etc, and I still want to kill myself. " \
    #          "Because I already have two attempts, I'm considered a 'high risk' for another one. (and 3rd time's the charm, am I right guys? finger guns)" \
    #          "I have no sense of the future. I literally cannot imagine growing older. I have absolutely no sense of long-term thinking or goal planning or anything beyond 'get through the next couple days and figure it out later.' Exactly 50% of me is 'don't do X thing, that'll lead to Y bad effect' and the other half is 'who cares about the consequences, you'll be dead by this time next month.'" \
    #          "I DONT WANT TO KILL MYSELF" \
    #          "I REALLY DONT" \
    #          "BUT I AM GOING TO EVENTUALLY AND I DONT KNOW HOW TO STOPI DONT KNOW HOW TO STOP ALL THIS AND I FEEL SO PATHETIC AND OVERWHELMED"
    # print("Orginal: \n",string, "\n")
    # lowerCase = string.lower()
    # print("Lower Case: \n", lowerCase, "\n")
    # removePunctuation = string.replace('[^\w\s]',' ')
    # print("Removed Punctuation: \n", removePunctuation, "\n")
    # removeNewLine = string.replace('\\n', ' ')
    # print("Removed New Line: \n", removeNewLine, "\n")
    #
    # stop = set(stopwords.words('english'))
    # word_tokens = word_tokenize(removeNewLine)
    #
    # removeNonEnglishWords = " ".join()
    # print("Remove stopwords and small unnecessary words: \n", removeNonEnglishWords, "\n")
    #
    # lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()
    # lemmatized = " ".join([lemmatizer.lemmatize(word) for word in removeNonEnglishWords.split()])
    # print("Lemmatized Words: \n", lemmatized, "\n")
    #
    # stemmatized = " ".join([stemmer.stem(word) for word in lemmatized.split()])
    # print("Stemmatized Words: \n", stemmatized, "\n")


    pass

if __name__ == '__main__':
    main()