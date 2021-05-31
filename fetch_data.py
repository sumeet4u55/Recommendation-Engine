import pandas as pd

class Fetchdata:
    def __init__(self):
        """Initialize instance of Class."""
        self.originalData = pd.read_csv('sample30.csv')
        self.cleanData = pd.read_pickle('./data/cleaned_text_data',compression='zip')
        self.user_user_rating = pd.read_pickle('./data/user_user_rating',compression='zip')
        self.finalEDA = pd.read_pickle('./data/final_eda',compression='zip')
        self.userMapping  = self.finalEDA.drop_duplicates(subset=['id'],keep='first')[['id','name','categories']]
        self.sentiment_stats = pd.read_pickle('./data/pdt_sentiment_stats',compression='zip')
    
    def getOrignalData(self):
        return self.originalData

    def getTextProcessedData(self):
        return self.cleanData
    
    def getMappedData(self):
        return self.userMapping

    def getFinalUserRatingData(self):
        return self.user_user_rating
    
    def getEDACleanedData(self):
        return self.finalEDA

    def getSentimentStats(self):
        return self.sentiment_stats