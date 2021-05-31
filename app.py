from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from fetch_data import Fetchdata

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

#Loading all the relevant files at the time of app startup
df_main = Fetchdata().getOrignalData()
df_processed = Fetchdata().getTextProcessedData()
df_mapped = Fetchdata().getMappedData()
df_final_rating = Fetchdata().getFinalUserRatingData()
df_eda = Fetchdata().getEDACleanedData()
df_sentiment_stats = Fetchdata().getSentimentStats()

'''
    @params userIds - top 20 userIds, df_sentiments - dataframe with text processing
    returns top 5 positive percentage ids
'''
def checkProductSentiment(userIds, df_sentiments):
    product_percentage = {}

    for id in userIds:
        filteredProduct = df_sentiments[df_sentiments['id']==id]
        percentPositive = filteredProduct['Prediction'].sum()/len(filteredProduct)
        product_percentage[id]=percentPositive
        productPercentAsc = sorted(product_percentage.items(), key=lambda x: x[1])
        finalprodList = [i [0] for i in productPercentAsc[::-1][:5]]
    return finalprodList

'''
    @params username - query search string, map_df - dataframe for mapping pdt details with user
    returns data to index.html to be displayed on webpage
'''
def getRecommendedProduct(username, map_df):
    df_final = pd.DataFrame()
    df_final_5 = pd.DataFrame()
    isError = None
    userInfo = None

    if username not in df_final_rating.index:
        isError = 'Data Not Available'
    else:
        df_final = df_final_rating.loc[username].sort_values(ascending=False)[0:20]
        df_final = pd.concat({"id": pd.Series(list(df_final.index)),
                            "probScore": pd.Series(list(df_final.values))},axis=1)
        df_final = pd.merge(df_final, map_df, left_on='id', right_on='id', how = 'left')
        userIds = list(df_final['id'])
        final_5 = checkProductSentiment(userIds, df_sentiment_stats)
        df_final_5= df_final[df_final['id'].isin(final_5)]
        userInfo = username

    return render_template('index.html', username=userInfo, data=[df_final_5.to_html(classes='prediction')], error=isError, users=None, titles=[])

'''
   Route for handling default page. 
'''
@app.route('/')
@app.route('/index')
def index():
    all_users = np.random.choice(df_eda.reviews_username.unique(), size=5)
    return render_template('index.html', username=None, error=None, users=all_users)

'''
   Route for submit request on the form. 
'''
@app.route('/topProducts', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['name']
        return getRecommendedProduct(user, df_mapped)

# # Auto-suggest which auto-populates the names of the user
# @app.route('/search/names',methods=['GET'])
# def process():
#     query = request.args.get('query')
#     suggestions = list(products[(~products.userId.isnull()) & (products.userId.str.startswith(query))]['userId'])
#     suggestions = [{'value':suggestion,'data':suggestion} for suggestion in suggestions]
#     return jsonify({"suggestions":suggestions[:5]})

if __name__ == "__main__":
  app.run()
