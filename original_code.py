import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


stemmer = SnowballStemmer('english')

df_train = pd.read_csv('Contents/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('Contents/test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('Contents/attributes.csv')
df_pro_desc = pd.read_csv('Contents/product_descriptions.csv')

num_train = df_train.shape[0]

def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())


df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

df_train = df_all.iloc[:num_train]

df_train.to_csv('df_all_stemmed.csv')

X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['id','relevance'],axis=1).values, df_train['relevance'].values, test_size=0.20, random_state=42)

id_test = X_test[:, 0]
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(rmse)
