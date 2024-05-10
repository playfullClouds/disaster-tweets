import dill
from sklearn.feature_extraction.text import TfidfVectorizer




# try:
#     with open("C:\\disaster-tweets\\artifacts\\data_transformation\\tfidf_vectorizer_new.pkl", "rb") as file_obj:
#         vectorizer = dill.load(file_obj)
#         print("Vectorizer loaded successfully")
# except Exception as e:
#     print(f"Error while loading the vectorizer: {e}")




# vectorizer = TfidfVectorizer(
#                 sublinear_tf=True,
#                 min_df=10,
#                 norm='l2',
#                 ngram_range=(1, 2),
#                 stop_words='english'
#             )


# # Save it to a new file
# with open("C:\\disaster-tweets\\artifacts\\data_transformation\\tfidf_vectorizer_new.pkl", "wb") as file_obj:
#     dill.dump(vectorizer, file_obj)