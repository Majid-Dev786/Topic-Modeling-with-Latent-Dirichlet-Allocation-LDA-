# Importing necessary libraries and modules
from gensim import corpora, models
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk

# Downloading NLTK stopwords
nltk.download('stopwords')

# Creating a set of English stopwords
stop_words = stopwords.words('english')

class LDATopicModel:
    # Constructor for LDATopicModel class
    def __init__(self, documents):
        # Initializing LDATopicModel object with provided documents
        self.documents = documents
        self.dictionary = None
        self.corpus = None
        self.lda_model = None

    # Preprocess input documents by tokenizing and removing stopwords
    def preprocess_documents(self):
        processed_docs = [[word for word in simple_preprocess(doc) if word not in stop_words] 
                          for doc in self.documents]
        
        self.dictionary = corpora.Dictionary(processed_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

    # Build and train the LDA topic model
    def build_and_train_model(self, num_topics=2, passes=15):
        self.lda_model = models.LdaModel(self.corpus, num_topics=num_topics,
                                         id2word=self.dictionary,
                                         passes=passes)

    # Display the top words for each topic in the trained model
    def display_topics(self, num_words=5):
        topics = self.lda_model.print_topics(num_words=num_words)
        for topic in topics:
            print(topic)

# Main function to demonstrate the usage of LDATopicModel
def main():
    documents = [
        "Machine learning is a subfield of artificial intelligence.",
        "It involves the study of algorithms that can improve automatically.",
        "Deep learning is a popular branch of machine learning.",
        "Natural language processing is used in various applications.",
        "Topic modeling is a technique to discover hidden topics in documents."
    ]

    # Creating an instance of LDATopicModel
    lda_topic_model = LDATopicModel(documents)
    lda_topic_model.preprocess_documents()
    lda_topic_model.build_and_train_model(num_topics=2, passes=15)
    lda_topic_model.display_topics(num_words=5)

# Execute main function if the script is run directly
if __name__ == "__main__":
    main()