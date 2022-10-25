from collections import defaultdict
import json
import sys
import re
import numpy as np

TRUTH_ENCODING = {"Fake" : 1, "True": -1}
SENTIMENT_ENCODING = {"Pos": 1, "Neg": -1}

class PerceptronTrain:
    """
        This class trains perceptron model

        Attributes:
            b: bias value
            w: 1-D list of weights            
    """

    def __init__(self):
        """
            Inits PerceptronTrain
        """
        self.b = 0
        self.w = []

    def train_vanilla(self, X, y, epochs):
        """
            train a vanilla perceptron
        """
        # initializing the weights and biases
        self.w = [0] * len(X[0])
        self.b = 0

        shuffled_indexes = list(range(len(X)))
        
        for _ in range(epochs):
            for i in range(len(X)):
                xi = X[shuffled_indexes[i]]
                yi = y[shuffled_indexes[i]]

                a = np.multiply(xi, self.w).sum() + self.b

                if yi*a <= 0:
                    # update weights on error
                    self.w = self.w + np.multiply(xi, yi)*0.01
                    self.b = self.b + yi*0.01
            
            # y_pred = self.predict(X)
            # print(self.accuracy(y_pred, y))

        return self.w, self.b

    def train_average(self, X, y, epochs):
        """
            train an average perceptron
        """
        # initializing the weights and biases
        self.w = [0] * len(X[0])
        self.b = 0
        u = [0] * len(X[0])
        beta = 0
        c = 1

        shuffled_indexes = list(range(len(X)))
        
        for _ in range(epochs):
            for i in range(len(X)):
                xi = X[shuffled_indexes[i]]
                yi = y[shuffled_indexes[i]]

                a = np.multiply(xi, self.w).sum() + self.b

                if yi*a <= 0:
                    # update weights on error
                    self.w = self.w + np.multiply(xi, yi)*0.01
                    self.b = self.b + yi*0.01
                    u = u + yi*xi*c*0.01
                    beta = beta + yi*c*0.01
                
                c += 1
            
            y_pred = self.predict(X)
            print("average accuracy:", self.accuracy(y_pred, y))

        u_by_c = np.multiply(u, 1/c)
        beta_by_c = np.multiply(beta, 1/c)

        self.w = np.subtract(self.w, u_by_c)
        self.b = np.subtract(self.b, beta_by_c)

        return self.w, self.b
    
    def accuracy(self, y_pred, y_test):
        """
            Calculate accuracy for predicted output
        """
        correct = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                correct += 1
        acc = correct/len(y_pred)*100
        return acc

    def predict(self, X):
        """
            Predict output label for input data
        """
        y_pred = []
        for i in range(len(X)):
            xi = X[i]
            a = np.multiply(xi, self.w).sum() + self.b
            y_pred.append(1 if a > 0 else -1)
        return y_pred

def tf_idf_vectorize_fit_transform(lines, word_set, num_docs):
    """
        Fit and Generate TF-IDF vectors for set of input text documents
    """
    # getting index for each word
    index = dict()
    for i, word in enumerate(word_set):
        index[word] = i
    
    sentence_word_count = defaultdict(lambda: 0)

    tf_idf = np.zeros(shape=(num_docs, len(word_set)))

    for i, line in enumerate(lines):
        text = line[0]
        curr_text_word_dict = defaultdict(lambda: 0)
        for word in text:
            if not word in word_set: continue
            curr_text_word_dict[word] += 1
        n = len(text)
        for word in curr_text_word_dict.keys():
            # adding count for occurance in sentence
            sentence_word_count[word] += 1
            # finding the document frequency
            tf_idf[i][index[word]] = curr_text_word_dict[word]/n

    idf = {}
    for word in word_set:
        idf[word] = np.log(num_docs/(sentence_word_count[word] + 1)) # doing smoothing

    for i, line in enumerate(lines):
        text = line[0]
        curr_sentence_word_set  = set(text)
        for word in curr_sentence_word_set:
            if not word in word_set: continue
            tf_idf[i][index[word]] = tf_idf[i][index[word]] * idf[word]

    return tf_idf, index, idf


def tf_idf_vectorize_transform(lines, word_index, idf):
    """
        Tranform input in text format to TF-IDF Vectors
    """
    tf_idf = np.zeros(shape=(len(lines), len(word_index)))
    
    for i, line in enumerate(lines):
        curr_text_word_dict = defaultdict(lambda: 0)
        for word in line:
            curr_text_word_dict[word] += 1
        n = len(line)
        for word in line:
            if word in word_index:
                tf = curr_text_word_dict[word]/n
                tf_idf[i][word_index[word]] = tf * idf[word]
    return tf_idf


def process_text(text, word_set, word_freq):
    """
        Process text for feeding to perceptron
    """
    text = text.lower()
    text = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(text))
    text = re.sub('[^a-zA-Z ]+', '', text)
    text = text.strip()
    text = re.sub(' +', ' ', text)
    split_text = text.split(' ')
    
    for word in split_text:
        word_set.add(word)
        word_freq[word] += 1

    return split_text

def parse_input(lines):
    """
        Parse input text line by line
    """
    parsed_lines = []
    word_set = set()
    word_freq = defaultdict(lambda: 0)
    for line in lines:
        try:
            _, truth_label, sentiment_label, text = line.split(' ', 3)
        except Exception as e:
            print("error on line : ",  line, e)
        text = process_text(text, word_set, word_freq)                        
        parsed_lines.append([text, truth_label, sentiment_label])

    # removing top k works (stop words)
    top_k = [x[0] for x in sorted(word_freq.items(), key = lambda x: x[1], reverse=True)]
    k = 25
    top_k = top_k[:k]
    word_set = word_set - set(top_k)

    return parsed_lines, word_set, word_freq

def store_weights(filename, word_index, idf, w_truth, b_truth, w_sentiment, b_sentiment):
    """
        Store weights in JSON format to a file
    """
    json_dict = {}
    json_dict["word_index"] = word_index
    json_dict["idf"] = idf
    json_dict["w_truth"] = w_truth.tolist()
    json_dict["b_truth"] = b_truth
    json_dict["w_sentiment"] = w_sentiment.tolist()
    json_dict["b_sentiment"] = b_sentiment
    with open(filename, 'w') as fp:
        json.dump(json_dict, fp, indent=2, ensure_ascii=False)

def encode(arr, encoding_map):
    """
        Encode labels based on encoding map
    """
    return [encoding_map[x] for x in arr]

def main():

    try:
        input_file = sys.argv[1]
    except:
        input_file = "train-labeled.txt"

    lines = []
    with open(input_file, 'r') as file:
        for line in file:
            lines.append(line)

    parsed_lines, word_set, _ = parse_input(lines)
    tf_idf_vectorize_transformd_lines, word_index, idf = tf_idf_vectorize_fit_transform(parsed_lines, word_set, len(parsed_lines))
    
    truth_column = encode([row[1] for row in parsed_lines], TRUTH_ENCODING)
    sentiment_column = encode([row[2] for row in parsed_lines], SENTIMENT_ENCODING)

    perceptron = PerceptronTrain()

    # VANILLA
    w_vanilla_truth, b_vanilla_truth = perceptron.train_vanilla(tf_idf_vectorize_transformd_lines, truth_column, 10)
    y_pred = perceptron.predict(tf_idf_vectorize_transformd_lines)
    y_test = truth_column
    print("Training Accuracy for TRUE/FAKE detection (VANILLA) - ", perceptron.accuracy(y_pred, y_test))
    
    w_vanilla_sentiment, b_vanilla_sentiment = perceptron.train_vanilla(tf_idf_vectorize_transformd_lines, sentiment_column, 10)
    y_pred = perceptron.predict(tf_idf_vectorize_transformd_lines)
    y_test = sentiment_column
    print("Training Accuracy for SENTIMENT classification (VANILLA) - ", perceptron.accuracy(y_pred, y_test))
    
    store_weights("vanillamodel.txt", word_index, idf, w_vanilla_truth, b_vanilla_truth, w_vanilla_sentiment, b_vanilla_sentiment )    

    # AVERAGE
    w_average_truth, b_average_truth = perceptron.train_average(tf_idf_vectorize_transformd_lines, truth_column, 15)
    y_pred = perceptron.predict(tf_idf_vectorize_transformd_lines)
    y_test = truth_column
    print("Training Accuracy for TRUE/FAKE detection (AVERAGE) - ", perceptron.accuracy(y_pred, y_test))

    w_average_sentiment, b_average_sentiment = perceptron.train_average(tf_idf_vectorize_transformd_lines, sentiment_column, 15)
    y_pred = perceptron.predict(tf_idf_vectorize_transformd_lines)
    y_test = sentiment_column
    print("Training Accuracy for SENTIMENT classification (AVERAGE) - ", perceptron.accuracy(y_pred, y_test))
    
    store_weights("averagedmodel.txt", word_index, idf, w_average_truth, b_average_truth, w_average_sentiment, b_average_sentiment )    

    # dev test
    # test_lines = []
    # with open("dev-text.txt", 'r') as file:
    #     for line in file:
    #         id, text = line.split(' ', 1)
    #         test_lines.append(process_text(text, set(), defaultdict(lambda: 0)))
    # tf_idf_vectorize_transformd_test = tf_idf_vectorize_transform(test_lines, word_index, idf)
    # test_truth = []
    # test_sentiment = []
    # with open("dev-key.txt", 'r', encoding="utf-8") as file:
    #     for line in file:
    #         id, truth, sentiment = line.split()
    #         # print(truth, sentiment)
    #         test_truth.append(truth)
    #         test_sentiment.append(sentiment)
    # test_truth = [TRUTH_ENCODING[x] for x in test_truth]
    # test_sentiment = [SENTIMENT_ENCODING[x] for x in test_sentiment]
    # y_pred = perceptron.predict(tf_idf_vectorize_transformd_test)
    # perceptron.accuracy(y_pred, test_sentiment)
    # from sklearn.metrics import classification_report
    # report = classification_report(test_sentiment, y_pred, digits=4)
    # print(report)

if __name__ == "__main__":
    main()