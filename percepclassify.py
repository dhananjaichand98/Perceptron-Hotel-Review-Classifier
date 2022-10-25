import sys
import json
import numpy as np
import re
from collections import defaultdict

TRUTH_REVERSE_ENCODING = {1: "Fake", -1: "True"}
SENTIMENT_REVERSE_ENCODING = {1: "Pos", -1: "Neg"}

class PerceptronClassifier():
    """
        This class classifies input data using pre-trained weights

        Attributes:
            b: bias value
            w: 1-D list of weights            
    """

    def __init__(self):
        """
            Inits PerceptronClassifier
        """
        self.b = 0
        self.w = []

    def load_weights(self, w, b):
        """
            Load trained weights for perceptorn model
        """
        self.w = np.asarray(w)
        self.b = np.asarray(b)

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
    
def save_output(file, id_column, truth_pred, truth_reverse_encoding, sentiment_pred, sentiment_reverse_encoding):
    """
        Save predicted output to output file
    """
    with open(file, 'w', encoding='utf-8') as file:
        for i in range(len(id_column)):
            output_line = [id_column[i], truth_reverse_encoding[truth_pred[i]], sentiment_reverse_encoding[sentiment_pred[i]]]
            file.write(' '.join(output_line))
            file.write('\n')

def process_text(text):
    """
        Process text for feeding to perceptron
    """
    text = text.lower()
    text = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(text))
    text = re.sub('[^a-zA-Z ]+', '', text)
    text = text.strip()
    text = re.sub(' +', ' ', text)
    split_text = text.split(' ')
    return split_text

def load_model(model_file):
    """
        Load model related data saved in JSON format
    """
    with open(model_file, 'r', encoding='utf-8', errors='ignore') as f:
        json_dict = json.loads(f.read())
        word_index = json_dict["word_index"] 
        idf = json_dict["idf"]
    return json_dict, word_index, idf

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
            # ignore words not seen while fitting tfidf
            if word in word_index:
                tf = curr_text_word_dict[word]/n
                tf_idf[i][word_index[word]] = tf * idf[word]
    
    return tf_idf

def main():

    try:
        model_file = sys.argv[1]
    except:
        model_file = "vanillamodel.txt"

    model_dict, word_index, idf = load_model(model_file)

    try:
        input_file = sys.argv[2]
    except:
        input_file = "dev-text.txt"
    
    test_lines, id_column = [], []
    with open(input_file, 'r') as file:
        for line in file:
            id, text = line.split(' ', 1)
            test_lines.append(process_text(text))
            id_column.append(id)

    tf_idf_vectorize_transformd_test = tf_idf_vectorize_transform(test_lines, word_index, idf)

    truth_perceptron = PerceptronClassifier()
    truth_perceptron.load_weights(model_dict["w_truth"], model_dict["b_truth"])
    y_pred_truth = truth_perceptron.predict(tf_idf_vectorize_transformd_test)

    sentiment_perceptron = PerceptronClassifier()
    sentiment_perceptron.load_weights(model_dict["w_sentiment"], model_dict["b_sentiment"])
    y_pred_sentiment = sentiment_perceptron.predict(tf_idf_vectorize_transformd_test)

    output_file = "percepoutput.txt"
    save_output(output_file, id_column, y_pred_truth, TRUTH_REVERSE_ENCODING, y_pred_sentiment, SENTIMENT_REVERSE_ENCODING)

if __name__ == "__main__":
    main()