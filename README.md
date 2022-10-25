<a name="readme-top"></a>

# Perceptron - Hotel Review Classifier

## Description

A hotel review classifier perceptron model written from scratch on Python that classifies hotel reviews based on the nature of review (TRUE/FAKE) and sentiment (POSITIVE/NEGATIVE). It uses input text transformed into TF-IDF vectors. The model has two variations, viz, vanilla and average. The vanilla model is the default perceptron that works by iteratively modifying weights and bias whereas the average model takes an average of weights trained over the epochs.

The model attains macro-average test F1 Scores of 83.12% for TRUE/FAKE classification with vanilla model and 84.69 for TRUE/FAKE classification with average model. Additionally, It achieves F1 Scores of 91.87% for POSITIVE/NEGATIVE classification with vanilla model and 91.24% for POSITIVE/NEGATIVE classification with average model.

## Getting Started

### Dependencies

* Python3
* NumPy

### Installing

Use the following steps for installation.

1. Clone the repo
   ```sh
   git clone https://github.com/dhananjaichand98/Perceptron-Hotel-Review-Classifier.git
   ```
3. Install required Python packages
   ```sh
   pip3 install -r requirements.txt
   ```

### Executing program

There are two programs: perceplearn.py will learn perceptron models (vanilla and averaged) from the training data, and percepclassify.py will use the models to classify new data.

* The learning program will be invoked in the following way; The argument is a single file contatining the training data; the program will learn perceptron models, and write the model parameters to two files: ```vanillamodel.txt``` for the vanilla perceptron and ```averagedmodel.txt``` for the averaged perceptron.
    ```
    python perceplearn.py /path/to/input
    ```
* The classification program will be invoked in the following way; The first argument is the path to the model file (vanillamodel.txt or averagedmodel.txt), and the second argument is the path to a file containing the test data; the program will read the parameters of a perceptron model from the model file, classify each entry in the test data, and write the results to a text file called ```percepoutput.txt``` in the same format as the answer key specified below.
    ```
    python percepclassify.py /path/to/model /path/to/input
    ```

Alternatively, 
* execute bash script to run both the files and get classification result.
    ```
    bash bash.sh /path/to/training_input /path/to/saved_model /path/to/raw_input
    ```

## Data

The data folder contains train-labeled.txt, dev-key.txt and dev-text.txt files in the following format:

* train-labeled.txt: File containing labeled training data with a single training instance (hotel review) per line (total 960 lines). The tokens in each line are:
    * a unique 7-character alphanumeric identifier
    * a label True or Fake
    * a label Pos or Neg
    * review text
* dev-text.txt : File with unlabeled development data, containing just the unique identifier followed by the text of the review (total 320 lines).
* dev-key.txt: File dev-key.txt with the corresponding labels for the development data, to serve as an answer key.

## Authors

- [Dhananjai Chand](https://www.linkedin.com/in/dhananjai-chand/)

## Acknowledgment

* [Hal Daumé III, A Course in Machine Learning (v. 0.99 draft), Chapter 4: The Perceptron](http://www.ciml.info/dl/v0_99/ciml-v0_99-ch04.pdf)

<p align="right">(<a href="#readme-top">back to top</a>)</p>