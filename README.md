# Multinomial naive Bayes Spam massages Identifier
Identifying and distinguishing spam massages using the multinomial Naïve Bayes model. 
## what is Naive Bayes classifier
In statistics, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features. 
At the time of writing this repository, there are 5 different types of Naive Bayes classifiers, which as follow:

> - 1- Bernoulli Naive Bayes classifier
> - 2- Categorical Naive Bayes classifier
> - 3- Complement Naive Bayes classifier
> - 4- Gaussian Naive Bayes classifier
> - 5- multinomial Naive Bayes classifier

In this repository, we have used the multinomial Naive Bayes classifier to detect spam messages, the reason for using this classifier is the simple implementation, high accuracy, and vector implementation method of this model. It should be noted that other methods can also be used to detect spam messages, such as the Complement Naive Bayes classifier and Tf-Idf.

## Let's learn more about the Multinomial naive Bayes classifier
MultinomialNB implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification (where the data are typically represented as word vector counts, although tf-idf vectors are also known to work well in practice).
The distribution is parametrized by vectors <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>&#x3B8;</mi>
    <mi>y</mi>
  </msub>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>y</mi>
      <mn>1</mn>
    </mrow>
  </msub>
  <mo>,</mo>
  <mo>&#x2026;</mo>
  <mo>,</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>y</mi>
      <mi>n</mi>
    </mrow>
  </msub>
  <mo stretchy="false">)</mo>
</math>
for each class <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>y</mi>
</math> where <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>n</mi>
</math>  is the number of features (in text classification, the size of the vocabulary) and <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>&#x3B8;</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>y</mi>
      <mi>i</mi>
    </mrow>
  </msub>
</math> is the probability  <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>P</mi>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>x</mi>
    <mi>i</mi>
  </msub>
  <mo>&#x2223;</mo>
  <mi>y</mi>
  <mo stretchy="false">)</mo>
</math>  of feature  <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>i</mi>
</math> appearing in a sample belonging to class <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>y</mi>
</math> 

The parameters <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>&#x3B8;</mi>
    <mi>y</mi>
  </msub>
</math>  is estimated by a smoothed version of maximum likelihood, i.e. relative frequency counting: 

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>&#x3B8;</mi>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>y</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <msub>
        <mi>N</mi>
        <mrow data-mjx-texclass="ORD">
          <mi>y</mi>
          <mi>i</mi>
        </mrow>
      </msub>
      <mo>+</mo>
      <mi>&#x3B1;</mi> /
    </mrow>
    <mrow>
      <msub>
        <mi>N</mi>
        <mi>y</mi>
      </msub>
      <mo>+</mo>
      <mi>&#x3B1;</mi>
      <mi>n</mi>
    </mrow>
  </mfrac>
</math>

where <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>N</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>y</mi>
      <mi>i</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <munder>
    <mo data-mjx-texclass="OP">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>x</mi>
      <mo>&#x2208;</mo>
      <mi>T</mi>
    </mrow>
  </munder>
  <msub>
    <mi>x</mi>
    <mi>i</mi>
  </msub>
</math> is the number of times feature  <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>i</mi>
</math> appears in a sample of class  <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>y</mi>
</math> in the training set <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>T</mi>
</math> and  <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>N</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>y</mi>
    </mrow>
  </msub>
  <mo>=</mo>
  <munderover>
    <mo data-mjx-texclass="OP">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>n</mi>
    </mrow>
  </munderover>
  <msub>
    <mi>N</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>y</mi>
      <mi>i</mi>
    </mrow>
  </msub>
</math> is the total count of all features for class  <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>y</mi>
</math> 
## Used database
I used the smsSpamCollection dataset to train my model, which can be accessed via the link below:
 https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

## Reviewing the results of the outputs of our trained model

The accuracy of our Naïve Bayes multinomial model is 99.01345291479821 %
The Precision of our Naïve Bayes multinomial model is 97.88732394366197 %
The Recall of our Naïve Bayes multinomial model is 94.5578231292517 %

We can use the confusion matrix to observe the performance of our model:

![download](https://user-images.githubusercontent.com/53332753/202918749-a4700297-d395-4b0e-99ca-060a270b4e69.png)


### Steps
- [x] Import libraries
- [x] Upload dataset
- [x] Create the data frame
- [x] Split the data
- [x] Vectorize the data
- [x] Train & predict
- [x] calculate accuracy, precision, and recall
- [x] calculate the confusion matrix
- [x] Test the model with a new Sms/Email massage

More information is available in the Jupyter Notebook file
