
# wink-naive-bayes-text-classifier

Naive Bayes Text Classifier

### [![Build Status](https://app.travis-ci.com/winkjs/wink-naive-bayes-text-classifier.svg?branch=master)](https://app.travis-ci.com/winkjs/wink-naive-bayes-text-classifier) [![Coverage Status](https://coveralls.io/repos/github/winkjs/wink-naive-bayes-text-classifier/badge.svg?branch=master)](https://coveralls.io/github/winkjs/wink-naive-bayes-text-classifier?branch=master) [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/winkjs/Lobby)

<img align="right" src="https://decisively.github.io/wink-logos/logo-title.png" width="100px" >

Classify text, analyse sentiments, recognize user intents for chatbot using **`wink-naive-bayes-text-classifier`**. Its [API](http://winkjs.org/wink-naive-bayes-text-classifier/NaiveBayesTextClassifier.html) offers a rich set of features including cross validation to compute confusion matrix, precision, and recall. It delivers impressive accuracy levels with right text pre-processing using [wink-nlp](https://www.npmjs.com/package/wink-nlp):

| Dataset | Accuracy |
| --- | --- |
| **Sentiment Analysis**<br/>Amazon Product Review [Sentiment Labelled Sentences Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/00331/) at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) | **90%** <br/> 800 training examples & 200 validation reviews<br/><br/>Refer to `sentiment-analysis-example` directory for the reference code |
| **Intent Classification**<br/>Chatbot corpus from [NLU Evaluation Corpora](https://github.com/sebischair/NLU-Evaluation-Corpora) as mentioned in paper titled [Evaluating Natural Language Understanding Services for Conversational Question Answering Systems](https://aclanthology.org/W17-5522.pdf) | **99%** <br/>100 training examples & 106 validation<br/><br/>Refer to `chatbot-example` directory for the reference code |

#### Text Pre-processing
A [winkNLP](https://github.com/winkjs/wink-nlp) based helper function for general purpose text pre-processing is available that (a) tokenizes, (b) removes punctuations, symbols, numerals, URLs, stop words, (c) stems each token and (d) handles negations. It can be required from `wink-naive-bayes-text-classifier/src/prep-text.js`. WinkNLP's [Named Entity Recognition](https://winkjs.org/wink-nlp/getting-started.html) may be used to further enhance the pre-processing.

#### Hyperparameters
These include smoothing factor to control [additive smoothing](https://winkjs.org/wink-naive-bayes-text-classifier/NaiveBayesTextClassifier.html#defineConfig) and a [consider presence only flag](https://winkjs.org/wink-naive-bayes-text-classifier/NaiveBayesTextClassifier.html#defineConfig) to choose from Multinomial/Binarized naive bayes.

The trained model can be [exported](https://winkjs.org/wink-naive-bayes-text-classifier/NaiveBayesTextClassifier.html#exportJSON) as JSON and can be [reloaded](https://winkjs.org/wink-naive-bayes-text-classifier/NaiveBayesTextClassifier.html#importJSON) later for predictions.


### Installation
Use [npm](https://www.npmjs.com/package/wink-naive-bayes-text-classifier) to install:
```
npm install wink-naive-bayes-text-classifier --save
```

> It requires Node.js version **16.x** or **18.x**.

### Example
```javascript

// Load Naive Bayes Text Classifier
var Classifier = require( 'wink-naive-bayes-text-classifier' );
// Instantiate
var nbc = Classifier();
// Load wink nlp and its model
const winkNLP = require( 'wink-nlp' );
// Load language model
const model = require( 'wink-eng-lite-web-model' );
const nlp = winkNLP( model );
const its = nlp.its;

const prepTask = function ( text ) {
  const tokens = [];
  nlp.readDoc(text)
      .tokens()
      // Use only words ignoring punctuations etc and from them remove stop words
      .filter( (t) => ( t.out(its.type) === 'word' && !t.out(its.stopWordFlag) ) )
      // Handle negation and extract stem of the word
      .each( (t) => tokens.push( (t.out(its.negationFlag)) ? '!' + t.out(its.stem) : t.out(its.stem) ) );

  return tokens;
};
nbc.definePrepTasks( [ prepTask ] );
// Configure behavior
nbc.defineConfig( { considerOnlyPresence: true, smoothingFactor: 0.5 } );
// Train!
nbc.learn( 'I want to prepay my loan', 'prepay' );
nbc.learn( 'I want to close my loan', 'prepay' );
nbc.learn( 'I want to foreclose my loan', 'prepay' );
nbc.learn( 'I would like to pay the loan balance', 'prepay' );

nbc.learn( 'I would like to borrow money to buy a vehicle', 'autoloan' );
nbc.learn( 'I need loan for car', 'autoloan' );
nbc.learn( 'I need loan for a new vehicle', 'autoloan' );
nbc.learn( 'I need loan for a new mobike', 'autoloan' );
nbc.learn( 'I need money for a new car', 'autoloan' );
// Consolidate all the training!!
nbc.consolidate();
// Start predicting...
console.log( nbc.predict( 'I would like to borrow 50000 to buy a new Audi R8 in New York' ) );
// -> autoloan
console.log( nbc.predict( 'I want to pay my car loan early' ) );
// -> prepay
```

Try [experimenting with this example on Runkit](https://npm.runkit.com/wink-naive-bayes-text-classifier) in the browser.

### Documentation
Check out the [Naive Bayes Text Classifier](http://winkjs.org/wink-naive-bayes-text-classifier/) API documentation to learn more.

### Need Help?
If you spot a bug and the same has not yet been reported, raise a new [issue](https://github.com/winkjs/wink-naive-bayes-text-classifier/issues) or consider fixing it and sending a pull request.

### About wink
[Wink](http://winkjs.org/) is a family of open source packages for **Natural Language Processing**, **Statistical Analysis** and **Machine Learning** in NodeJS. The code is **thoroughly documented** for easy human comprehension and has a **test coverage of ~100%** for reliability to build production grade solutions.


### Copyright & License
**wink-naive-bayes-text-classifier** is copyright 2017-22 [GRAYPE Systems Private Limited](http://graype.in/).

It is licensed under the terms of the MIT License.
