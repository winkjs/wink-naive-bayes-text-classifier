
# wink-naive-bayes-text-classifier

> Configurable [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) Classifier for text with cross-validation support

### [![Build Status](https://api.travis-ci.org/decisively/wink-naive-bayes-text-classifier.svg?branch=master)](https://travis-ci.org/decisively/wink-naive-bayes-text-classifier) [![Coverage Status](https://coveralls.io/repos/github/decisively/wink-naive-bayes-text-classifier/badge.svg?branch=master)](https://coveralls.io/github/decisively/wink-naive-bayes-text-classifier?branch=master) [![Inline docs](http://inch-ci.org/github/decisively/wink-naive-bayes-text-classifier.svg?branch=master)](http://inch-ci.org/github/decisively/wink-naive-bayes-text-classifier) [![dependencies Status](https://david-dm.org/decisively/wink-naive-bayes-text-classifier/status.svg)](https://david-dm.org/decisively/wink-naive-bayes-text-classifier) [![devDependencies Status](https://david-dm.org/decisively/wink-naive-bayes-text-classifier/dev-status.svg)](https://david-dm.org/decisively/wink-naive-bayes-text-classifier?type=dev)

<img align="right" src="https://decisively.github.io/wink-logos/logo-title.png" width="100px" >

**wink-naive-bayes-text-classifier** is a part of **[wink](https://www.npmjs.com/~sanjaya)**, which is a family of Machine Learning NPM packages. They consist of simple and/or higher order functions that can be combined with NodeJS `stream` and `child processes` to create recipes for analytics driven business solutions.

Easily classify text, analyse sentiments, recognize intents using **wink-naive-bayes-text-classifier**. It's [API](#api) offers a rich set of features:

1. Configure text preparation task such as **amplify negation**, **tokenize**, **stem**, **remove stop words**, and **propagate negation** using [wink-nlp-utils](https://www.npmjs.com/package/wink-nlp-utils) or any other package of your choice.
2. Configure **Lidstone** or **Lapalce** additive smoothing.
3. Configure **Multinomial** or **Binarized Multinomial** Naive Bayes model.
4. Export and import learnings in JSON format that can be easily saved on hard-disk.
5. Evaluate learning to perform n-fold cross validation.
6. Obtain comprehensive metrics including **confusion matrix**, **precision**, and **recall**.

## Installation
Use **[npm](https://www.npmjs.com/package/wink-naive-bayes-text-classifier)** to install:
```
npm install wink-naive-bayes-text-classifier --save
```


## Usage
```javascript

// Load Naive Bayes Text Classifier
var nbc = require( 'wink-naive-bayes-text-classifier' )();
// Load NLP utilities
var nlp = require( 'wink-nlp-utils' );
// Configure preparation tasks
nbc.definePrepTasks( [
  // Simple tokenizer
  nlp.string.tokenize0,
  // Common Stop Words Remover
  nlp.tokens.removeWords,
  // Stemmer to obtain base word
  nlp.tokens.stem
] );
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

## API


#### definePrepTasks( tasks )

Defines the text preparation `tasks` to transform raw incoming text into an array of tokens required during `learn()`, `evaluate()` and `predict()` operations. The `tasks` should be an array of functions. The first function in this array must accept a string as input; and the last function must return an array of tokens as JavaScript Strings. Each function must accept one input argument and return a single value. `definePrepTasks` returns the count of `tasks`.

As illustrated in the usage, [wink-nlp-utils](https://www.npmjs.com/package/wink-nlp-utils) offers a rich set of such functions.

#### defineConfig( config )
Defines the configuration from the `config` object. This object must define 2 properties viz. (a) `considerOnlyPresence` and `smoothingFactor`. The `considerOnlyPresence` must be a boolean — true indicates a binarized model; default value is false. The `smoothingFactor` defines the value for additive smoothing; its default value is 0.5. The `defineConfig()` must be called before attempting to learn.

#### learn( input, label )
Simply learns that the `input` belongs to the `label`. If the input is a JavaScript String, then `definePrepTasks()` must be called before learning.


#### consolidate()
Consolidates the learning. It is a prerequisite for `evaluate()` and/or `predict()`.

#### evaluate( input, label )

It is used to evaluate the learning against a test data set. The `input` is used to predict the label, which is compared with the `label` to populate a confusion matrix.

#### metrics()

It computes a detailed metrics consisting of macro-averaged *precision*, *recall* and *f-measure* along with their label-wise values and the *confusion matrix*.

#### predict( input )
Predicts the label for the `input`.

#### exportJSON()
The learning can be exported as JSON text that may be saved in a file.

#### importJSON( json )
An existing JSON learning can be imported for prediction. It is essential to `definePrepTasks()` and `consolidate()` before attempting to predict.

#### stats()
Returns basic stats of learning in terms of count of samples under each label, total words, and the size of vocabulary.

#### reset()
It completely resets the classifier by re-initializing all the learning related variables, except the preparatory tasks. It is useful during cross fold-validation.

## Need Help?
If you spot a bug and the same has not yet been reported, raise a new [issue](https://github.com/decisively/wink-naive-bayes-text-classifier/issues) or consider fixing it and sending a pull request.


## Copyright & License
**wink-naive-bayes-text-classifier** is copyright 2017 GRAYPE Systems Private Limited.

It is licensed under the under the terms of the GNU Affero General Public License as published by the Free
Software Foundation, version 3 of the License.
