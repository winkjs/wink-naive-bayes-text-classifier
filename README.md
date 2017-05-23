
# wink-naive-bayes-text-classifier

> Configurable [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) Classifier for text with cross-validation support

### [![Build Status](https://api.travis-ci.org/decisively/wink-naive-bayes-text-classifier.svg?branch=master)](https://travis-ci.org/decisively/wink-naive-bayes-text-classifier) [![Coverage Status](https://coveralls.io/repos/github/decisively/wink-naive-bayes-text-classifier/badge.svg?branch=master)](https://coveralls.io/github/decisively/wink-naive-bayes-text-classifier?branch=master)

<img align="right" src="https://decisively.github.io/wink-logos/logo-title.png" width="100px" >

**wink-naive-bayes-text-classifier** is a part of **[wink](https://www.npmjs.com/~sanjaya)**, which is a family of Machine Learning NPM packages. They consist of simple and/or higher order functions that can be combined with NodeJS `stream` and `child processes` to create recipes for analytics driven business solutions.

Easily classify text, analyse sentiments, recognize intents using **wink-naive-bayes-text-classifier**. It's API offers a rich set of features:

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

## Need Help?
If you spot a bug and the same has not yet been reported, raise a new [issue](https://github.com/decisively/wink-naive-bayes-text-classifier/issues) or consider fixing it and sending a pull request.


## Copyright & License
**wink-naive-bayes-text-classifier** is copyright 2017 GRAYPE Systems Private Limited.

It is licensed under the under the terms of the GNU Affero General Public License as published by the Free
Software Foundation, version 3 of the License.
