
# wink-naive-bayes-text-classifier

Configurable [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) Classifier for text with cross-validation support

### [![Build Status](https://app.travis-ci.com/winkjs/wink-naive-bayes-text-classifier.svg?branch=master)](https://app.travis-ci.com/winkjs/wink-naive-bayes-text-classifier) [![Coverage Status](https://coveralls.io/repos/github/winkjs/wink-naive-bayes-text-classifier/badge.svg?branch=master)](https://coveralls.io/github/winkjs/wink-naive-bayes-text-classifier?branch=master) [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/winkjs/Lobby)

<img align="right" src="https://decisively.github.io/wink-logos/logo-title.png" width="100px" >

Classify text, analyse sentiments, recognize user intents for chatbot using **`wink-naive-bayes-text-classifier`**. Its [API](http://winkjs.org/wink-naive-bayes-text-classifier/NaiveBayesTextClassifier.html) offers a rich set of features:

1. Preprocess text using [wink-nlp](https://www.npmjs.com/package/wink-nlp) — tokenize, stem, remove stop words, and handle negation. It also supports [Named Entity Recognition](https://winkjs.org/wink-nlp/getting-started.html) to further enhance preprocessing. A single winkNLP based helper function for preparing text is available that (a) tokenizes, (b) removes punctuations, symbols, numerals, URLs, stop words and (c) stems. It can be required from `wink-naive-bayes-text-classifier/src/prep-text.js`.
2. Configure **Lidstone** or **Laplace** additive smoothing.
3. Configure **Multinomial** or **Binarized Multinomial** Naive Bayes model.
4. Export and import learnings in JSON format that can be easily saved on hard-disk.
5. Evaluate learning to perform n-fold cross validation.
6. Obtain comprehensive metrics including **confusion matrix**, **precision**, and **recall**.

### Installation
Use [npm](https://www.npmjs.com/package/wink-naive-bayes-text-classifier) to install:
```
npm install wink-naive-bayes-text-classifier --save
```


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
