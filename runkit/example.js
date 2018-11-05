// Load Naive Bayes Text Classifier
var Classifier = require( 'wink-naive-bayes-text-classifier' );
// Instantiate
var nbc = Classifier();
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
