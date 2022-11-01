var trainingData = require( './shuffled-sentiment-training-data.json' );
var trainingExamples = trainingData.slice( 0, 800 );
var testingData = trainingData.slice( 800 );

const nbc = require( '../src/wink-naive-bayes-text-classifier.js' );
const prepTask = require( './prep-task.js' );
var count = 0;
var total = 0;

const c = nbc();
c.defineConfig( { considerOnlyPresence: false, smoothingFactor: 0.1 } );
c.definePrepTasks( [ prepTask ] );

console.log( '\nUsing 800 examples out of 1000 for training the classifier.' );
trainingExamples.forEach( ( d ) => c.learn( d[ 0 ], d[ 1 ] ) );

c.consolidate();

console.log( '\nUsing balance 200 examples out of 1000 for validating the classifier.' );
testingData.forEach( ( d ) => {
  total += 1;
  if ( c.predict( d[ 0 ] ) == d[ 1 ] ) {
    count += 1;
  }
} );

console.log( `\tAccuracy = ${( count * 100 / total ).toFixed( 2 )}%\n` );
