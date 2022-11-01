/* eslint-disable no-sync */
const data = require( './chatbot-training-data.json' );
const nbc = require( '../src/wink-naive-bayes-text-classifier' );
const prepTask = require( './prep-task.js' );
let count = 0;
let validationExamples = 0;
let trainingExamples = 0;

const c = nbc();
c.defineConfig( { considerOnlyPresence: false, smoothingFactor: 0.1 } );
c.definePrepTasks( [ prepTask ] );


for ( let i = 0; i < data.sentences.length; i += 1 ) {
  if ( data.sentences[ i ].training ) {
    c.learn( data.sentences[ i ].text, data.sentences[ i ].intent );
    trainingExamples += 1;
  }
}

console.log( `\nCompleted training using ${trainingExamples} examples labeled as 'training'.` );

c.consolidate();

for ( let i = 0; i < data.sentences.length; i += 1 ) {
  if ( !data.sentences[ i ].training ) {
    validationExamples += 1;
    const p = c.predict( data.sentences[ i ].text, data.sentences[ i ].intent );
    if ( p === data.sentences[ i ].intent ) {
      count += 1;
    }
  }
}

console.log( `\nCompleted intent classification using ${validationExamples} intents meant for validation.`)

console.log(  `\tAccuracy = ${+( ( count / validationExamples ) * 100 ).toFixed( 1 )}%` );
