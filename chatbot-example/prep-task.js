const model = require( 'wink-eng-lite-web-model' );
const nlp = require( 'wink-nlp' )( model );
const its = nlp.its;

// POS and types to be preserved, rest is filtered out.
const posSet = new Set( [  'VERB', 'NOUN', 'ADP', 'DET', 'PART' ] );
const typeSet = new Set( [ 'number' ] );

// Text preprocessing for chatbot.
module.exports = ( text ) => {
  const tokens = [];
  const doc = nlp.readDoc( text );
  doc.tokens()
     // Preserve only predefined pos & types.
     .filter( ( t ) =>  ( posSet.has( t.out( its.pos ) ) || typeSet.has( t.out( its.type ) ) ) )
     // Handle numbers.
     .each( ( t ) => tokens.push(
       ( t.out( its.type ) === 'number' ) ? '$number' : t.out( its.lemma ) )
     );

  return tokens;
}; // prepTask()
