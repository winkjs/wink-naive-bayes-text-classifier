const model = require( 'wink-eng-lite-web-model' );
const nlp = require( 'wink-nlp' )(model);
const its = nlp.its;

// POS and types to be preserved, rest is filtered out.
const posSet = new Set( [ 'ADJ', 'ADV', 'VERB', 'PROPN', 'INTJ' ] );
const typeSet = new Set( [ 'number' ] );

// Text preprocessing for sentiment analysis.
module.exports = ( text ) => {
  const tokens = [];
  const doc = nlp.readDoc( text );
  doc.tokens()
     // Preserve only predefined pos & types.
     .filter( ( t ) =>  ( posSet.has( t.out( its.pos ) ) || typeSet.has( t.out( its.type ) ) ) )
     // Handle negation & numbers.
     .each( ( t ) => tokens.push(
       ( t.out( its.type ) === 'number' ) ? '$number' :
         ( t.out( its.negationFlag ) ) ? '!' + t.out(its.lemma) : t.out(its.lemma) )
     );

     // Add bigrams!
  let i, imax;
  for ( i = 0, imax = tokens.length - 1; i < imax; i += 1 ) {
    tokens.push( tokens[ i ] + '_' + tokens[ i + 1 ] );
  }

  // Inject sentiment from winkNLP.
  if ( doc.out( its.sentiment ) > 0.05 ) tokens.push( '$pos' );
  if ( doc.out( its.sentiment ) < -0.05 ) tokens.push( '$neg' );
  return tokens;
}; // prepTask()
