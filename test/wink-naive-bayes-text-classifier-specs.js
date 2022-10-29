//     wink-naive-bayes-text-classifier
//     Configurable Naive Bayes Classifier for text
//     with cross-validation support.
//
//     Copyright (C) GRAYPE Systems Private Limited
//
//     This file is part of “wink-naive-bayes-text-classifier”.
//
//     Permission is hereby granted, free of charge, to any person obtaining a
//     copy of this software and associated documentation files (the "Software"),
//     to deal in the Software without restriction, including without limitation
//     the rights to use, copy, modify, merge, publish, distribute, sublicense,
//     and/or sell copies of the Software, and to permit persons to whom the
//     Software is furnished to do so, subject to the following conditions:
//
//     The above copyright notice and this permission notice shall be included
//     in all copies or substantial portions of the Software.
//
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//     THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//     DEALINGS IN THE SOFTWARE.

//
var chai = require( 'chai' );
var mocha = require( 'mocha' );
var tnbc = require( '../src/wink-naive-bayes-text-classifier.js' );

const winkNLP = require( 'wink-nlp' );
const model = require( 'wink-eng-lite-web-model' );
const nlp = winkNLP( model );
const its = nlp.its;

const tokenize0 = function ( text ) {
  const tokens = nlp.readDoc(text).tokens().out();
  return tokens;
};

const stem = function ( text ) {
  const tokens = nlp.readDoc(text).tokens().out( its.stem );
  return tokens;
};

var expect = chai.expect;
var describe = mocha.describe;
var it = mocha.it;


describe( 'definePrepTasks() Error Cases', function () {
  var prepTNBC = tnbc();
  var prepTasks = [
    { whenInputIs: [ undefined, ( p ) => ( p ) ], expectedOutputIs: 'winkNBTC: each task should be a function, instead found: undefined' },
    { whenInputIs: null, expectedOutputIs: 'winkNBTC: tasks should be an array, instead found: null' },
    { whenInputIs: undefined, expectedOutputIs: 'winkNBTC: tasks should be an array, instead found: undefined' },
    { whenInputIs: 1, expectedOutputIs: 'winkNBTC: tasks should be an array, instead found: 1' },
    { whenInputIs: { a: 3 }, expectedOutputIs: 'winkNBTC: tasks should be an array, instead found: {"a":3}' }
  ];

  prepTasks.forEach( function ( ptask ) {
    it( 'should throw "' + ptask.expectedOutputIs + '" if the input is ' + JSON.stringify( ptask.whenInputIs ), function () {
      expect( prepTNBC.definePrepTasks.bind( null, ptask.whenInputIs ) ).to.throw( ptask.expectedOutputIs );
    } );
  } );
}
);


describe( 'definePrepTasks() Proper Cases', function () {
  var prepTNBC = tnbc();
  var prepTasks = [
    { whenInputIs: [ stem, ( p ) => ( p ) ], expectedOutputIs: 2 },
    { whenInputIs: [ ], expectedOutputIs: 0 }
  ];

  prepTasks.forEach( function ( ptask ) {
    it( 'should throw "' + ptask.expectedOutputIs + '" if the input is ' + JSON.stringify( ptask.whenInputIs ), function () {
      expect( prepTNBC.definePrepTasks( ptask.whenInputIs ) ).to.equal( ptask.expectedOutputIs );
    } );
  } );
}
);

describe( 'textNBC() with considerOnlyPresence as true', function () {
  var learnTNBC = tnbc();
  var examples = [
    { whenInputIs: [ 'i want to prepay my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i want to close my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i want to foreclose my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i would like to pay the loan balance', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i would like to borrow money to buy a vehice', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for car', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for a new vehicle', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for a new mobike', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need money for a new car', 'autoloan' ] , expectedOutputIs: true }
  ];

  it( 'definePrepTasks should return 1', function () {
    expect( learnTNBC.definePrepTasks( [ tokenize0 ] ) ).to.equal( 1 );
  } );
  it( 'defineConfig should return true', function () {
    expect( learnTNBC.defineConfig.bind( null, 1 ) ).to.throw( 'winkNBTC: config must be an object, instead found: number' );
  } );
  it( 'defineConfig should return true', function () {
    // It will ensure that smoothing factor is set to 1.
    expect( learnTNBC.defineConfig( { considerOnlyPresence: true } ) ).to.equal( true );
  } );
  examples.forEach( function ( example ) {
    it( 'should return ' + example.expectedOutputIs + ' if the input is ' + JSON.stringify( example.whenInputIs ), function () {
      expect( learnTNBC.learn( example.whenInputIs[ 0 ], example.whenInputIs[ 1 ] ) ).to.equal( example.expectedOutputIs );
    } );
  } );

  it( 'should return true', function () {
    expect( learnTNBC.consolidate() ).to.equal( true );
  } );

  var predict = [
    { whenInputIs: 'I would like to borrow 50000 to buy a new audi r8 in new york', expectedOutputIs: 'autoloan'  },
    { whenInputIs: 'happy', expectedOutputIs: 'unknown' },
    { whenInputIs: 'I want to pay my car loan early', expectedOutputIs: 'prepay' },
    { whenInputIs: '', expectedOutputIs: 'unknown' },
    { whenInputIs: 'happy', expectedOutputIs: 'unknown' }
  ];

  predict.forEach( function ( p ) {
    it( 'should return ' + p.expectedOutputIs + ' if the input is ' + JSON.stringify( p.whenInputIs ), function () {
     expect( learnTNBC.predict( p.whenInputIs ) ).to.equal( p.expectedOutputIs );
    } );
  } );

  var odds = [
    { whenInputIs: 'I would like to borrow 50000 to buy a new audi r8 in new york', expectedOutputIs: [ [ 'autoloan', 3.850918263715954 ], [ 'prepay', -3.850918263715954 ] ]  },
    { whenInputIs: 'happy', expectedOutputIs: [ [ 'unknown', 0 ] ] },
    { whenInputIs: 'I want to pay my car loan early', expectedOutputIs: [ [ 'prepay', 6.169686751688911 ], [ 'autoloan', -6.169686751688911 ] ] },
    { whenInputIs: '', expectedOutputIs: [ [ 'unknown', 0 ] ] },
    { whenInputIs: 'happy', expectedOutputIs: [ [ 'unknown', 0 ] ] }
  ];
  odds.forEach( function ( p ) {
    it( 'should return ' + p.expectedOutputIs + ' if the input is ' + JSON.stringify( p.whenInputIs ), function () {
     expect( learnTNBC.computeOdds( p.whenInputIs ) ).to.deep.equal( p.expectedOutputIs );
    } );
  } );
} );

describe( 'textNBC() with considerOnlyPresence as undefined', function () {
  var learnTNBC = tnbc();
  var examples = [
    { whenInputIs: [ 'i want to prepay my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i want to close my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i want to foreclose my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i would like to pay the loan balance', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i would like to borrow money to buy a vehice', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for car', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for a new vehicle', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for a new mobike', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need money for a new car', 'autoloan' ] , expectedOutputIs: true }
  ];

  it( 'definePrepTasks should return 1', function () {
    expect( learnTNBC.definePrepTasks( [ tokenize0 ] ) ).to.equal( 1 );
  } );

  examples.forEach( function ( example ) {
    it( 'should return ' + example.expectedOutputIs + ' if the input is ' + JSON.stringify( example.whenInputIs ), function () {
      expect( learnTNBC.learn( example.whenInputIs[ 0 ], example.whenInputIs[ 1 ] ) ).to.equal( example.expectedOutputIs );
    } );
  } );

  it( 'should return true', function () {
    expect( learnTNBC.consolidate() ).to.equal( true );
  } );

  var predict = [
    { whenInputIs: 'I would like to borrow 50000 to buy a new audi r8 in new york', expectedOutputIs: 'autoloan'  },
    { whenInputIs: 'I want to pay my car loan early', expectedOutputIs: 'prepay' }
    // { whenInputIs: '', expectedOutputIs: 'unknown' },
    // { whenInputIs: 'happy', expectedOutputIs: 'unknown' }
  ];

  predict.forEach( function ( p ) {
    it( 'should return ' + p.expectedOutputIs + ' if the input is ' + JSON.stringify( p.whenInputIs ), function () {
     expect( learnTNBC.predict( p.whenInputIs ) ).to.equal( p.expectedOutputIs );
    } );
  } );

  var stats = {
        labelWiseSamples: {
          autoloan: 5,
          prepay: 4
        },
        labelWiseWords: {
          autoloan: 36,
          prepay: 26
        },
        vocabulary: 24
      };

  it( 'should return stats if the stats() is called', function () {
   expect( learnTNBC.stats() ).to.deep.equal( stats );
  } );
} );

describe( 'textNBC() with considerOnlyPresence as undefined', function () {
  var learnTNBC = tnbc();
  // Training Data.
  var examples = [
    { whenInputIs: [ 'i want to prepay my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i want to close my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i want to foreclose my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for car', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for a new vehicle', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need money for a new car', 'autoloan' ] , expectedOutputIs: true }
  ];
  // Prediction Data.
  var predict = [
    { whenInputIs: 'want to take loan', expectedOutputIs: 'autoloan' },
    { whenInputIs: 'want to prepay', expectedOutputIs: 'unknown' },
    { whenInputIs: 'buying 4-wheeler, need money', expectedOutputIs: 'unknown' }
  ];
  // Test prepTasks definition.
  it( 'definePrepTasks should return 1', function () {
    expect( learnTNBC.definePrepTasks( [ tokenize0 ] ) ).to.equal( 1 );
  } );
  it( 'defineConfig should return true', function () {
    expect( learnTNBC.defineConfig( { considerOnlyPresence: true, smoothingFactor: 0 } ) ).to.equal( true );
  } );
  // Test learn.
  examples.forEach( function ( example ) {
    it( 'learn should return ' + example.expectedOutputIs + ' if the input is ' + JSON.stringify( example.whenInputIs ), function () {
      expect( learnTNBC.learn( example.whenInputIs[ 0 ], example.whenInputIs[ 1 ] ) ).to.equal( example.expectedOutputIs );
    } );
  } );
  it( 'consolidate should return true', function () {
    expect( learnTNBC.consolidate() ).to.equal( true );
  } );
  predict.forEach( function ( p ) {
    it( 'should return ' + p.expectedOutputIs + ' if the input is ' + JSON.stringify( p.whenInputIs ), function () {
     expect( learnTNBC.predict( p.whenInputIs ) ).to.equal( p.expectedOutputIs );
    } );
  } );
} );


describe( 'textNBC() with considerOnlyPresence as undefined', function () {
  var learnTNBC = tnbc();
  var anotherTNBC = tnbc();
  // Training Data.
  var examples = [
    { whenInputIs: [ 'i want to prepay my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i want to close my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i want to foreclose my loan', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i would like to pay the loan balance', 'prepay' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i would like to borrow money to buy a vehice', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for car', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for a new vehicle', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need loan for a new mobike', 'autoloan' ] , expectedOutputIs: true },
    { whenInputIs: [ 'i need money for a new car', 'autoloan' ] , expectedOutputIs: true }
  ];
  // Prediction Data.
  var predict = [
    { whenInputIs: 'I would like to borrow 50000 to buy a new audi r8 in new york', expectedOutputIs: 'autoloan'  },
    { whenInputIs: 'I want to pay my car loan early', expectedOutputIs: 'prepay' },
    { whenInputIs: '', expectedOutputIs: 'unknown' },
    { whenInputIs: 'buying 4-wheeler, require dollars', expectedOutputIs: 'unknown' }
  ];
  // Test prepTasks definition.
  it( 'definePrepTasks should return 1', function () {
    expect( learnTNBC.definePrepTasks( [ tokenize0 ] ) ).to.equal( 1 );
  } );
  it( 'definePrepTasks should return 1', function () {
    expect( anotherTNBC.definePrepTasks( [ tokenize0 ] ) ).to.equal( 1 );
  } );
  it( 'defineConfig should throw error with illegal smoothingFactor value', function () {
    expect( learnTNBC.defineConfig.bind( null, { considerOnlyPresence: 1, smoothingFactor: 'x' } ) ).to.throw( 'winkNBTC: smoothing factor must be a number between 0 & 1, instead found: null' );
  } );
  // Test learn.
  examples.forEach( function ( example ) {
    it( 'learn should return ' + example.expectedOutputIs + ' if the input is ' + JSON.stringify( example.whenInputIs ), function () {
      expect( learnTNBC.learn( example.whenInputIs[ 0 ], example.whenInputIs[ 1 ] ) ).to.equal( example.expectedOutputIs );
    } );
  } );
  // Define Config should throw error.
  it( 'defineConfig should throw error post learning', function () {
    expect( learnTNBC.defineConfig.bind( null, { considerOnlyPresence: 1, smoothingFactor: 'x' } ) ).to.throw( 'winkNBTC: config must be defined before learning starts!' );
  } );
  // Test consolidation.
  it( 'consolidate should return true', function () {
    expect( learnTNBC.consolidate() ).to.equal( true );
  } );
  // Test predict.
  predict.forEach( function ( p ) {
    it( 'should return ' + p.expectedOutputIs + ' if the input is ' + JSON.stringify( p.whenInputIs ), function () {
     expect( learnTNBC.predict( p.whenInputIs ) ).to.equal( p.expectedOutputIs );
    } );
  } );
  // Test export to JSON.
  var json;
  it( 'exportJSON should return an array', function () {
    json = learnTNBC.exportJSON();
    expect( Array.isArray(JSON.parse( json ) ) ).to.equal( true );
  } );
  // Test reset.
  it( 'reset should return true', function () {
    expect( learnTNBC.reset() ).to.equal( true );
  } );
  // Test import JSON
  it( 'importJSON should return true', function () {
    expect( learnTNBC.importJSON( json ) ).to.equal( true );
  } );
  it( 'imoprtJSON should throw error when input is empty', function () {
    expect( anotherTNBC.importJSON.bind( null ) ).to.throw( 'winkNBTC: undefined or null JSON encountered, import failed!' );
  } );
  it( 'imoprtJSON should throw error when input is an object', function () {
    expect( anotherTNBC.importJSON.bind( null, JSON.stringify( { a: 1 } ) ) ).to.throw( 'winkNBTC: invalid JSON encountered, can not import.' );
  } );
  it( 'imoprtJSON should throw error when input is an array of wrong length', function () {
    expect( anotherTNBC.importJSON.bind( null, JSON.stringify( [ 1, 2 ] ) ) ).to.throw( 'winkNBTC: invalid JSON encountered, can not import.' );
  } );
  it( 'imoprtJSON should throw error when input is an array has wrong elements', function () {
    expect( anotherTNBC.importJSON.bind( null, JSON.stringify( [ [], {}, [], {}, [] ] ) ) ).to.throw( 'winkNBTC: invalid JSON encountered, can not import.' );
  } );
  it( 'imoprtJSON should return an true on valid JSON input', function () {
    expect( anotherTNBC.importJSON( json ) ).to.equal( true );
  } );
  // Test throwing errors without consolidation
  predict.forEach( function ( p ) {
    it( 'learnTNBC should return throw error unconsolidated', function () {
     expect( learnTNBC.predict.bind( null, p.whenInputIs ) ).to.throw( 'winkNBTC: prediction is not possible unless learnings are consolidated!' );
    } );
  } );
  predict.forEach( function ( p ) {
    it( 'anotherTNBC should return throw error unconsolidated', function () {
     expect( anotherTNBC.predict.bind( null, p.whenInputIs ) ).to.throw( 'winkNBTC: prediction is not possible unless learnings are consolidated!' );
    } );
  } );
  // Now again consolidat.
  it( 'learnTNBC consolidate should return true post import', function () {
    expect( learnTNBC.consolidate() ).to.equal( true );
  } );
  it( 'anotherTNBC consolidate should return true post import', function () {
    expect( anotherTNBC.consolidate() ).to.equal( true );
  } );
  // Learning must fail after consolidation.
  examples.forEach( function ( example ) {
    it( 'learnTNBC learn post consolidation should return throw error', function () {
      expect( learnTNBC.learn.bind( null, example.whenInputIs[ 0 ], example.whenInputIs[ 1 ] ) ).to.throw( 'winkNBTC: post consolidation learning is not possible!' );
    } );
  } );
  examples.forEach( function ( example ) {
    it( 'anotherTNBC learn post consolidation should return throw error', function () {
      expect( anotherTNBC.learn.bind( null, example.whenInputIs[ 0 ], example.whenInputIs[ 1 ] ) ).to.throw( 'winkNBTC: post consolidation learning is not possible!' );
    } );
  } );
  // Test metrics.
  it( 'metrics should throw error without evaluation', function () {
    expect( anotherTNBC.metrics.bind( null ) ).to.throw( 'winkNBTC: metrics can not be computed before evaluation.' );
  } );
  // Now evaluate; partial not for all labels
  it( 'evaluate should throw error with unknown label', function () {
    expect( anotherTNBC.evaluate.bind( null, 'some funny input', 'fun' ) ).to.throw( 'winkNBTC: can not evaluate, unknown label enountered: "fun"' );
  } );
  // ***
  it( 'evaluate should return false with unknown vocab inputs', function () {
    expect( anotherTNBC.evaluate( 'some funny input', 'prepay' ) ).to.equal( false );
  } );
  it( 'evaluate should return true with proper inputs', function () {
    expect( anotherTNBC.evaluate( 'can i close my loan', 'prepay' ) ).to.equal( true );
  } );
  // Test metrics.
  it( 'metrics should throw error without evaluation', function () {
    expect( Object.keys( anotherTNBC.metrics( ) ).length ).to.equal( 4 );
  } );
  // More evaluation - complete it now.
  it( 'evaluate should return true with proper inputs', function () {
    expect( anotherTNBC.evaluate( 'i will use excess fund to close the loan', 'autoloan' ) ).to.equal( true );
  } );
  it( 'evaluate should return true with proper inputs', function () {
    expect( anotherTNBC.evaluate( 'i need to buy a car on loan', 'autoloan' ) ).to.equal( true );
  } );
  // Test metrics.
  it( 'metrics should throw error without evaluation', function () {
    expect( Object.keys( anotherTNBC.metrics( ) ).length ).to.equal( 4 );
  } );
  // Small vocal/labels test
  it( 'reset should return true', function () {
    expect( learnTNBC.reset() ).to.equal( true );
  } );
  // Learn only one label
  it( 'learn  with just one label and it should return true', function () {
    expect( learnTNBC.learn( 'Hi, good morning', 'greet' ) ).to.equal( true );
  } );
  // Consolidation should fail.
  it( 'anotherTNBC learn post consolidation should return throw error', function () {
    expect( learnTNBC.consolidate.bind( null ) ).to.throw( 'winkNBTC: can not consolidate as classification require 2 or more labels!' );
  } );
  // Learn with one more label but ensure less vocab.
  it( 'learn  with just one label and it should return true', function () {
    expect( learnTNBC.learn( 'See you soon', 'bye' ) ).to.equal( true );
  } );
  // Consolidation should fail due to low vocab.
  it( 'anotherTNBC learn post consolidation should return throw error', function () {
    expect( learnTNBC.consolidate.bind( null ) ).to.throw( 'winkNBTC: vocabulary is too small to learn meaningful classification!' );
  } );
} );
