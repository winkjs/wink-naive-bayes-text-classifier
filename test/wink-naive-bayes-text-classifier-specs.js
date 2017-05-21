//     wink-naive-bayes-text-classifier
//     Configurable Naive Bayes Classifier for text
//     with cross-validation support.
//
//     Copyright (C) 2017  GRAYPE Systems Private Limited
//
//     This file is part of “wink-naive-bayes-text-classifier”.
//
//     “wink-naive-bayes-text-classifier” is free software: you can redistribute it
//     and/or modify it under the terms of the GNU Affero
//     General Public License as published by the Free
//     Software Foundation, version 3 of the License.
//
//     “wink-naive-bayes-text-classifier” is distributed in the hope that it will
//     be useful, but WITHOUT ANY WARRANTY; without even
//     the implied warranty of MERCHANTABILITY or FITNESS
//     FOR A PARTICULAR PURPOSE.  See the GNU Affero General
//     Public License for more details.
//
//     You should have received a copy of the GNU Affero
//     General Public License along with “wink-naive-bayes-text-classifier”.
//     If not, see <http://www.gnu.org/licenses/>.

//
var chai = require( 'chai' );
var mocha = require( 'mocha' );
var tnbc = require( '../src/wink-naive-bayes-text-classifier.js' );
var prepare = require( 'wink-nlp-utils' );

var expect = chai.expect;
var describe = mocha.describe;
var it = mocha.it;


describe( 'definePrepTasks() Error Cases', function () {
  var prepTNBC = tnbc();
  var prepTasks = [
    { whenInputIs: [ prepare.string.incorrect, prepare.string.lowerCase ], expectedOutputIs: 'winkNBTC: each task should be a function, instead found: undefined' },
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
    { whenInputIs: [ prepare.string.tokenize0, prepare.string.stem ], expectedOutputIs: 2 },
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

  learnTNBC.definePrepTasks( [ prepare.string.tokenize0 ] );
  examples.forEach( function ( example ) {
    it( 'should return ' + example.expectedOutputIs + ' if the input is ' + JSON.stringify( example.whenInputIs ), function () {
      expect( learnTNBC.learn( example.whenInputIs[ 0 ], example.whenInputIs[ 1 ], true ) ).to.equal( example.expectedOutputIs );
    } );
  } );

  it( 'should return true', function () {
    expect( learnTNBC.consolidate() ).to.equal( true );
  } );

  var predict = [
    { whenInputIs: 'I would like to borrow 50000 to buy a new audi r8 in new york', expectedOutputIs: 'autoloan'  },
    { whenInputIs: 'I want to pay my car loan early', expectedOutputIs: 'prepay' },
    { whenInputIs: '', expectedOutputIs: undefined },
    { whenInputIs: 'happy', expectedOutputIs: undefined }
  ];

  predict.forEach( function ( p ) {
    it( 'should return ' + p.expectedOutputIs + ' if the input is ' + JSON.stringify( p.whenInputIs ), function () {
     expect( learnTNBC.predict( p.whenInputIs ) ).to.equal( p.expectedOutputIs );
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

  learnTNBC.definePrepTasks( [ prepare.string.tokenize0 ] );
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
    { whenInputIs: 'I want to pay my car loan early', expectedOutputIs: 'prepay' },
    { whenInputIs: '', expectedOutputIs: undefined },
    { whenInputIs: 'happy', expectedOutputIs: undefined }
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
