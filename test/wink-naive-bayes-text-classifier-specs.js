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

// ### Define common errors.

// These are common test data for `null`, `undefined`, and `numeric` inputs
// across all the functions included in the script.
// The exception cases specific to the function are part of the test script of the function.
// var errors = [
//   { whenInputIs: null, expectedOutputIs: /^Cannot read.*/ },
//   { whenInputIs: undefined, expectedOutputIs: /^Cannot read.*/ },
//   { whenInputIs: 1, expectedOutputIs: /is not a function$/ }
// ];

describe( 'defineprep()', function () {
  var prepTNBC = tnbc();
  var prepTasks = [
    // Contains non-function.
    { whenInputIs: [ prepare.string.incorrect, prepare.string.lowerCase ], expectedOutputIs: false },
    // Has valid functions only.
    { whenInputIs: [ prepare.string.upperCase, prepare.string.lowerCase ], expectedOutputIs: true },
    // Error cases.
    { whenInputIs: null, expectedOutputIs: false },
    { whenInputIs: undefined, expectedOutputIs: false },
    { whenInputIs: 1, expectedOutputIs: false },
    // Empty array is valid!
    { whenInputIs: [], expectedOutputIs: true }
  ];

  prepTasks.forEach( function ( ptask ) {
    it( 'should return ' + ptask.expectedOutputIs + ' if the input is ' + JSON.stringify( ptask.whenInputIs ), function () {
      expect( prepTNBC.definePrepTasks( ptask.whenInputIs ) ).to.equal( ptask.expectedOutputIs );
    } );
  } );
}
);

// ### XXXXXX test cases.

describe( 'textNBC()', function () {
  var learnTNBC = tnbc();
  var examples = [
    { whenInputIs: [ 'a great start and continues to amuse!', 'positive' ] , expectedOutputIs: true },
    { whenInputIs: [ 'a great beginning, lot of funny dialogues ', 'positive' ] , expectedOutputIs: true },
    { whenInputIs: [ 'hillarious and upbeat', 'positive' ] , expectedOutputIs: true },
    { whenInputIs: [ 'boring, failed to appeal', 'negative' ] , expectedOutputIs: true },
    { whenInputIs: [ 'felt like disappearing from the cinema hall, pathetic', 'negative' ] , expectedOutputIs: true },
    { whenInputIs: [ 'will not recommend to anyone!', 'negative' ] , expectedOutputIs: true }
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
    { whenInputIs: 'recommend', expectedOutputIs: 'negative'  },
    { whenInputIs: 'disappearing', expectedOutputIs: 'negative' },
    { whenInputIs: '', expectedOutputIs: undefined },
    { whenInputIs: 'happy', expectedOutputIs: undefined }
  ];

  predict.forEach( function ( p ) {
    it( 'should return ' + p.expectedOutputIs + ' if the input is ' + JSON.stringify( p.whenInputIs ), function () {
     expect( learnTNBC.predict( p.whenInputIs ) ).to.equal( p.expectedOutputIs );
    } );
  } );
} );
