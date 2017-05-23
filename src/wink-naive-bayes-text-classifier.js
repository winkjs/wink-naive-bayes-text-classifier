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
var helpers = require( 'wink-helpers' );

// Because we want to logically group the variables.
/* eslint sort-vars: 0 */

// It is a **N**aive **B**ayes **C**lassifier for **text** classification.
// It exposes following methods:
// 1. `definePrepTasks` allows to define a pipeline of functions that will be
// used to prepare each input prior to *learning*, *prediction*, and *evaluation*.
// 2. `defineConfig` sets up the configuration for *mode* and *smoothing factor*.
// 3. `learn` from example *input* and *label* pair(s).
// 4. `consolidate` learnings prior to evaluation and/or prediction.
// 5. `predict` the best *label* for the given *input*.
// 6. `stats` of learnings.
// 7. `exportJSON` exports the learnings in JSON format.
// 8. `importJSON` imports the learnings from JSON that may have been saved on disk.
// 9. `evaluate` the learnings from known examples of *input* and corresponding
// *label* by internally building a confusion matrix.
// 10. `metrics` are primarily macro-averages of *precison*, *recall*,
// and *f-measure* computed from the confusion matrix built during the evaluation
// phase.
// 11. `reset` all the learnings except the preparatory tasks; useful during
// cross-validation.
var textNBC = function () {
  // Total samples encountered under each label during learning.
  var samples = Object.create( null );
  // Maintains label-wise count of each word encountered during learning.
  var count = Object.create( null );
  // Maintains count of words encountered under a label during learning.
  var words = Object.create( null );
  // The entire vocabulary.
  var voc = new Set();
  // Preparatory tasks that are executed on the `learn` & `predict` input.
  var pTasks = [];
  // And its count.
  var pTaskCount;
  // All the labels seen till the point of consolidation.
  var labels;
  // And their count: meant to be used in for-loops.
  var labelCount;
  // The `defineConfig()` checks this before latering config.
  var learned = false;
  // The `predict()` function checks for this being true; set in `consolidate()`.
  var consolidated = false;
  // The `metrics()` checks this; set in `evaluate()`.
  var evaluated = false;
  // Confusion Matrix.
  var cm = Object.create( null );
  // metrics: Precision, Recall, and F-Measure
  var precision = Object.create( null );
  var recall = Object.create( null );
  var fmeasure = Object.create( null );
  // Returned!
  var methods = Object.create( null );
  // Define unknown prediction.
  var unknown = 'unknown';
  // Configuration - `considerOnlyPresence` flag and `smoothingFactor`.
  var config = Object.create( null );
  // Set their default values.
  config.considerOnlyPresence = false;
  config.smoothingFactor = 0.5;

  // ### Private functions

  // #### Prepare Input

  // Prepares the `input` by building a pipeline of tasks defined in the variable
  // `pTasks` via `definePrepTasks()`
  var prepareInput = function ( input ) {
    var processedInput = input;
    for ( var i = 0; i < pTaskCount; i += 1 ) {
      processedInput = pTasks[ i ]( processedInput );
    }
    return ( processedInput );
  }; // prepareInput()

  // #### Log Likelihood

  // Computes the pre-definable smoothed log likelihood `( w | label )`.
  var logLikelihood = function ( w, label ) {
    // If there is a **non-zero** `smoothingFactor`, then use the regular
    // formula for computation. When it is **0**, in that case if the `w`
    // is not found in vocabulary, return 0; otherwise perform add-1.
    // Note, a 0 `smoothingFactor` can lead to `unknown` prediction if non-zero
    // of the words are found in the vocabulary.
    return (
      ( config.smoothingFactor > 0 ) ?
        Math.log2( ( ( count[ label ][ w ] || 0 ) + config.smoothingFactor ) /
                ( words[ label ] + ( voc.size * config.smoothingFactor ) ) ) :
        voc.has( w ) ?  Math.log2( ( ( count[ label ][ w ] || 0 ) + 1 ) /
                ( words[ label ] + voc.size ) ) :
                0
    );
  }; // logLikelihood()

  // #### Inverse Log Likelihood

  // Computes the 1+ smoothed log likelihood `( w | label )`.
  var inverseLogLikelihood = function ( w, label ) {
    // Index and temporary label.
    var i, l;
    // `count[ l ][ w ]`.
    var clw = 0;
    // `words[ l ]`
    var wl = 0;

    for ( i = 0; i < labelCount; i += 1 ) {
      l = labels[ i ];
      if ( l !== label ) {
        wl += words[ l ];
        clw += ( count[ l ][ w ] || 0 );
      }
    }
    // No need to perform `voc.has( w )` check as `odds()` will not call the
    // `inverseLogLikelihood()` if `logLikelihood()` returns a **0**. It does
    // so to avoid recomputation. See comments in `logLikelihood()`.
    return ( Math.log2( ( clw + ( config.smoothingFactor || 1 ) ) /
              ( wl + ( voc.size * ( config.smoothingFactor || 1 ) ) ) ) );

  }; // inverseLogLikelihood()

  // #### Odds

  // Computes the odds for `( tokens | label )`.
  var odds = function ( tokens, label ) {
    // Total number of samples encountered during training.
    var sum = 0;
    // Samples enountered under `label` during training.
    var samplesInLabel = samples[ label ];
    // Samples NOT enountered under the `label`.
    var samplesNotInLabel = 0;
    // Log Base 2 Likelihood & Inverse likelihood
    var lh = 0,
        ilh = 0;
    // Temp Label.
    var lbl, i, imax;

    // Compute `samplesNotInLabel`.
    for ( i = 0; i < labelCount; i += 1 ) {
      lbl = labels[ i ];
      sum += samples[ lbl ];
      samplesNotInLabel += ( lbl === label ) ? 0 : samples[ lbl ];
    }

    // Update them for the given tokens for `label`
    for ( i = 0, imax = tokens.length; i < imax; i += 1 ) {
      lh += logLikelihood( tokens[ i ], label );
      // If `lh` is **0** then ilh will be zero - avoid computation.
      ilh += ( lh === 0 ) ? 0 : inverseLogLikelihood( tokens[ i ], label );
    }

    // Add prior probablities only if 1 or more tokens are found in `voc`.
    if ( lh !== 0 ) {
      // Add prior probabilities as `lh` (and therefore `ilh`) is **0**.
      lh += Math.log2( samplesInLabel / sum );
      ilh += Math.log2( samplesNotInLabel / sum  );
    }

    // Return the log likelihoods ratio; subtract as it is a log. This will
    // be a measure of distance between the probability & inverse probability.
    return ( lh - ilh );
  }; // odds()

  // ### Exposed Functions

  // #### Define Config

  // Defines the `considerOnlyPresence` and `smoothingFactor` parameters. The
  // `considerOnlyPresence` is a boolean parameter. An incorrect value is
  // forced to `false`. Setting `considerOnlyPresence` to `true` ignores
  //  the frequency of each token and instead only considers it's presence.
  // The `smoothingFactor` can have any value between 0 and 1. If the input
  // value > 1 can have any value between 0 and 1. If the input value > 1
  // then it is set to **1** and if it is <0 then it is set to **0**.
  // The config can not be set once the learning has started.
  var defineConfig = function ( cfg ) {
    if ( learned ) {
      throw Error( 'winkNBTC: config must be defined before learning starts!' );
    }
    if ( !helpers.object.isObject( cfg ) ) {
      throw Error( 'winkNBTC: config must be an object, instead found: ' + ( typeof cfg ) );
    }
    config.considerOnlyPresence = ( typeof cfg.considerOnlyPresence === 'boolean' ) ?
                                    cfg.considerOnlyPresence : false;
    config.smoothingFactor = ( isNaN( cfg.smoothingFactor ) ) ?
            0 : Math.max( Math.min( cfg.smoothingFactor, 1 ), 0 );
    return true;
  }; // defineConfig()

  // #### Define Prep Tasks

  // Defines the `tasks` required to prepare the input for `learn()` and `predict()`
  // The `tasks` should be an array of functions; using these function a simple
  // pipeline is built to serially transform the input to the output.
  // It validates the `tasks` before updating the `pTasks`.
  // If validation fails it throws error; otherwise it sets the
  // `pTasks` and returns length of `pTask` array.
  var definePrepTasks = function ( tasks ) {
    if ( !helpers.array.isArray( tasks ) ) {
      throw Error( 'winkNBTC: tasks should be an array, instead found: ' + JSON.stringify( tasks ) );
    }
    for ( var i = 0, imax = tasks.length; i < imax; i += 1 ) {
      if ( typeof tasks[ i ] !== 'function' ) {
        throw Error( 'winkNBTC: each task should be a function, instead found: ' + JSON.stringify( tasks[ i ] ) );
      }
    }
    pTasks = tasks;
    pTaskCount = tasks.length;
    return pTaskCount;
  }; // definePrepTasks()

  // #### Learn

  // Learns from example pair of `input` and `label`. It throws error if
  // consolidation has already been carried out.
  // If learning was successful then it returns `true`.
  var learn = function ( input, label ) {
    // No point in learning further, if learnings so far have been consolidated.
    if ( consolidated ) {
      throw Error( 'winkNBTC: post consolidation learning is not possible!' );
    }
    // Set learning started.
    learned = true;
    // Prepare the input.
    var tkns = prepareInput( input );
    // Update vocubulary, count, and words i.e. learn!
    samples[ label ] = 1 + ( samples[ label ] || 0 );
    if ( config.considerOnlyPresence ) tkns = new Set( tkns );
    count[ label ] = count[ label ] || Object.create( null );
    tkns.forEach( function ( token ) {
      count[ label ][ token ] = 1 + ( count[ label ][ token ] || 0 );
      voc.add( token );
      words[ label ] = 1 + ( words[ label ] || 0 );
    } );
    return true;
  }; // learn()

  // #### Consolidate

  // Consolidates the learnings in following steps:
  // 1. Check presence of minimal learning mass, if present proceed further;
  // otherwise it throws appropriate error.
  // 2. Initializes the confusion matrix and metrics.
  var consolidate = function () {
    var row, col;
    var i, j;
    // Extract all labels that have been seen during learning phase.
    labels = helpers.object.keys( samples );
    labelCount = labels.length;
    // A quick & simple check of some minimal learning mass!
    if ( labelCount < 2 ) {
      throw Error( 'winkNBTC: can not consolidate as classification require 2 or more labels!' );
    }
    if ( voc.size < 10 ) {
      throw Error( 'winkNBTC: vocabulary is too small to learn meaningful classification!' );
    }
    // Initialize confusion matrix and metrics.
    for ( i = 0; i < labelCount; i += 1 ) {
      row = labels[ i ];
      cm[ row ] = Object.create( null );
      precision[ row ] = 0;
      recall[ row ] = 0;
      fmeasure[ row ] = 0;
      for ( j = 0; j < labelCount; j += 1 ) {
        col = labels[ j ];
        cm[ row ][ col ] = 0;
      }
    }
    // Set `consolidated` as `true`.
    consolidated = true;
    return true;
  }; // consolidate()

  // #### Predict

  // Predicts the potential **label** for the given `input`, provided learnings
  // have been consolidated. If all the `input` tokens have never been seen
  // in past (i.e. absent in learnings), then the predicted label is `unknown`.
  // It throws error if the learnings have not been consolidated.
  var predict = function ( input ) {
    // Predict only if learnings have been consolidated!
    if ( !consolidated ) {
      throw Error( 'winkNBTC: prediction is not possible unless learnings are consolidated!' );
    }
    // Contains label & the corresponding odds pairs.
    var allOdds = [];
    // Temporary label.
    var label;
    for ( var i = 0; i < labelCount; i += 1 ) {
      label = labels[ i ];
      allOdds.push( [ label, odds( prepareInput( input ), label ) ] );
    }
    // Sort descending for argmax.
    allOdds.sort( helpers.array.descendingOnValue );
    // If odds for the top label is 0 means prediction is `unknown`
    // otherwise return the corresponding label.
    return ( ( allOdds[ 0 ][ 1 ] ) ? allOdds[ 0 ][ 0 ] : unknown );
  };

  // #### Stats

  // Returns basic stats of learning.
  var stats = function () {
    return (
      {
        // Count of samples under each label.
        labelWiseSamples: JSON.parse( JSON.stringify( samples ) ),
        // Total words (a single word occuring twice is counted as 2)
        // under each label.
        labelWiseWords: JSON.parse( JSON.stringify( words ) ),
        // Size of the vocubulary.
        vocabulary: voc.size
      }
    );
  }; // predict()

  // #### Export JSON

  // Returns the learnings, without any consolidation check, in JSON format.
  var exportJSON = function ( ) {
    var vocArray = [];
    // Vocubulary set needs to be converted to an array.
    voc.forEach( function ( e ) {
      vocArray.push( e );
    } );
    return ( JSON.stringify( [ config, samples, count, words, vocArray ] ) );
  }; // exportJSON()

  // #### Reset

  // Resets the classifier completely by re-initializing all the learning
  // related variables, except the preparatory tasks. Useful during cross-
  // validation.
  var reset = function () {
    // Reset values of variables that are associated with learning; Therefore
    // `pTasks` & `pTaskCount` are not re-initialized.
    samples = Object.create( null );
    count = Object.create( null );
    words = Object.create( null );
    voc = new Set();
    labels = null;
    labelCount = 0;
    learned = false;
    consolidated = false;
    evaluated = false;
    cm = Object.create( null );
    precision = Object.create( null );
    recall = Object.create( null );
    fmeasure = Object.create( null );
    return true;
  }; // reset()

  // #### Import JSON

  // Imports the `json` in to learnings after validating the format of input JSON.
  // If validation fails then throws error; otherwise on success import it
  // returns `true`. Note, importing leads to resetting the classifier.
  var importJSON = function ( json ) {
    if ( !json ) {
      throw Error( 'winkNBTC: undefined or null JSON encountered, import failed!' );
    }
    // Validate json format
    var isOK = [
      helpers.object.isObject,
      helpers.object.isObject,
      helpers.object.isObject,
      helpers.object.isObject,
      helpers.array.isArray
    ];
    var parsedJSON = JSON.parse( json );
    if ( !helpers.array.isArray( parsedJSON ) || parsedJSON.length !== isOK.length ) {
      throw Error( 'winkNBTC: invalid JSON encountered, can not import.' );
    }
    for ( var i = 0; i < isOK.length; i += 1 ) {
      if ( !isOK[ i ]( parsedJSON[ i ] ) ) {
        throw Error( 'winkNBTC: invalid JSON encountered, can not import.' );
      }
    }
    // All good, setup variable values.
    // First reset everything.
    reset();
    // To prevent config change.
    learned = true;
    // Load variable values.
    config = parsedJSON[ 0 ];
    samples = parsedJSON[ 1 ];
    count = parsedJSON[ 2 ];
    words = parsedJSON[ 3 ];
    // Vocabulary is a set!
    voc = new Set( parsedJSON[ 4 ] );
    // Return success.
    return true;
  }; // importJSON()

  // #### Evaluate

  // Evaluates the prediction using the `input` and its known `label`. It
  // accordingly updates the confusion matrix. If the `label` is unknown
  // then it throws error; errors may be thrown by the `predict()`. If
  // prediction fails (nunknown), then it does not uppdate
  // the confusion matrix and returns `false`; otherwise it updates the matrix
  // and returns `true`.
  var evaluate = function ( input, label ) {
    // In case of unknown label, indicate failure
    if ( !samples[ label ] ) {
      throw Error( 'winkNBTC: can not evaluate, unknown label enountered: ' + JSON.stringify( label ) );
    }
    var prediction = predict( input );
    // If prediction failed then return false!
    if ( prediction === unknown ) return false;
    // Update confusion matrix.
    if ( prediction === label ) {
      cm[ label ][ prediction ] += 1;
    } else {
      cm[ prediction ][ label ] += 1;
    }
    evaluated = true;
    return true;
  }; // evaluate()

  // #### metrics

  // Computes the metrics from the confusion matrix built during the evaluation
  // phase via `evaluate()`. In absence of evaluations, it throws error; otherwise
  // it returns an object containing summary metrics along with the details.
  var metrics = function () {
    if ( !evaluated ) {
      throw Error( 'winkNBTC: metrics can not be computed before evaluation.' );
    }
    // Numerators for every label; they are same for precision & recall both.
    var n = Object.create( null );
    // Only denominators differs for precision & recall
    var pd = Object.create( null );
    var rd = Object.create( null );
    // `row` and `col` of confusion matrix.
    var row, col;
    var i, j;
    // Macro average values for metrics.
    var avgPrecision = 0;
    var avgRecall = 0;
    var avgFMeasure = 0;

    // Compute label-wise numerators & denominators!
    for ( i = 0; i < labelCount; i += 1 ) {
      row = labels[ i ];
      for ( j = 0; j < labelCount; j += 1 ) {
        col = labels[ j ];
        if ( row === col ) {
          n[ row ] = cm[ row ][ col ];
        }
        pd[ row ] = cm[ row ][ col ] + ( pd[ row ] || 0 );
        rd[ row ] = cm[ col ][ row ] + ( rd[ row ] || 0 );
      }
    }
    // Ready to compute metrics.
    for ( i = 0; i < labelCount; i += 1 ) {
      row = labels[ i ];
      precision[ row ] = +( n[ row ] / pd[ row ] ).toFixed( 4 );
      // NaN can occur if a label has not been encountered.
      if ( isNaN( precision[ row ] ) ) precision[ row ] = 0;

      recall[ row ] = +( n[ row ] / rd[ row ] ).toFixed( 4 );
      if ( isNaN( recall[ row ] ) ) recall[ row ] = 0;

      fmeasure[ row ] = +( 2 * precision[ row ] * recall[ row ] / ( precision[ row ] + recall[ row ] ) ).toFixed( 4 );
      if ( isNaN( fmeasure[ row ] ) ) fmeasure[ row ] = 0;
    }
    // Compute thier averages, note they will be macro avegages.
    for ( i = 0; i < labelCount; i += 1 ) {
      avgPrecision += ( precision[ labels[ i ] ] / labelCount );
      avgRecall += ( recall[ labels[ i ] ] / labelCount );
      avgFMeasure += ( fmeasure[ labels[ i ] ] / labelCount );
    }
    // Return metrics.
    return (
      {
        // Macro-averaged metrics.
        avgPrecision: +avgPrecision.toFixed( 4 ),
        avgRecall: +avgRecall.toFixed( 4 ),
        avgFMeasure: +avgFMeasure.toFixed( 4 ),
        details: {
          // Confusion Matrix.
          confusionMatrix: cm,
          // Label wise metrics details, from those averages were computed.
          precision: precision,
          recall: recall,
          fmeasure: fmeasure
        }
      }
    );
  }; // metrics()


  methods.learn = learn;
  methods.consolidate = consolidate;
  methods.predict = predict;
  methods.stats = stats;
  methods.definePrepTasks = definePrepTasks;
  methods.defineConfig = defineConfig;
  methods.evaluate = evaluate;
  methods.metrics = metrics;
  methods.exportJSON = exportJSON;
  methods.importJSON = importJSON;
  methods.reset = reset;

  return ( methods );
};

// Export textNBC.
module.exports = textNBC;
