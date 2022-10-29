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
/**
 *
 * Creates an instance of a {@link NaiveBayesTextClassifier}.
 *
 * @return {NaiveBayesTextClassifier} object conatining set of API methods for tasks like configuration,
 * data ingestion, learning, and prediction etc.
 * @example
 * // Load wink Naive Bayes Text Classifier.
 * var naiveBayesTextClassifier = require( 'wink-naive-bayes-text-classifier' );
 * // Create your instance of classifier.
 * var myClassifier = naiveBayesTextClassifier();
*/
var naiveBayesTextClassifier = function () {
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

  /**
   * @classdesc Naive Bayes Text Classifier class.
   * @class NaiveBayesTextClassifier
   * @hideconstructor
   */
  var methods = Object.create( null );
  // Define unknown prediction.
  var unknown = 'unknown';
  // Configuration - `considerOnlyPresence` flag and `smoothingFactor`.
  var config = Object.create( null );
  // Set their default values.
  config.considerOnlyPresence = false;
  // Default smoothingFactor is set to Laplace add+1 smoothing.
  config.smoothingFactor = 1;

  // ### Private functions

  // #### Prepare Input
  /**
   *
   * Prepares the `input` by building a pipeline of tasks defined in the variable
   * `pTasks` via `definePrepTasks()`.
   *
   * @param {string} input usually a text
   * @return {string[]} tokens.
   * @private
  */
  var prepareInput = function ( input ) {
    var processedInput = input;
    for ( var i = 0; i < pTaskCount; i += 1 ) {
      processedInput = pTasks[ i ]( processedInput );
    }
    return ( processedInput );
  }; // prepareInput()

  // #### Log Likelihood

  /**
   *
   * Computes the pre-definable smoothed log likelihood `( w | label )`.
   *
   * @param {string} w word or token.
   * @param {string} label i.e. class.
   * @return {number} smoothed log likelihood.
   * @private
  */
  var logLikelihood = function ( w, label ) {
    // To avoid recomputation.
    var clw = ( count[ label ][ w ] || 0 );
    return (
      ( config.smoothingFactor > 0 ) ?
        // Numerator will never be **0** due to smoothing.
        ( Math.log2( ( clw + config.smoothingFactor ) ) -
          Math.log2( words[ label ] + ( voc.size * config.smoothingFactor ) ) ) :
        // Numerator will be 0 if `w` is not found under the `label`.
        ( clw ) ?
          // Non-zero numerator means normal handling
          ( Math.log2( clw ) - Math.log2( ( words[ label ] + voc.size ) ) ) :
          // Zero numerator: return **0**.
          0
    );
  }; // logLikelihood()

  // #### Inverse Log Likelihood

  /**
   *
   * Computes the pre-definable smoothed inverse log likelihood `( w | label )`.
   *
   * @param {string} w word or token.
   * @param {string} label i.e. class.
   * @return {number} smoothed inverse log likelihood.
   * @private
  */
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

    return (
      ( config.smoothingFactor > 0 ) ?
        // Numerator will never be **0** due to smoothing.
        ( Math.log2( ( clw + config.smoothingFactor ) ) -
          Math.log2( wl + ( voc.size * config.smoothingFactor ) ) ) :
        // Numerator may be 0.
        ( clw ) ?
          // Non-zero numerator means normal handling
          ( Math.log2( clw ) - Math.log2( ( wl + voc.size ) ) ) :
          // Zero numerator: return **0**.
          0
    );
  }; // inverseLogLikelihood()

  // #### Odds

  // Computes the odds for `( tokens | label )`.
  /**
   *
   * Computes the odds for `( tokens | label )`.
   *
   * @param {string[]} tokens of the sentence.
   * @param {string} label i.e. class of sentence.
   * @return {number} odds for `( tokens | label )`.
   * @private
  */
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

    // Filter unknown tokens.
    var ivTokens = tokens.filter( function ( e ) {
      return voc.has( e );
    } );
    // No known tokens means simply return **0**.
    if ( ivTokens.length === 0 ) return 0;

    // Compute `samplesNotInLabel`.
    for ( i = 0; i < labelCount; i += 1 ) {
      lbl = labels[ i ];
      sum += samples[ lbl ];
      samplesNotInLabel += ( lbl === label ) ? 0 : samples[ lbl ];
    }

    // Update them for the given tokens for `label`
    for ( i = 0, imax = ivTokens.length; i < imax; i += 1 ) {
      lh += logLikelihood( ivTokens[ i ], label );
      // If `lh` is **0** then ilh will be zero - avoid computation.
      ilh += ( lh === 0 ) ? 0 : inverseLogLikelihood( ivTokens[ i ], label );
    }

    // Add prior probablities only if 1 or more tokens are found in `voc`.
    if ( lh !== 0 ) {
      // Add prior probabilities as `lh` (and therefore `ilh`) is **0**.
      lh += ( Math.log2( samplesInLabel ) - Math.log2( sum ) );
      ilh += ( Math.log2( samplesNotInLabel ) - Math.log2( sum  ) );
    }

    // Return the log likelihoods ratio; subtract as it is a log. This will
    // be a measure of distance between the probability & inverse probability.
    return ( lh - ilh );
  }; // odds()

  // ### Exposed Functions

  // #### Define Config

  /**
   *
   * Defines the configuration for naive bayes text classifier. This
   * must be called before attempting to [learn](#learn); in other words it can not be
   * set once learning has started.
   *
   * @method NaiveBayesTextClassifier#defineConfig
   * @param {object} cfg defines the configuration in terms of the following
   * parameters:
   * @param {boolean} [considerOnlyPresence=false] true indicates a binarized model.
   * @param {number} [smoothingFactor=1] defines the value for additive smoothing.
   * It can have any value between 0 and 1.
   * @return {boolean} Always true.
   * @example
   * myClassifier.defineConfig( { considerOnlyPresence: true, smoothingFactor: 0.5 } );
   * // -> true
   * @throws Error if `cfg` is not a valid Javascript object, or `smoothingFactor` is invalid,
   * or an attempt to define configuration is made after learning starts.
  */
  var defineConfig = function ( cfg ) {
    if ( learned ) {
      throw Error( 'winkNBTC: config must be defined before learning starts!' );
    }
    if ( !helpers.object.isObject( cfg ) ) {
      throw Error( 'winkNBTC: config must be an object, instead found: ' + ( typeof cfg ) );
    }
    config.considerOnlyPresence = ( typeof cfg.considerOnlyPresence === 'boolean' ) ?
                                    cfg.considerOnlyPresence : false;

    // If smoothing factor is undefined set it to laplace add+1 smoothing.
    var sf = ( cfg.smoothingFactor === undefined ) ? 1 : parseFloat( cfg.smoothingFactor );
    // Throw error for a value beyond 0-1 or NaN.
    if ( isNaN( sf ) || ( sf < 0 ) || ( sf > 1 ) ) {
      throw Error( 'winkNBTC: smoothing factor must be a number between 0 & 1, instead found: ' + JSON.stringify( sf ) );
    }
    // All good, set smoothingFactor as `sf`.
    config.smoothingFactor = sf;
    return true;
  }; // defineConfig()

  // #### Define Prep Tasks

  // It sets the `pTasks` and returns length of `pTask` array.
  /**
   * Defines the text preparation `tasks` to transform raw incoming
   * text into tokens required during
   * [`learn()`](#learn), [`evaluate()`](#evaluate) and [`predict()`](#predict) operations.
   * The `tasks` should be an array of functions;
   * using these function a simple pipeline is built to serially transform the
   * input to the output. A single helper function for preparing text is available that (a) tokenizes,
   * (b) removes punctuations, symbols, numerals, URLs, stop words and (c) stems.
   *
   * @method NaiveBayesTextClassifier#definePrepTasks
   * @param {function[]} tasks the first function
   * in this array must accept a string as input and the last function must
   * return tokens i.e. array of strings. Please refer to example.
   * @return {number} The number of functions in `task` array.
   * @example
   * // Load wink NLP utilities
   * var prepText = require( 'wink-naive-bayes-text-classifier/src/prep-text.js' );
   * // Define the text preparation tasks.
   * myClassifier.definePrepTasks( [ prepText ] );
   * // -> 1
   * @throws Error if `tasks` is not an array of functions.
  */
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
  /**
   *
   * Learns from the example pair of `input` and its `label`.
   *
   * @method NaiveBayesTextClassifier#learn
   * @param {string|string[]} input if it is a string, then [`definePrepTasks()`](#definePrepTasks)
   * must be called before learning so that `input` string is transformed
   * into tokens on the fly.
   * @param {string} label of class to which `input` belongs.
   * @return {boolean} Always true.
   * @example
   * myClassifier.learn( 'I need loan for a new vehicle', 'autoloan' );
   * // -> true
   * @throws Error if learnings have been already [consolidated](#consolidate).
  */
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
  /**
   *
   * Consolidates the learning. It is a prerequisite for [`evaluate()`](#evaluate)
   * and/or [`predict()`](#predict).
   *
   * @method NaiveBayesTextClassifier#consolidate
   * @return {boolean} Always true.
   * @example
   * myClassifier.consolidate();
   * // -> true
   * @throws Error if training data belongs to only a single class label or
   * the training data is too small for learning.
  */
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

  // #### compute odds

  // Computes odds for every **label** for the given `input`, provided learnings
  // have been consolidated. They are sorted in descending order of their odds.
  // It throws error if the learnings have not been consolidated. Note, the odds
  // is actually the **log2** of odds.
  /**
   * Computes the log base-2 of odds of every label for the input; and returns
   * the array of `[ label, odds ]` in descending order of odds.
   *
   * @method NaiveBayesTextClassifier#computeOdds
   * @param {String|String[]} input is either text or tokens determined by the
   * choice of [`preparatory tasks`](#definePrepTasks).
   * @return {array[]} Array of `[ label, odds ]` in descending order of odds.
   * @example
   * myClassifier.computeOdds( 'I want to pay my car loan early' );
   * // -> [
   *         [ 'prepay', 6.169686751688911 ],
   *         [ 'autoloan', -6.169686751688911 ]
   *       ]
  */
  var computeOdds = function ( input ) {
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
    return ( ( allOdds[ 0 ][ 1 ] ) ? allOdds : [ [ unknown, 0 ] ] );
  };

  // #### Predict

  // Predicts the potential **label** for the given `input`, provided learnings
  // have been consolidated. If all the `input` tokens have never been seen
  // in past (i.e. absent in learnings), then the predicted label is `unknown`.
  // It throws error if the learnings have not been consolidated.
  /**
   *
   * Predicts the class label for the `input`. If it is unable to predict then it
   * returns a value **`unknown`**.
   *
   * @method NaiveBayesTextClassifier#predict
   * @param {String|String[]} input is either text or tokens determined by the
   * choice of [`preparatory tasks`](#definePrepTasks).
   * @return {String} The predicted class label for the `input`.
   * @example
   * myClassifier.predict( 'I want to pay my car loan early' );
   * // -> prepay
  */
  var predict = function ( input ) {
    // Contains label & the corresponding odds pairs.
    var allOdds = computeOdds( input );
    return ( allOdds[ 0 ][ 0 ] );
  }; // predict()

  // #### Stats

  /**
   * Returns basic stats of learning in terms of count of samples under
   * each label, total words, and the size of vocabulary.
   *
   * @method NaiveBayesTextClassifier#stats
   * @return {object} An object containing count of samples under
   * each label, total words, and the size of vocabulary.
   * @example
   * myClassifier.stats();
   * // -> {
   * //      labelWiseSamples: {
   * //        autoloan: 5,
   * //        prepay: 4
   * //      },
   * //      labelWiseWords: {
   * //        autoloan: 36,
   * //        prepay: 26
   * //      },
   * //      vocabulary: 24
   * //    };
  */
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
  }; // stats()

  // #### Export JSON

  // Returns the learnings, without any consolidation check, in JSON format.
  /**
   * Exports the learning as a JSON, which may be saved as a text file for
   * later use via [`importJSON()`](#importjson).
   *
   * @method NaiveBayesTextClassifier#exportJSON
   * @return {string} Learning in JSON format.
   * @example
   * myClassifier.exportJSON();
   * // returns JSON.
  */
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
  /**
   * It completely resets the classifier by re-initializing all the learning
   * related variables, except the preparatory tasks. It is useful during
   * cross fold-validation.
   *
   * @method NaiveBayesTextClassifier#reset
   * @return {boolean} Always true.
   * @example
   * myClassifier.reset();
   * // -> true
  */
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
  /**
   * Imports an existing JSON learning for prediction.
   * It is essential to [`definePrepTasks()`]()#definepreptasks and
   * [`consolidate()`](#consolidate) before attempting to predict.
   *
   * @method NaiveBayesTextClassifier#importJSON
   * @param {JSON} json containing learnings in as exported by [`exportJSON`](#exportjson).
   * @return {boolean} Always true.
   * @throws Error if `json` is invalid.
  */
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
  // prediction fails (unknown), then it does not uppdate
  // the confusion matrix and returns `false`; otherwise it updates the matrix
  // and returns `true`.
  /**
   *
   * Evaluates the learning against a test data set.
   * The `input` is used to predict the class label, which is compared with the
   * actual class `label` to populate confusion matrix incrementally.
   *
   * @method NaiveBayesTextClassifier#evaluate
   * @param {String|String[]} input is either text or tokens determined by the
   * choice of [`preparatory tasks`](#definePrepTasks).
   * @param {string} label of class to which `input` belongs.
   * @return {boolean} Always true.
   * @example
   * myClassifier.evaluate( 'can i close my loan', 'prepay' );
   * // -> true
  */
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
  /**
   *
   * Computes a detailed metrics consisting of macro-averaged precision, recall
   * and f-measure along with their label-wise values and the confusion matrix.
   *
   * @method NaiveBayesTextClassifier#metrics
   * @return {object} Detailed metrics.
   * @example
   * // Assuming that evaluation has been already carried out
   * JSON.stringify( myClassifier.metrics(), null, 2 );
   * // -> {
   * //      "avgPrecision": 0.75,
   * //      "avgRecall": 0.75,
   * //      "avgFMeasure": 0.6667,
   * //      "details": {
   * //        "confusionMatrix": {
   * //          "prepay": {
   * //            "prepay": 1,
   * //            "autoloan": 1
   * //          },
   * //          "autoloan": {
   * //            "prepay": 0,
   * //            "autoloan": 1
   * //          }
   * //        },
   * //        "precision": {
   * //          "prepay": 0.5,
   * //          "autoloan": 1
   * //        },
   * //        "recall": {
   * //          "prepay": 1,
   * //          "autoloan": 0.5
   * //        },
   * //        "fmeasure": {
   * //          "prepay": 0.6667,
   * //          "autoloan": 0.6667
   * //        }
   * //      }
   * //    }
   * @throws Error if attempt to generate metrics is made prior to proper evaluation.
  */
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
  methods.computeOdds = computeOdds;
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
}; // naiveBayesTextClassifier()

// Export textNBC.
module.exports = naiveBayesTextClassifier;
