// https://www.tensorflow.org/js/guide/models_and_layers
// https://www.tensorflow.org/js/guide/tensors_operations
// https://www.tensorflow.org/js/guide/train_models

function plotGroundTruth(data) {
  let data0 = data.filter(x => x[0]*x[0] + x[1]*x[1] < 1);
  let data1 = data.filter(x => x[0]*x[0] + x[1]*x[1] >= 1);
  let trace1 = {x:data0.map(row=>row[0]), y:data0.map(row=>row[1]), mode:'markers', type:'scatter'};
  let trace2 = {x:data1.map(row=>row[0]), y:data1.map(row=>row[1]), mode:'markers', type:'scatter'};
  let layout = {title:"Ground Truth Labels",
                autosize: false,
                width: 600,
                height: 600};
  Plotly.newPlot("gtPlot", [trace1, trace2], layout);
}


/**
 * Plot the hidden layer
 * @param {list} Y Mapped points in 3D
 * @param {list} y Labels of points
 */
function plot3DDataMap(Y, y) {
  let data0 = Y.filter((yy, idx) => y[idx] == 0);
  let data1 = Y.filter((yy, idx) => y[idx] == 1);
  console.log(data0);
  let trace1 = {x:data0.map(row=>row[0]), y:data0.map(row=>row[1]), z:data0.map(row=>row[2]), mode:'markers', marker:{size:3},type:'scatter3d'};
  let trace2 = {x:data1.map(row=>row[0]), y:data1.map(row=>row[1]), z:data1.map(row=>row[2]), mode:'markers', marker:{size:3}, type:'scatter3d'};
  let layout = {title:"Hidden Layer",
                autosize: false,
                width: 600,
                height: 600};
  Plotly.newPlot("hiddenLayer", [trace1, trace2], layout);
}

let model = null;
let xs = tf.randomNormal([200, 2]);
xs.array().then(data => {
  let ys = [];
  xs = [];
  for (let i = 0; i < data.length; i++) {
    let dSqr = data[i][0]*data[i][0] + data[i][1]*data[i][1];
    if (dSqr < 1) {
      xs.push(data[i]);
      ys.push(0); // Label as 0 if it's within a radius of 1
    }
    else {
      xs.push([data[i][0]*1.5, data[i][1]*1.5]);
      ys.push(1);
    }
    data[i] = xs[i];
  }
  let ysorig = ys;
  xs = tf.tensor(xs);
  ys = tf.tensor(ys);
  plotGroundTruth(data);

  // Create an arbitrary graph of layers, by connecting them
  // via the apply() method.
  const input = tf.input({shape: [2]});
  const dense1 = tf.layers.dense({units: 10, activation: 'relu', useBias:true}).apply(input);
  const dense2 = tf.layers.dense({units: 1, activation: 'relu', useBias:true}).apply(dense1);
  model = tf.model({inputs: input, outputs: dense2});
  //tfvis.show.modelSummary({name: 'Model Summary'}, model);

  model.compile({loss: tf.losses.sigmoidCrossEntropy,
                optimizer: tf.train.adam()});

  const onEpochEnd = epoch => {
      model.predict(xs).array().then(d => {
        // Show predictions on input data
        let trace = {x:data.map(row=>row[0]), y:data.map(row=>row[1]), mode:'markers', type:'scatter', marker:{color:d.map(a => 100*a)}};
        let layout = {title:"Classifications Epoch " + epoch,
                      autosize: false,
                      width: 600,
                      height: 600};
        Plotly.newPlot("classificationPlot", [trace], layout);

        // Update output of hidden layer at every 25 iterations
        if (epoch%25 == 0) {
          model.layers[1].apply(xs).array().then(Y => {
            plot3DDataMap(Y, ysorig);
          });
        }

        // Show loss
        trace = {y:model.history.history.loss};
        layout = {title:"Loss Epoch " + epoch,
                      autosize: false,
                      width: 600,
                      height: 600};
        Plotly.newPlot("lossPlot", [trace], layout);
      });
  }

  // Train the model using the data, and test on the same data for now
  model.fit(xs, ys, 
    {epochs: 1000,
      callbacks: {
        "onEpochEnd":onEpochEnd
      }
    }
  );


});