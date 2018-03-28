package com.github.chen0040.tensorflow.classifiers.sentiment.models;

import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class BidirectionalLstmSentimentClassifier extends SentimentClassifier {
    @Override
    public Tensor<Float> runPredict(Tensor<Float> textTensor, Session sess) {
        return sess.runner()
                .feed("embedding_1_input:0", textTensor)
                .feed("spatial_dropout1d_1/keras_learning_phase:0", Tensor.create(false))
                .fetch("output_node0:0").run().get(0).expect(Float.class);
    }

}
