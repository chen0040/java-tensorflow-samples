package com.github.chen0040.tensorflow.classifiers.sentiment.models;

import com.github.chen0040.tensorflow.classifiers.images.utils.InputStreamUtils;
import com.github.chen0040.tensorflow.classifiers.sentiment.utils.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.io.InputStream;

public abstract class SentimentClassifier implements AutoCloseable {

    private Graph graph = new Graph();
    private TextModel textModel  = null;

    private static final Logger logger = LoggerFactory.getLogger(SentimentClassifier.class);

    public void load_model(InputStream inputStream) throws IOException {
        byte[] bytes = InputStreamUtils.getBytes(inputStream);
        graph.importGraphDef(bytes);
    }

    public void load_vocab(InputStream inputStream) {
        textModel = ResourceUtils.getTextModel(inputStream);
    }

    public String predict_label(String text) {
        float[] predicted = predict(text);
        int argmax = 0;
        float max = predicted[0];
        for(int i=1; i < predicted.length; ++i) {
            if(predicted[i] > max) {
                max = predicted[i];
                argmax = i;
            }
        }
        return textModel.toLabel(argmax);
    }

    public float[] predict(String text) {

        Tensor<Float> textTensor = textModel.toTensor(text);
        try {
            Session sess = new Session(graph);
            Tensor<Float> result = runPredict(textTensor, sess);
            try {
                long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) throw new RuntimeException(String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s", java.util.Arrays.toString(rshape)));
                int nlabels = (int)rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            } catch(Exception ex) {
                    logger.error("Failed to predict image", ex);
            } finally {
                if (sess != null) sess.close();
                if (result != null) result.close();
            }
        } catch(Exception ex2) {
            ex2.printStackTrace();
        }
        return new float[2];
    }

    public abstract Tensor<Float> runPredict(Tensor<Float> textTensor, Session sess);

    @Override
    public void close() {
        if (graph != null) {
            graph.close();
            graph = null;
        }
    }
}
