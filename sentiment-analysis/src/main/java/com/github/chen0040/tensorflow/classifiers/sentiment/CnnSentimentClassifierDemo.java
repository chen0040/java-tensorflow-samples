package com.github.chen0040.tensorflow.classifiers.sentiment;

import com.github.chen0040.tensorflow.classifiers.sentiment.models.CnnSentimentClassifier;
import com.github.chen0040.tensorflow.classifiers.sentiment.utils.ResourceUtils;

import java.io.IOException;
import java.util.List;

public class CnnSentimentClassifierDemo {
    public static void main(String[] args) throws IOException {
        CnnSentimentClassifier classifier = new CnnSentimentClassifier();
        classifier.load_model(ResourceUtils.getInputStream("tf_models/wordvec_cnn.pb"));
        classifier.load_vocab(ResourceUtils.getInputStream("tf_models/wordvec_cnn.csv"));

        List<String> lines = ResourceUtils.getLines("data/umich-sentiment-train.txt");
        for(String line : lines){
            String label = line.split("\t")[0];
            String text = line.split("\t")[1];
            float[] predicted = classifier.predict(text);
            String predicted_label = classifier.predict_label(text);
            System.out.println(text);
            System.out.println("Outcome: " + predicted[0] + ", " + predicted[1]);
            System.out.println("Predicted: " + predicted_label + " Actual: " + label);
        }
    }
}
