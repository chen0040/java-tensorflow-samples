package com.github.chen0040.tensorflow.classifiers.audio.models;

import java.awt.image.BufferedImage;
import java.io.File;

public interface AudioClassifier extends AudioEncoder {
    String[] labels = new String[]{

            "blues", "classical", "country", "disco", "hiphop", "jazz", "metal",
            "pop", "reggae", "rock"

    };



    String predict_image(BufferedImage image);

    String predict_audio(File f);

    default String predict_image(BufferedImage image, int imgWidth, int imgHeight) {

        float[] predicted = encode_image(image, imgWidth, imgHeight);
        if(predicted != null) {
            int nlabels = predicted.length;
            int argmax = 0;
            float max = predicted[0];
            for (int i = 1; i < nlabels; ++i) {
                if (max < predicted[i]) {
                    max = predicted[i];
                    argmax = i;
                }
            }

            return labels[argmax];
        }

        return "unknown";
    }
}
