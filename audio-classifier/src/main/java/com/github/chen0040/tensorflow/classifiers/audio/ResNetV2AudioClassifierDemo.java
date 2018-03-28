package com.github.chen0040.tensorflow.classifiers.audio;

import com.github.chen0040.tensorflow.classifiers.audio.models.resnet.ResNetV2AudioClassifier;
import com.github.chen0040.tensorflow.classifiers.audio.utils.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ResNetV2AudioClassifierDemo {

    private static final Logger logger = LoggerFactory.getLogger(ResNetV2AudioClassifierDemo.class);

    private static List<String> getAudioFiles() {
        List<String> result = new ArrayList<>();
        File dir = new File("music_samples");
        System.out.println(dir.getAbsolutePath());
        if (dir.isDirectory()) {

            for (File f : dir.listFiles()) {
                String file_path = f.getAbsolutePath();
                if (file_path.endsWith("au")) {
                    result.add(file_path);

                }
            }
        }

        return result;
    }

    public static void main(String[] args) throws IOException {
        InputStream inputStream = ResourceUtils.getInputStream("tf_models/resnet-v2.pb");
        ResNetV2AudioClassifier classifier = new ResNetV2AudioClassifier();
        classifier.load_model(inputStream);

        List<String> paths = getAudioFiles();

        Collections.shuffle(paths);

        for (String path : paths) {
            System.out.println("Predicting " + path + " ...");
            File f = new File(path);
            String label = classifier.predict_audio(f);

            System.out.println("Predicted: " + label);
        }
    }
}
