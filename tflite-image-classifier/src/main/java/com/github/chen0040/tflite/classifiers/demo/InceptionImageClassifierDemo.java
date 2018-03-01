package com.github.chen0040.tflite.classifiers.demo;

import com.github.chen0040.tflite.classifiers.cifar10.Cifar10ImageClassifier;
import com.github.chen0040.tflite.classifiers.inception.InceptionImageClassifier;
import com.github.chen0040.tflite.classifiers.utils.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;

public class InceptionImageClassifierDemo {

    private static final Logger logger = LoggerFactory.getLogger(InceptionImageClassifierDemo.class);

    public static void main(String[] args) throws IOException {


        InceptionImageClassifier classifier = new InceptionImageClassifier();
        classifier.load_model();

        String[] image_names = new String[] {
                "airplane1",
                "airplane2",
                "airplane3",
                "automobile1",
                "automobile2",
                "automobile3",
                "bird1",
                "bird2",
                "bird3",
                "cat1",
                "cat2",
                "cat3"
        };

        for(String image_name :image_names) {
            String image_path = "images/cifar10/" + image_name + ".png";
            BufferedImage img = ResourceUtils.getImage(image_path);
            String predicted_label = classifier.predict_image(img);
            logger.info("predicted class for {}: {}", image_name, predicted_label);
        }
    }
}
