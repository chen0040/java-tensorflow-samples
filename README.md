# java-tensorflow-samples

Java sample codes on how to load tensorflow pretrained model file and predict based on these pretrained model files

# Usage

### Image Classification using Cifar10

Below show the [demo codes](image-classifier/src/main/java/com/github/chen0040/tflite/classifiers/demo/Cifar10ImageClassifierDemo.java)
of the  Cifar10ImageClassifier which loads the [cnn_cifar10.pb](image-classifier/src/main/resources/tf_models/cnn_cifar10.pb)
tensorflow model file, and uses it to do image classification:

```java
package com.github.chen0040.tflite.classifiers.demo;

import com.github.chen0040.tflite.classifiers.utils.ResourceUtils;
import com.github.chen0040.tflite.classifiers.cifar10.Cifar10ImageClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;

public class Cifar10ImageClassifierDemo {

    private static final Logger logger = LoggerFactory.getLogger(Cifar10ImageClassifierDemo.class);

    public static void main(String[] args) throws IOException {


        InputStream inputStream = ResourceUtils.getInputStream("tf_models/cnn_cifar10.pb");
        Cifar10ImageClassifier classifier = new Cifar10ImageClassifier();
        classifier.load_model(inputStream);

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
```

### Image Classification using Inception 

Below show the [demo codes](image-classifier/src/main/java/com/github/chen0040/tflite/classifiers/demo/InceptionImageClassifierDemo.java)
of the  Cifar10ImageClassifier which loads the [tensorflow_inception_graph.pb](image-classifier/src/main/resources/tf_models/tensorflow_inception_graph.pb)
tensorflow model file, and uses it to do image classification:

```java
package com.github.chen0040.tflite.classifiers.demo;

import com.github.chen0040.tflite.classifiers.inception.InceptionImageClassifier;
import com.github.chen0040.tflite.classifiers.utils.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;

public class InceptionImageClassifierDemo {

    private static final Logger logger = LoggerFactory.getLogger(InceptionImageClassifierDemo.class);

    public static void main(String[] args) throws IOException {


        InceptionImageClassifier classifier = new InceptionImageClassifier();
        classifier.load_model(ResourceUtils.getInputStream("tf_models/tensorflow_inception_graph.pb"));
        classifier.load_labels(ResourceUtils.getInputStream("tf_models/imagenet_comp_graph_label_strings.txt"));

        String[] image_names = new String[] {
                "tiger",
                "lion"
        };

        for(String image_name :image_names) {
            String image_path = "images/inception/" + image_name + ".jpg";
            BufferedImage img = ResourceUtils.getImage(image_path);
            String predicted_label = classifier.predict_image(img);
            logger.info("predicted class for {}: {}", image_name, predicted_label);
        }
    }
}
```


