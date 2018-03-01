package com.github.chen0040.tflite.hello;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;

public class Cifar10ImageClassifier {

    private static final String[] labels = new String[]{
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
    };

    private static final Logger logger = LoggerFactory.getLogger(Cifar10ImageClassifier.class);

    private static Tensor<Float> getImage(String image_path) {

        // Generate image file to array
        int index = 0;
        FloatBuffer fb = FloatBuffer.allocate(32 * 32 * 3);
        // Convert image file to multi-dimension array
        InputStream is = ResourceUtils.getInputStream(image_path);
        try {
            BufferedImage image = ImageIO.read(is);

            int imageWidth = 32;
            int imageHeight = 32;
            for (int row = 0; row < imageHeight; row++) {
                for (int column = 0; column < imageWidth; column++) {
                    int pixel = image.getRGB(column, row);

                    float red = (pixel >> 16) & 0xff;
                    float green = (pixel >> 8) & 0xff;
                    float blue = pixel & 0xff;
                    red = red / 255.0f;
                    green = green / 255.0f;
                    blue = blue / 255.0f;
                    fb.put(index++, red);
                    fb.put(index++, green);
                    fb.put(index++, blue);
                }
            }
        } catch (IOException e) {
            logger.info("Failed to get the image tensor input", e);
            System.exit(1);
        }


        return Tensor.create(new long[]{1, 32, 32, 3}, fb);
    }

    public static void main(String[] args) {


        String[] image_names = new String[] {
                "airplane1",
                "airplane2",
                "airplane3",
                "automobile1",
                "automobile2",
                "automobile3",
        };

        for(String image_name :image_names) {
            String image_path = "images/cifar10/" + image_name + ".png";
            String predicted_label = predict_image(image_path);
            logger.info("predicted class for {}: {}", image_name, predicted_label);
        }




    }

    private static String predict_image(String image_path){
        Tensor<Float> image = getImage(image_path);

        try(Graph g = new Graph()) {
            byte[] bytes = ResourceUtils.getBytes("tf_models/cnn_cifar10.pb");
            g.importGraphDef(bytes);
            try (Session sess = new Session(g);
                 Tensor<Float> result =
                         sess.runner().feed("conv2d_1_input:0", image)
                                 .feed("dropout_1/keras_learning_phase:0", Tensor.create(false))
                                 .fetch("output_node0:0").run().get(0).expect(Float.class)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                float[] predicted = result.copyTo(new float[1][nlabels])[0];
                int argmax = 0;
                float max = predicted[0];
                for(int i=1; i < nlabels; ++i) {
                    if(max < predicted[i]) {
                        max = predicted[i];
                        argmax = i;
                    }
                }

                return labels[argmax];
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return "unknown";
    }
}
