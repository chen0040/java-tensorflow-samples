package com.github.chen0040.tensorflow.classifiers.utils;

import org.tensorflow.Tensor;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;

public class TensorUtils {
    public static Tensor<Float> getImageTensor(BufferedImage image, int imgWidth, int imgHeight) {

        final int channels = 3;
        // Generate image file to array
        int index = 0;
        FloatBuffer fb = FloatBuffer.allocate(imgWidth * imgHeight * channels);
        // Convert image file to multi-dimension array

        for (int row = 0; row < imgHeight; row++) {
            for (int column = 0; column < imgWidth; column++) {
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

        return Tensor.create(new long[]{1, imgWidth, imgHeight, channels}, fb);
    }

    public static Tensor<Float> getImageTensor(InputStream inputStream) throws IOException {
        BufferedImage img = ImageIO.read(inputStream);
        int imgWidth = img.getWidth();
        int imgHeight = img.getHeight();
        return getImageTensor(img, imgWidth, imgHeight);
    }
}
