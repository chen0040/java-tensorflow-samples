package com.github.chen0040.tensorflow.classifiers.utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;

public class ResourceUtils {
    public static InputStream getInputStream(String file_path) {
        return ResourceUtils.class.getClassLoader().getResourceAsStream(file_path);
    }

    public static BufferedImage getImage(String file_path) throws IOException {
        return ImageIO.read(getInputStream(file_path));
    }
}
