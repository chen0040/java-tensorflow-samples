package com.github.chen0040.tensorflow.classifiers.utils;

import java.awt.*;
import java.awt.image.BufferedImage;

public class ImageUtils {
    public static BufferedImage resizeImage(BufferedImage img, int imgWidth, int imgHeight) {
        if(img.getWidth() != imgWidth || img.getHeight() != imgHeight) {
            Image newImg = img.getScaledInstance(imgWidth, imgHeight, Image.SCALE_SMOOTH);
            BufferedImage newBufferedImg = new BufferedImage(newImg.getWidth(null),
                    newImg.getHeight(null),
                    BufferedImage.TYPE_INT_RGB);
            newBufferedImg.getGraphics().drawImage(newImg, 0, 0, null);
            return newBufferedImg;
        }
        return img;
    }
}
