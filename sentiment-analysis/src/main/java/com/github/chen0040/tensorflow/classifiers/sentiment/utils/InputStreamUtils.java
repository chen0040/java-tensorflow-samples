package com.github.chen0040.tensorflow.classifiers.sentiment.utils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class InputStreamUtils {
    public static byte[] getBytes(InputStream is) throws IOException {
        ByteArrayOutputStream mem = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int len = is.read(buffer, 0, 1024);
        while (len > 0) {
            mem.write(buffer, 0, len);
            len = is.read(buffer, 0, 1024);
        }
        return mem.toByteArray();
    }
}
