package com.github.chen0040.tensorflow.classifiers.utils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class InputStreamUtils {
    public static byte[] getBytes(InputStream is) throws IOException {
        ByteArrayOutputStream mem = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int len = 0;
        while((len = is.read(buffer, 0, 1024)) > 0){
            mem.write(buffer, 0, len);
        }
        return mem.toByteArray();
    }
}
