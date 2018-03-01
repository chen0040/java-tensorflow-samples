package com.github.chen0040.tflite.hello;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class ResourceUtils {
    public static byte[] getBytes(String file_path) throws IOException {
        InputStream is = ResourceUtils.class.getClassLoader().getResourceAsStream(file_path);
        ByteArrayOutputStream mem = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int len = 0;
        while((len = is.read(buffer, 0, 1024)) > 0){
            mem.write(buffer, 0, len);
        }
        return mem.toByteArray();
    }

    public static InputStream getInputStream(String file_path) {
        return ResourceUtils.class.getClassLoader().getResourceAsStream(file_path);
    }
}
