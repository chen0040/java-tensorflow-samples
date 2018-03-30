package com.github.chen0040.tensorflow.classifiers.audio.utils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class FileUtils {
    public static List<String> getAudioFiles() {
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
}
