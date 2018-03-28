package com.github.chen0040.tensorflow.classifiers.sentiment.utils;

import com.github.chen0040.tensorflow.classifiers.sentiment.models.TextModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ResourceUtils {
    private static final Logger logger = LoggerFactory.getLogger(ResourceUtils.class);

    public static InputStream getInputStream(String file_path){
        return ResourceUtils.class.getClassLoader().getResourceAsStream(file_path);
    }

    public static TextModel getTextModel(InputStream inputStream) {
        int maxLen = 0;
        Map<String, Integer> word2idx = new HashMap<>();
        Map<Integer, String> idx2label = new HashMap<>();
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            boolean firstLine = true;
            String line = reader.readLine();

            while(line != null){
                if(firstLine) {
                    firstLine = false;
                    maxLen = Integer.parseInt(line.trim());
                } else {
                    if(line.startsWith("label")) {
                        String[] parts = line.trim().split("\t");
                        String word = parts[1];
                        int index = Integer.parseInt(parts[2]);
                        idx2label.put(index, word);
                    } else {
                        String[] parts = line.trim().split("\t");
                        String word = parts[0];
                        int index = Integer.parseInt(parts[1]);
                        word2idx.put(word, index);
                    }

                }
                line = reader.readLine();
            }
        } catch(Exception ex){
                logger.error("Failed to get text model", ex);
        }
        return new TextModel(maxLen, word2idx, idx2label);
    }

    public static TextModel getTextModel(String file_path) {
        return getTextModel(getInputStream(file_path));
    }

    public static List<String> getLines(String  path) throws IOException {
        InputStream is = getInputStream(path);

        List<String> result = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        String line = reader.readLine();
        while(line != null) {
            result.add(line);
            line = reader.readLine();
        }
        return result;
    }

    public static void main(String[] args) throws IOException {
        TextModel model = getTextModel("tf_models/lstm_softmax.csv");
        System.out.println("max_len: " + model.getMaxLen());
        for(Map.Entry<String, Integer> entry : model.getWord2idx().entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        List<String> lines = getLines("data/umich-sentiment-train.txt");
        for(String line : lines) {
            System.out.println(line);
        }
    }


}
