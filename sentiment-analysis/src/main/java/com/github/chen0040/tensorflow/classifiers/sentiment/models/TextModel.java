package com.github.chen0040.tensorflow.classifiers.sentiment.models;

import lombok.Getter;
import lombok.Setter;
import org.tensorflow.Tensor;

import java.nio.FloatBuffer;
import java.util.Map;

@Getter
@Setter
public class TextModel {
    private final int maxLen;
    private final Map<String, Integer> word2idx;
    private final Map<Integer, String> idx2label;

    public TextModel(int maxLen, Map<String, Integer> word2idx, Map<Integer, String> idx2label) {
        this.maxLen = maxLen;
        this.word2idx = word2idx;
        this.idx2label = idx2label;
    }

    public Tensor<Float> toTensor(String text) {
        FloatBuffer ib = FloatBuffer.allocate(maxLen);

        int index = 0;
        String[] parts = text.toLowerCase().split(" ");
        int textLen = Math.min(maxLen, parts.length);
        for (String word : parts) {
            int idx = 0;
            if (word2idx.containsKey(word)) {
                idx = word2idx.get(word);
            }
            ib.put(maxLen - textLen + index, idx);
            index += 1;
        }

        return Tensor.create(new long[] {1, maxLen},ib);

    }

    public String toLabel(int idx) {
        return idx2label.getOrDefault(idx, "");
    }
}
