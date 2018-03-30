package com.github.chen0040.tensorflow.recommenders.models;

import com.github.chen0040.tensorflow.recommenders.models.AudioMemo;
import com.github.chen0040.tensorflow.search.models.AudioSearchEntry;

import java.util.List;

public interface AudioRecommender {
    List<AudioSearchEntry> recommends(List<AudioMemo> userHistory, int k);
}
