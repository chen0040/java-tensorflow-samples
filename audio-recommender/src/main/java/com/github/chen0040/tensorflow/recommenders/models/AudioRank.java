package com.github.chen0040.tensorflow.recommenders.models;

import com.github.chen0040.tensorflow.search.models.AudioSearchEntry;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class AudioRank {
    private String audioPath;
    private double distance1;
    private double distance2;
    private double meanDistance;
    private float[] features;

    public AudioSearchEntry toSearchEntry() {
        return new AudioSearchEntry(audioPath, features).withDistance(meanDistance);
    }
}
