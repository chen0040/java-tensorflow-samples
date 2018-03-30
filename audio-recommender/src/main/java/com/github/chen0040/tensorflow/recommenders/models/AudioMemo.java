package com.github.chen0040.tensorflow.recommenders.models;

import lombok.Getter;
import lombok.Setter;

import java.util.Date;

@Getter
@Setter
public class AudioMemo {
    private String audioPath;
    private long eventTime;

    public AudioMemo() {

    }

    public AudioMemo(String filePath){
        this.audioPath = filePath;
        eventTime = new Date().getTime();
    }
}
