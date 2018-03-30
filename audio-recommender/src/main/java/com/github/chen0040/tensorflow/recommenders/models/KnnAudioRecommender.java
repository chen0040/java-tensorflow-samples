package com.github.chen0040.tensorflow.recommenders.models;

import com.github.chen0040.tensorflow.search.models.AudioSearchEngine;
import com.github.chen0040.tensorflow.search.models.AudioSearchEntry;

import java.io.File;
import java.util.*;

public class KnnAudioRecommender extends AudioSearchEngine implements AudioRecommender {
    public List<AudioSearchEntry> recommends(List<AudioMemo> userHistory, int k) {
        userHistory.sort((a, b) -> Long.compare(b.getEventTime(), a.getEventTime()));
        List<String> mostRecentHistory = new ArrayList<>();
        if(userHistory.size() > 60) {
            for(int i=0; i < 20; ++i) {
                AudioMemo memo = userHistory.get(i);
                String filePath = memo.getAudioPath();
                if(mostRecentHistory.indexOf(filePath) < 0) {
                    mostRecentHistory.add(filePath);
                }
            }
        } else if(userHistory.size() > 30) {
            for(int i=0; i < 10; ++i) {
                AudioMemo memo = userHistory.get(i);
                String filePath = memo.getAudioPath();
                if(mostRecentHistory.indexOf(filePath) < 0) {
                    mostRecentHistory.add(filePath);
                }
            }
        }

        Map<String, AudioRank> ranks = new HashMap<>();

        for(int i=0; i < mostRecentHistory.size(); ++i){
            String filePath = mostRecentHistory.get(i);
            double distance2 = (double)mostRecentHistory.size() / (i+1.0);

            File file = new File(filePath);
            List<AudioSearchEntry> similar_songs = query(file, 0, 10, true);

            for(AudioSearchEntry entry : similar_songs){
                double distance1 = Math.sqrt(entry.getDistance());

                double distance_mean = (distance1 * distance2) / (distance1 + distance2);

                AudioRank newRank = new AudioRank();
                newRank.setAudioPath(entry.getPath());
                newRank.setFeatures(entry.getFeatures());
                newRank.setDistance1(distance1);
                newRank.setDistance2(distance2);
                newRank.setMeanDistance(distance_mean);

                if(ranks.containsKey(entry.getPath())){
                    AudioRank rank = ranks.get(entry.getPath());
                    if(rank.getMeanDistance() < distance_mean){
                        ranks.put(entry.getPath(), newRank);
                    }
                } else {
                    ranks.put(entry.getPath(), newRank);
                }
            }
        }

        List<AudioRank> ranked = new ArrayList<>(ranks.values());
        ranked.sort(Comparator.comparingDouble(AudioRank::getMeanDistance));

        List<AudioSearchEntry> result = new ArrayList<>();
        for(int i=0; i < k; ++i){
            if(i < ranked.size()){
                result.add(ranked.get(i).toSearchEntry());
            }
        }
        return result;
    }
}
