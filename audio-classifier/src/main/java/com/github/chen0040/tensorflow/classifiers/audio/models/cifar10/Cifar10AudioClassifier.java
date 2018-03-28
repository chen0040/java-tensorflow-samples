package com.github.chen0040.tensorflow.classifiers.audio.models.cifar10;

import com.github.chen0040.tensorflow.classifiers.audio.utils.MelSpectrogram;
import com.github.chen0040.tensorflow.classifiers.audio.utils.consts.MelSpectrogramDimension;
import com.github.chen0040.tensorflow.classifiers.audio.utils.ImageUtils;
import com.github.chen0040.tensorflow.classifiers.audio.utils.InputStreamUtils;
import com.github.chen0040.tensorflow.classifiers.audio.utils.TensorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class Cifar10AudioClassifier implements AutoCloseable {

    private Graph graph = new Graph();
    public Cifar10AudioClassifier() {

    }

    public void load_model(InputStream inputStream) throws IOException {
        byte[] bytes = InputStreamUtils.getBytes(inputStream);
        graph.importGraphDef(bytes);

    }

    private static final String[] labels = new String[]{

        "blues", "classical", "country", "disco", "hiphop", "jazz", "metal",
                "pop", "reggae", "rock"

    };

    private static final Logger logger = LoggerFactory.getLogger(Cifar10AudioClassifier.class);

    public String predict_image(BufferedImage image) {
        return predict_image(image, MelSpectrogramDimension.Width,
                MelSpectrogramDimension.Height);
    }

    public String predict_image(BufferedImage image, int imgWidth, int imgHeight){

        image = ImageUtils.resizeImage(image, imgWidth, imgHeight);

        Tensor<Float> imageTensor = TensorUtils.getImageTensor(image, imgWidth, imgHeight);



        try (Session sess = new Session(graph);
             Tensor<Float> result =
                     sess.runner().feed("conv2d_1_input:0", imageTensor)
                             //.feed("dropout_1/keras_learning_phase:0", Tensor.create(false))
                             .fetch("output_node0:0").run().get(0).expect(Float.class)) {
            final long[] rshape = result.shape();
            if (result.numDimensions() != 2 || rshape[0] != 1) {
                throw new RuntimeException(
                        String.format(
                                "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                Arrays.toString(rshape)));
            }
            int nlabels = (int) rshape[1];
            float[] predicted = result.copyTo(new float[1][nlabels])[0];
            int argmax = 0;
            float max = predicted[0];
            for(int i=1; i < nlabels; ++i) {
                if(max < predicted[i]) {
                    max = predicted[i];
                    argmax = i;
                }
            }

            return labels[argmax];
        } catch(Exception ex) {
            logger.error("Failed to predict image", ex);
            ex.printStackTrace();
        }

        return "unknown";
    }

    @Override
    public void close() throws Exception {
        if(graph != null) {
            graph.close();
            graph = null;
        }
    }

    public String predict_audio(File f) {
        MelSpectrogram melGram = new MelSpectrogram();
        melGram.setOutputFrameWidth(MelSpectrogramDimension.Width);
        melGram.setOutputFrameHeight(MelSpectrogramDimension.Height);


        try {
            BufferedImage image = melGram.convertAudio(f);
            return predict_image(image);
        } catch (IOException | UnsupportedAudioFileException | LineUnavailableException e) {
            e.printStackTrace();
        }

        return "NA";
    }
}
