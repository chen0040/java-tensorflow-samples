package com.github.chen0040.tensorflow.classifiers.audio.utils;

import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.io.jvm.AudioDispatcherFactory;
import be.tarsos.dsp.io.jvm.AudioPlayer;
import be.tarsos.dsp.pitch.PitchDetectionHandler;
import be.tarsos.dsp.pitch.PitchDetectionResult;
import be.tarsos.dsp.pitch.PitchProcessor;
import be.tarsos.dsp.util.PitchConverter;
import be.tarsos.dsp.util.fft.FFT;
import com.github.chen0040.tensorflow.classifiers.audio.utils.consts.MelSpectrogramDimension;
import lombok.Getter;
import lombok.Setter;

import javax.imageio.ImageIO;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

@Getter
@Setter
public class MelSpectrogram  implements PitchDetectionHandler {


    private boolean log2Console = false;
    private boolean showMarkers = false;
    private boolean showPitch = false;

    private float sampleRate = 44100;
    private int bufferSize = 1024 * 4;
    private int overlap = 768 * 4 ;

    private double pitch;

    private int outputFrameWidth = 640*4;
    private int outputFrameHeight = 480*4;

    private boolean wrapEnabled = false;

    String currentPitch = "";
    int position = 0;

    BufferedImage bufferedImage = new BufferedImage(outputFrameWidth,outputFrameHeight, BufferedImage.TYPE_INT_RGB);

    //private PitchProcessor.PitchEstimationAlgorithm algorithm = PitchProcessor.PitchEstimationAlgorithm.YIN;

    AudioProcessor fftProcessor = new AudioProcessor(){

        FFT fft = new FFT(bufferSize);
        float[] amplitudes = new float[bufferSize];


        public void processingFinished() {
            // TODO Auto-generated method stub
        }


        public boolean process(AudioEvent audioEvent) {
            float[] audioFloatBuffer = audioEvent.getFloatBuffer();
            float[] transformBuffer = new float[bufferSize * 2];
            System.arraycopy(audioFloatBuffer, 0, transformBuffer, 0, audioFloatBuffer.length);
            fft.forwardTransform(transformBuffer);
            fft.modulus(transformBuffer, amplitudes);
            drawFFT(pitch, amplitudes,fft, bufferedImage);
            return true;
        }

    };

    private int frequencyToBin(final double frequency) {
        final double minFrequency = 50; // Hz
        final double maxFrequency = 11000; // Hz
        int bin = 0;
        final boolean logaritmic = true;
        if (frequency != 0 && frequency > minFrequency && frequency < maxFrequency) {
            double binEstimate = 0;
            if (logaritmic) {
                final double minCent = PitchConverter.hertzToAbsoluteCent(minFrequency);
                final double maxCent = PitchConverter.hertzToAbsoluteCent(maxFrequency);
                final double absCent = PitchConverter.hertzToAbsoluteCent(frequency * 2);
                binEstimate = (absCent - minCent) / maxCent * outputFrameHeight;
            } else {
                binEstimate = (frequency - minFrequency) / maxFrequency * outputFrameHeight;
            }
            if (binEstimate > 700 && log2Console) {
                System.out.println(binEstimate + "");
            }
            bin = outputFrameHeight - 1 - (int) binEstimate;
        }
        return bin;
    }

    private void drawFFT(double pitch, float[] amplitudes, FFT fft, BufferedImage bufferedImage) {


        if(position >= outputFrameWidth && !wrapEnabled){
            return;
        }


        Graphics2D bufferedGraphics = bufferedImage.createGraphics();

        double maxAmplitude=0;
        //for every pixel calculate an amplitude
        float[] pixelAmplitudes = new float[outputFrameHeight];
        //iterate the large array and map to pixels
        for (int i = amplitudes.length/800; i < amplitudes.length; i++) {
            int pixelY = frequencyToBin(i * 44100 / (amplitudes.length * 8));
            pixelAmplitudes[pixelY] += amplitudes[i];
            maxAmplitude = Math.max(pixelAmplitudes[pixelY], maxAmplitude);
        }

        //draw the pixels
        for (int i = 0; i < pixelAmplitudes.length; i++) {
            Color color = Color.black;
            if (maxAmplitude != 0) {

                final int greyValue = (int) (Math.log1p(pixelAmplitudes[i] / maxAmplitude) / Math.log1p(1.0000001) * 255);
                color = new Color(greyValue, greyValue, greyValue);
            }
            bufferedGraphics.setColor(color);
            bufferedGraphics.fillRect(position, i, 3, 1);
        }


        if (showPitch && pitch != -1) {
            int pitchIndex = frequencyToBin(pitch);
            bufferedGraphics.setColor(Color.RED);
            bufferedGraphics.fillRect(position, pitchIndex, 1, 1);
            currentPitch = "Current frequency: " + (int) pitch + "Hz";
        }

        if(showMarkers) {
            bufferedGraphics.clearRect(0,0, 190,30);
            bufferedGraphics.setColor(Color.WHITE);


            bufferedGraphics.drawString(currentPitch, 20, 20);


            for(int i = 100 ; i < 500; i += 100){
                int bin = frequencyToBin(i);
                bufferedGraphics.drawLine(0, bin, 5, bin);
            }

            for(int i = 500 ; i <= 20000; i += 500){
                int bin = frequencyToBin(i);
                bufferedGraphics.drawLine(0, bin, 5, bin);
            }

            for (int i = 100; i <= 20000; i *= 10) {
                int bin = frequencyToBin(i);
                bufferedGraphics.drawString(String.valueOf(i), 10, bin);
            }
        }

        position+=3;
        position = position % outputFrameWidth;
    }

    public static BufferedImage convert_to_image(File f) {
        MelSpectrogram melGram = new MelSpectrogram();
        melGram.setOutputFrameWidth(MelSpectrogramDimension.Width);
        melGram.setOutputFrameHeight(MelSpectrogramDimension.Height);


        try {
            return melGram.convertAudio(f);
        } catch (IOException | UnsupportedAudioFileException | LineUnavailableException e) {
            e.printStackTrace();
        }
        return null;
    }


    public BufferedImage convertAudio(File audioFile) throws IOException, UnsupportedAudioFileException, LineUnavailableException {

        AudioDispatcher dispatcher = AudioDispatcherFactory.fromFile(audioFile, bufferSize, overlap);
        //AudioFormat format = AudioSystem.getAudioFileFormat(audioFile).getFormat();
        //dispatcher.addAudioProcessor(new AudioPlayer(format));

        bufferedImage = new BufferedImage(outputFrameWidth,outputFrameHeight, BufferedImage.TYPE_INT_RGB);

        // add a processor, handle pitch event.
        //dispatcher.addAudioProcessor(new PitchProcessor(algorithm, sampleRate, bufferSize, this));
        dispatcher.addAudioProcessor(fftProcessor);

        position = 0;
        currentPitch = "";

        // run the dispatcher (on a new thread).
        dispatcher.run();

        return bufferedImage;
    }

    public void handlePitch(PitchDetectionResult pitchDetectionResult,AudioEvent audioEvent) {
        if(pitchDetectionResult.isPitched()){
            pitch = pitchDetectionResult.getPitch();
        } else {
            pitch = -1;
        }

    }

    public static void main(String[] args) throws UnsupportedAudioFileException, IOException, LineUnavailableException {
        File file = new File("gtzan/genres");
        System.out.println(file.getAbsolutePath());
        if(file.isDirectory()) {
            for (File class_folder : file.listFiles()) {
                if(class_folder.isDirectory()) {
                    for (File f : class_folder.listFiles()) {
                        String file_path = f.getAbsolutePath();
                        if (file_path.endsWith("au")) {
                            System.out.println("Converting " + file_path + " ...");
                            String output_image_path = file_path + ".png";
                            File outputFile = new File(output_image_path);

                            if(outputFile.exists()) continue;

                            MelSpectrogram melGram = new MelSpectrogram();
                            melGram.setOutputFrameWidth(MelSpectrogramDimension.Width);
                            melGram.setOutputFrameHeight(MelSpectrogramDimension.Height);
                            BufferedImage image = melGram.convertAudio(f);

                            ImageIO.write(image, "png", outputFile);
                        }
                    }
                }
            }
        }
    }
}
