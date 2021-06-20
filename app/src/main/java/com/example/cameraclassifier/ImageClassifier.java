package com.example.cameraclassifier;

import android.app.Activity;
import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class ImageClassifier {
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    private final Interpreter classifier;
    private final int img_resize_x;
    private final int img_resize_y;
    private TensorImage input_image_buffer;
    private final TensorBuffer probability_tensor_buffer;
    private final float PROBABILITY_MEAN = 0.0f;
    private final float PROBABILITY_STD = 255.0f;
    private final TensorProcessor probability_processor;
    private final List<String> labels;
    private int MAX_RESULTS = 5;


    public ImageClassifier(Activity activity) throws IOException {
        //Load the model into a MappedByteBuffer object
        MappedByteBuffer classifier_model = FileUtil.loadMappedFile(activity, "mobilenet_v1_1.0_224_quant.tflite");
        //Load the labels into a list of strings
        labels = FileUtil.loadLabels(activity, "labels_mobilenet_quant_v1_224.txt");
        //Create the interpreter object that performs inference with no options
        classifier = new Interpreter(classifier_model, null);

        //The input index = 0 and the output index = 0 since we have only one input and one output
        int INPUT_IMAGE_INDEX = 0;
        int OUTPUT_PROP_INDEX = 0;

        int[] input_image_shape = classifier.getInputTensor(INPUT_IMAGE_INDEX).shape();     //{Batch_size, Height, Width, 3}
        DataType input_data_type = classifier.getInputTensor(INPUT_IMAGE_INDEX).dataType();

        int[] output_props_shape = classifier.getOutputTensor(OUTPUT_PROP_INDEX).shape();   //{Batch_size, Num_classes}
        DataType output_props_type = classifier.getOutputTensor(OUTPUT_PROP_INDEX).dataType();

        //Find the models required input image size
        img_resize_x = input_image_shape[1];
        img_resize_y = input_image_shape[2];

        //creates tensors for the input and the output
        input_image_buffer = new TensorImage(input_data_type);
        probability_tensor_buffer = TensorBuffer.createFixedSize(output_props_shape, output_props_type);
        //A processor that post-process the output probabilities tensor
        probability_processor = new TensorProcessor.Builder().add(new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD)).build();  //dequantize the output results
    }

    public List<Recognition> recognizeImage(final Bitmap bitmap, final int sensorOrientation){
        List<Recognition> recognitions = new ArrayList<>();
        input_image_buffer = loadImage(bitmap, sensorOrientation);
        //getBuffer transforms the ImageBuffer/TensorBuffer into a ByteBuffer that the interpreter accepts
        classifier.run(input_image_buffer.getBuffer(), probability_tensor_buffer.getBuffer().rewind()); //.rewind() ???
        //returns a map that has each label and its corresponding probability
        Map<String, Float> labelled_props = new TensorLabel(labels, probability_processor.process(probability_tensor_buffer)).getMapWithFloatValue();

        //Adds every element in the labelled probabilities to the list of Recognition
        for(Map.Entry<String, Float> entry : labelled_props.entrySet())
        {
            recognitions.add(new Recognition(entry.getKey(), entry.getValue()));
        }
        //Sorts the list of predictions based on confidence score
        Collections.sort(recognitions);
        //return the top 5 predictions
        recognitions = recognitions.subList(0, MAX_RESULTS);
        return recognitions;
    }

    //This part pre-process the input image taken from the camera and turn it into a TensorImage
    private TensorImage loadImage(Bitmap bitmap, int sensorOrientation) {
        input_image_buffer.load(bitmap);
        int no_of_rotations = sensorOrientation/90;
        int crop_size = Math.min(bitmap.getWidth(), bitmap.getHeight());
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(crop_size, crop_size))
                .add(new ResizeOp(img_resize_x, img_resize_y, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(new Rot90Op(no_of_rotations))
                .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                .build();
        return imageProcessor.process(input_image_buffer);
    }


    //Each subject in the labels has a recognition object that consists of its name and its
    //confidence score that it achieves of the image
    //These recognition objects can be compared according to their confidence score
    public class Recognition implements Comparable{
        private String name;
        private float confidence;

        public Recognition(){}
        public Recognition(String name, float confidence)
        {
            this.name = name;
            this.confidence = confidence;
        }

        public String getName(){return name;}
        public float getConfidence(){return confidence;}
        public void setName(String name){this.name = name;}
        public void setConfidence(float confidence){this.confidence = confidence;}

        @Override
        public String toString(){
            return "Recognition { name=" + name + ", confidence= " + confidence + " }";
        }

        @Override
        public int compareTo(Object o)
        {
            return Float.compare(((Recognition)o).confidence, this.confidence);
        }
    }
}