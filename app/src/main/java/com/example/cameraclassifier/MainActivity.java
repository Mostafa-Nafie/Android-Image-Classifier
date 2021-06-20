package com.example.cameraclassifier;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    static final int REQUEST_IMAGE_CAPTURE = 1;
    private ImageView imageView;
    private Button capture_button;
    private ListView probs_list;
    private ImageClassifier imageClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        capture_button = (Button)findViewById(R.id.button);
        imageView = (ImageView)findViewById(R.id.imageView);
        probs_list = (ListView)findViewById(R.id.probs_list);

        try {
            imageClassifier = new ImageClassifier(this);
        } catch (IOException e) {
            Log.e("classifier", "Error while creating the classifier: " + e);
        }

        if (!hasCamera())
            capture_button.setEnabled(false);


        capture_button.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        launchCamera();
                    }
                }
        );
    }

    private boolean hasCamera(){
        return getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY);
    }

    private void launchCamera(){
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            //Display the image returned from the camera in the imageview
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            imageView.setImageBitmap(imageBitmap);


            List<ImageClassifier.Recognition> predictions = imageClassifier.recognizeImage(imageBitmap, 0);
            List<String> predictionList = new ArrayList<>();
            for(ImageClassifier.Recognition recognition : predictions)
            {
                predictionList.add("Label: " + recognition.getName() + " ---- Confidence: " + recognition.getConfidence());
            }
            ArrayAdapter<String> predictionsAdapter = new ArrayAdapter<>(this, R.layout.support_simple_spinner_dropdown_item, predictionList);
            probs_list.setAdapter(predictionsAdapter);

        }
    }
}