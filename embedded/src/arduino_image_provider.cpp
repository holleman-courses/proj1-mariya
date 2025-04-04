/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#include "image_provider.h"

#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"
#include <TinyMLShield.h>

#define CAMERA_IMAGE_WIDTH    176
#define CAMERA_IMAGE_HEIGHT   144
#define MODEL_IMAGE_WIDTH     96
#define MODEL_IMAGE_HEIGHT    96
// Get an image from the camera module
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data) {

  byte data[CAMERA_IMAGE_WIDTH * CAMERA_IMAGE_HEIGHT]; // Receiving QCIF grayscale from camera = 176 * 144 * 1

  static bool g_is_camera_initialized = false;
  static bool serial_is_initialized = false;

  // Initialize camera if necessary
  if (!g_is_camera_initialized) {
    if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
      return kTfLiteError;
    }
    g_is_camera_initialized = true;
  }

  // Read camera data
  Camera.readFrame(data);

  int min_x = (CAMERA_IMAGE_WIDTH - MODEL_IMAGE_WIDTH) / 2;
  int min_y = (CAMERA_IMAGE_HEIGHT - MODEL_IMAGE_HEIGHT) / 2;
  int index = 0;

  // Crop full image. This lowers FOV, ideally we would downsample but this is simpler. 
  for (int y = min_y; y < min_y + MODEL_IMAGE_WIDTH; y++) {
    for (int x = min_x; x < min_x + MODEL_IMAGE_HEIGHT; x++) {
      image_data[index++] = static_cast<int8_t>(data[(y * CAMERA_IMAGE_WIDTH) + x] - 128); // convert TF input image to signed 8-bit
    }
  }

  return kTfLiteOk;
}

#endif  // ARDUINO_EXCLUDE_CODE