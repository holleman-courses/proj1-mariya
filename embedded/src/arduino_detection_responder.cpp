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

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "detection_responder.h"

#include "Arduino.h"

// Flash the blue LED after each inference
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        int8_t cat_score, int8_t no_cat_score) {
  static bool is_initialized = false;
  if (!is_initialized) {
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    is_initialized = true;
  }

  // Note: The RGB LEDs on the Arduino Nano 33 BLE
  // Sense are on when the pin is LOW, off when HIGH.

  // Switch the person/not person LEDs off
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDR, HIGH);

  // Flash the blue LED after every inference.
  digitalWrite(LEDB, LOW);
  delay(100);
  digitalWrite(LEDB, HIGH);

  // Switch on the green LED when a cat is detected,
  // the red when no cat is detected
  if (cat_score > no_cat_score) {
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDR, HIGH);
    TF_LITE_REPORT_ERROR(error_reporter, "Cat detected",
      cat_score, no_cat_score);
  } else {
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, LOW);
    TF_LITE_REPORT_ERROR(error_reporter, "No cat",
      cat_score, no_cat_score);
  }

  //TF_LITE_REPORT_ERROR(error_reporter, "Cat score: %d No cat score: %d",
  //                     cat_score, no_cat_score);
}

#endif  // ARDUINO_EXCLUDE_CODE
