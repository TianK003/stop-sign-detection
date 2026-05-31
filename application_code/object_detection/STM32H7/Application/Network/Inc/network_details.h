/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-01-09T17:04:55+0100
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[89]) {
   { .size_bytes = 110592, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_U8, .shape = {4, (const int32_t[4]){1, 192, 192, 3}}, .scale = {1, (const float[1]){0.007843137718737125}}, .zeropoint = {1, (const int16_t[1]){127}}, .name = "serving_default_input_10_output" },
   { .size_bytes = 61280, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 3830, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conversion_141_const_output_output" },
   { .size_bytes = 110593, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 192, 192, 3}}, .scale = {1, (const float[1]){0.007843137718737125}}, .zeropoint = {1, (const int16_t[1]){-1}}, .name = "conversion_0_output" },
   { .size_bytes = 73728, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 96, 8}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_1_output" },
   { .size_bytes = 76832, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 98, 98, 8}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_2_pad_before_output" },
   { .size_bytes = 73728, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 96, 8}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_2_output" },
   { .size_bytes = 147456, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 96, 16}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_3_output" },
   { .size_bytes = 150544, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 97, 97, 16}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_5_pad_before_output" },
   { .size_bytes = 36864, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 48, 48, 16}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_5_output" },
   { .size_bytes = 73728, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 48, 48, 32}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_6_output" },
   { .size_bytes = 80000, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 50, 50, 32}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_7_pad_before_output" },
   { .size_bytes = 73728, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 48, 48, 32}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_7_output" },
   { .size_bytes = 73728, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 48, 48, 32}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_8_output" },
   { .size_bytes = 76832, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 49, 49, 32}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_10_pad_before_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 24, 24, 32}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_10_output" },
   { .size_bytes = 36864, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 24, 24, 64}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_11_output" },
   { .size_bytes = 43264, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 26, 26, 64}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_12_pad_before_output" },
   { .size_bytes = 36864, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 24, 24, 64}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_12_output" },
   { .size_bytes = 36864, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 24, 24, 64}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_13_output" },
   { .size_bytes = 40000, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 25, 25, 64}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_16_pad_before_output" },
   { .size_bytes = 9216, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 64}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_16_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_17_output" },
   { .size_bytes = 25088, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 14, 14, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_18_pad_before_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_18_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_19_output" },
   { .size_bytes = 25088, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 14, 14, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_20_pad_before_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_20_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_21_output" },
   { .size_bytes = 25088, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 14, 14, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_22_pad_before_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_22_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_23_output" },
   { .size_bytes = 25088, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 14, 14, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_24_pad_before_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_24_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_25_output" },
   { .size_bytes = 25088, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 14, 14, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_26_pad_before_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_26_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_27_output" },
   { .size_bytes = 21632, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 13, 13, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_30_pad_before_output" },
   { .size_bytes = 4608, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 6, 6, 128}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_30_output" },
   { .size_bytes = 9216, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 6, 6, 256}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_31_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 256}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_32_pad_before_output" },
   { .size_bytes = 9216, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 6, 6, 256}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_32_output" },
   { .size_bytes = 9216, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 6, 6, 256}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_33_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 256}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_35_pad_before_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 256}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_35_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 256}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_36_output" },
   { .size_bytes = 6400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 5, 5, 256}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_37_pad_before_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 256}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_37_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 256}}, .scale = {1, (const float[1]){0.0235294122248888}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_38_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 256}}, .scale = {1, (const float[1]){0.009940041229128838}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pool_40_output" },
   { .size_bytes = 32, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 32}}, .scale = {1, (const float[1]){0.03695271909236908}}, .zeropoint = {1, (const int16_t[1]){-16}}, .name = "conv2d_45_output" },
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 32}}, .scale = {1, (const float[1]){0.03695271909236908}}, .zeropoint = {1, (const int16_t[1]){-16}}, .name = "conv2d_46_pad_before_output" },
   { .size_bytes = 10, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 10}}, .scale = {1, (const float[1]){0.08089115470647812}}, .zeropoint = {1, (const int16_t[1]){-3}}, .name = "conv2d_46_output" },
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 32}}, .scale = {1, (const float[1]){0.03695271909236908}}, .zeropoint = {1, (const int16_t[1]){-16}}, .name = "conv2d_51_pad_before_output" },
   { .size_bytes = 20, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 1, 20}}, .scale = {1, (const float[1]){0.009279800578951836}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "conv2d_51_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 2, 2, 32}}, .scale = {1, (const float[1]){0.03695271909236908}}, .zeropoint = {1, (const int16_t[1]){-16}}, .name = "tile_62_output" },
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 32}}, .scale = {1, (const float[1]){0.03695271909236908}}, .zeropoint = {1, (const int16_t[1]){-16}}, .name = "pad_63_output" },
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 32}}, .scale = {1, (const float[1]){0.08672560751438141}}, .zeropoint = {1, (const int16_t[1]){5}}, .name = "conv2d_39_output" },
   { .size_bytes = 288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 32}}, .scale = {1, (const float[1]){0.09198352694511414}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "eltwise_64_output" },
   { .size_bytes = 800, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 5, 5, 32}}, .scale = {1, (const float[1]){0.09198352694511414}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "conv2d_65_pad_before_output" },
   { .size_bytes = 180, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 20}}, .scale = {1, (const float[1]){0.009279800578951836}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "conv2d_65_output" },
   { .size_bytes = 800, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 5, 5, 32}}, .scale = {1, (const float[1]){0.09198352694511414}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "conv2d_76_pad_before_output" },
   { .size_bytes = 90, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 3, 3, 10}}, .scale = {1, (const float[1]){0.08089115470647812}}, .zeropoint = {1, (const int16_t[1]){-3}}, .name = "conv2d_76_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 6, 6, 32}}, .scale = {1, (const float[1]){0.09198352694511414}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "resize_81_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 6, 6, 32}}, .scale = {1, (const float[1]){0.11229196190834045}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "conv2d_34_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 6, 6, 32}}, .scale = {1, (const float[1]){0.11974716186523438}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "eltwise_82_output" },
   { .size_bytes = 2048, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 32}}, .scale = {1, (const float[1]){0.11974716186523438}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "conv2d_83_pad_before_output" },
   { .size_bytes = 360, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 6, 6, 10}}, .scale = {1, (const float[1]){0.08089115470647812}}, .zeropoint = {1, (const int16_t[1]){-3}}, .name = "conv2d_83_output" },
   { .size_bytes = 2048, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 32}}, .scale = {1, (const float[1]){0.11974716186523438}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "conv2d_88_pad_before_output" },
   { .size_bytes = 720, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 6, 6, 20}}, .scale = {1, (const float[1]){0.009279800578951836}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "conv2d_88_output" },
   { .size_bytes = 4608, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 32}}, .scale = {1, (const float[1]){0.11974716186523438}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "resize_99_output" },
   { .size_bytes = 4608, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 32}}, .scale = {1, (const float[1]){0.079196497797966}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "conv2d_28_output" },
   { .size_bytes = 4608, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 32}}, .scale = {1, (const float[1]){0.10870534926652908}}, .zeropoint = {1, (const int16_t[1]){4}}, .name = "eltwise_100_output" },
   { .size_bytes = 6272, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 14, 14, 32}}, .scale = {1, (const float[1]){0.10870534926652908}}, .zeropoint = {1, (const int16_t[1]){4}}, .name = "conv2d_101_pad_before_output" },
   { .size_bytes = 1440, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 10}}, .scale = {1, (const float[1]){0.08089115470647812}}, .zeropoint = {1, (const int16_t[1]){-3}}, .name = "conv2d_101_output" },
   { .size_bytes = 6272, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 14, 14, 32}}, .scale = {1, (const float[1]){0.10870534926652908}}, .zeropoint = {1, (const int16_t[1]){4}}, .name = "conv2d_106_pad_before_output" },
   { .size_bytes = 2880, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 12, 12, 20}}, .scale = {1, (const float[1]){0.009279800578951836}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "conv2d_106_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 24, 24, 32}}, .scale = {1, (const float[1]){0.10870534926652908}}, .zeropoint = {1, (const int16_t[1]){4}}, .name = "resize_117_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 24, 24, 32}}, .scale = {1, (const float[1]){0.02486031875014305}}, .zeropoint = {1, (const int16_t[1]){11}}, .name = "conv2d_14_output" },
   { .size_bytes = 18432, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 24, 24, 32}}, .scale = {1, (const float[1]){0.10854417085647583}}, .zeropoint = {1, (const int16_t[1]){3}}, .name = "eltwise_118_output" },
   { .size_bytes = 21632, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 26, 26, 32}}, .scale = {1, (const float[1]){0.10854417085647583}}, .zeropoint = {1, (const int16_t[1]){3}}, .name = "conv2d_119_pad_before_output" },
   { .size_bytes = 5760, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 24, 24, 10}}, .scale = {1, (const float[1]){0.08089115470647812}}, .zeropoint = {1, (const int16_t[1]){-3}}, .name = "conv2d_119_output" },
   { .size_bytes = 7660, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 3830, 2}}, .scale = {1, (const float[1]){0.08089115470647812}}, .zeropoint = {1, (const int16_t[1]){-3}}, .name = "concat_124_output" },
   { .size_bytes = 7660, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 3830, 2}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_125_output" },
   { .size_bytes = 30640, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 3830, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conversion_126_output" },
   { .size_bytes = 21632, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 26, 26, 32}}, .scale = {1, (const float[1]){0.10854417085647583}}, .zeropoint = {1, (const int16_t[1]){3}}, .name = "conv2d_127_pad_before_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 24, 24, 20}}, .scale = {1, (const float[1]){0.009279800578951836}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "conv2d_127_output" },
   { .size_bytes = 15320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 3830, 4}}, .scale = {1, (const float[1]){0.009279800578951836}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "concat_132_output" },
   { .size_bytes = 61280, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 3830, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conversion_133_output" }
  },
  .nodes = (const stai_node_details[88]){
    {.id = 141, .type = AI_LAYER_CONCAT_TYPE, .input_tensors = {0, NULL}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conversion_141_const_output */
    {.id = 0, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* conversion_0 */
    {.id = 1, .type = AI_LAYER_OPTIMIZED_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv2d_1 */
    {.id = 2, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* conv2d_2_pad_before */
    {.id = 2, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* conv2d_2 */
    {.id = 3, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* conv2d_3 */
    {.id = 4, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* conv2d_5_pad_before */
    {.id = 5, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* conv2d_5 */
    {.id = 6, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* conv2d_6 */
    {.id = 7, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* conv2d_7_pad_before */
    {.id = 7, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* conv2d_7 */
    {.id = 8, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* conv2d_8 */
    {.id = 9, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* conv2d_10_pad_before */
    {.id = 10, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* conv2d_10 */
    {.id = 11, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* conv2d_11 */
    {.id = 12, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* conv2d_12_pad_before */
    {.id = 12, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* conv2d_12 */
    {.id = 13, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* conv2d_13 */
    {.id = 15, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* conv2d_16_pad_before */
    {.id = 16, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){19}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* conv2d_16 */
    {.id = 17, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){20}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* conv2d_17 */
    {.id = 18, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* conv2d_18_pad_before */
    {.id = 18, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){22}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* conv2d_18 */
    {.id = 19, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){23}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* conv2d_19 */
    {.id = 20, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){24}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* conv2d_20_pad_before */
    {.id = 20, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){25}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* conv2d_20 */
    {.id = 21, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){26}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* conv2d_21 */
    {.id = 22, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){27}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* conv2d_22_pad_before */
    {.id = 22, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){28}}, .output_tensors = {1, (const int32_t[1]){29}} }, /* conv2d_22 */
    {.id = 23, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){29}}, .output_tensors = {1, (const int32_t[1]){30}} }, /* conv2d_23 */
    {.id = 24, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){30}}, .output_tensors = {1, (const int32_t[1]){31}} }, /* conv2d_24_pad_before */
    {.id = 24, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){31}}, .output_tensors = {1, (const int32_t[1]){32}} }, /* conv2d_24 */
    {.id = 25, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){32}}, .output_tensors = {1, (const int32_t[1]){33}} }, /* conv2d_25 */
    {.id = 26, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){33}}, .output_tensors = {1, (const int32_t[1]){34}} }, /* conv2d_26_pad_before */
    {.id = 26, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){34}}, .output_tensors = {1, (const int32_t[1]){35}} }, /* conv2d_26 */
    {.id = 27, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){35}}, .output_tensors = {1, (const int32_t[1]){36}} }, /* conv2d_27 */
    {.id = 29, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){36}}, .output_tensors = {1, (const int32_t[1]){37}} }, /* conv2d_30_pad_before */
    {.id = 30, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){37}}, .output_tensors = {1, (const int32_t[1]){38}} }, /* conv2d_30 */
    {.id = 31, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){38}}, .output_tensors = {1, (const int32_t[1]){39}} }, /* conv2d_31 */
    {.id = 32, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){39}}, .output_tensors = {1, (const int32_t[1]){40}} }, /* conv2d_32_pad_before */
    {.id = 32, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){40}}, .output_tensors = {1, (const int32_t[1]){41}} }, /* conv2d_32 */
    {.id = 33, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){41}}, .output_tensors = {1, (const int32_t[1]){42}} }, /* conv2d_33 */
    {.id = 35, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){42}}, .output_tensors = {1, (const int32_t[1]){43}} }, /* conv2d_35_pad_before */
    {.id = 35, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){43}}, .output_tensors = {1, (const int32_t[1]){44}} }, /* conv2d_35 */
    {.id = 36, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){44}}, .output_tensors = {1, (const int32_t[1]){45}} }, /* conv2d_36 */
    {.id = 37, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){45}}, .output_tensors = {1, (const int32_t[1]){46}} }, /* conv2d_37_pad_before */
    {.id = 37, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){47}} }, /* conv2d_37 */
    {.id = 38, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){47}}, .output_tensors = {1, (const int32_t[1]){48}} }, /* conv2d_38 */
    {.id = 40, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){48}}, .output_tensors = {1, (const int32_t[1]){49}} }, /* pool_40 */
    {.id = 45, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){49}}, .output_tensors = {1, (const int32_t[1]){50}} }, /* conv2d_45 */
    {.id = 46, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){50}}, .output_tensors = {1, (const int32_t[1]){51}} }, /* conv2d_46_pad_before */
    {.id = 46, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){51}}, .output_tensors = {1, (const int32_t[1]){52}} }, /* conv2d_46 */
    {.id = 51, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){50}}, .output_tensors = {1, (const int32_t[1]){53}} }, /* conv2d_51_pad_before */
    {.id = 51, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){53}}, .output_tensors = {1, (const int32_t[1]){54}} }, /* conv2d_51 */
    {.id = 62, .type = AI_LAYER_TILE_TYPE, .input_tensors = {1, (const int32_t[1]){50}}, .output_tensors = {1, (const int32_t[1]){55}} }, /* tile_62 */
    {.id = 63, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){55}}, .output_tensors = {1, (const int32_t[1]){56}} }, /* pad_63 */
    {.id = 39, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){48}}, .output_tensors = {1, (const int32_t[1]){57}} }, /* conv2d_39 */
    {.id = 64, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){57, 56}}, .output_tensors = {1, (const int32_t[1]){58}} }, /* eltwise_64 */
    {.id = 65, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){58}}, .output_tensors = {1, (const int32_t[1]){59}} }, /* conv2d_65_pad_before */
    {.id = 65, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){59}}, .output_tensors = {1, (const int32_t[1]){60}} }, /* conv2d_65 */
    {.id = 76, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){58}}, .output_tensors = {1, (const int32_t[1]){61}} }, /* conv2d_76_pad_before */
    {.id = 76, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){61}}, .output_tensors = {1, (const int32_t[1]){62}} }, /* conv2d_76 */
    {.id = 81, .type = AI_LAYER_UPSAMPLE_TYPE, .input_tensors = {1, (const int32_t[1]){58}}, .output_tensors = {1, (const int32_t[1]){63}} }, /* resize_81 */
    {.id = 34, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){42}}, .output_tensors = {1, (const int32_t[1]){64}} }, /* conv2d_34 */
    {.id = 82, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){64, 63}}, .output_tensors = {1, (const int32_t[1]){65}} }, /* eltwise_82 */
    {.id = 83, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){65}}, .output_tensors = {1, (const int32_t[1]){66}} }, /* conv2d_83_pad_before */
    {.id = 83, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){66}}, .output_tensors = {1, (const int32_t[1]){67}} }, /* conv2d_83 */
    {.id = 88, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){65}}, .output_tensors = {1, (const int32_t[1]){68}} }, /* conv2d_88_pad_before */
    {.id = 88, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){68}}, .output_tensors = {1, (const int32_t[1]){69}} }, /* conv2d_88 */
    {.id = 99, .type = AI_LAYER_UPSAMPLE_TYPE, .input_tensors = {1, (const int32_t[1]){65}}, .output_tensors = {1, (const int32_t[1]){70}} }, /* resize_99 */
    {.id = 28, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){36}}, .output_tensors = {1, (const int32_t[1]){71}} }, /* conv2d_28 */
    {.id = 100, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){71, 70}}, .output_tensors = {1, (const int32_t[1]){72}} }, /* eltwise_100 */
    {.id = 101, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){72}}, .output_tensors = {1, (const int32_t[1]){73}} }, /* conv2d_101_pad_before */
    {.id = 101, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){73}}, .output_tensors = {1, (const int32_t[1]){74}} }, /* conv2d_101 */
    {.id = 106, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){72}}, .output_tensors = {1, (const int32_t[1]){75}} }, /* conv2d_106_pad_before */
    {.id = 106, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){75}}, .output_tensors = {1, (const int32_t[1]){76}} }, /* conv2d_106 */
    {.id = 117, .type = AI_LAYER_UPSAMPLE_TYPE, .input_tensors = {1, (const int32_t[1]){72}}, .output_tensors = {1, (const int32_t[1]){77}} }, /* resize_117 */
    {.id = 14, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){78}} }, /* conv2d_14 */
    {.id = 118, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){78, 77}}, .output_tensors = {1, (const int32_t[1]){79}} }, /* eltwise_118 */
    {.id = 119, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){79}}, .output_tensors = {1, (const int32_t[1]){80}} }, /* conv2d_119_pad_before */
    {.id = 119, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){80}}, .output_tensors = {1, (const int32_t[1]){81}} }, /* conv2d_119 */
    {.id = 124, .type = AI_LAYER_CONCAT_TYPE, .input_tensors = {5, (const int32_t[5]){81, 74, 67, 62, 52}}, .output_tensors = {1, (const int32_t[1]){82}} }, /* concat_124 */
    {.id = 125, .type = AI_LAYER_SM_TYPE, .input_tensors = {1, (const int32_t[1]){82}}, .output_tensors = {1, (const int32_t[1]){83}} }, /* nl_125 */
    {.id = 126, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){83}}, .output_tensors = {1, (const int32_t[1]){84}} }, /* conversion_126 */
    {.id = 127, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){79}}, .output_tensors = {1, (const int32_t[1]){85}} }, /* conv2d_127_pad_before */
    {.id = 127, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){85}}, .output_tensors = {1, (const int32_t[1]){86}} }, /* conv2d_127 */
    {.id = 132, .type = AI_LAYER_CONCAT_TYPE, .input_tensors = {5, (const int32_t[5]){86, 76, 69, 60, 54}}, .output_tensors = {1, (const int32_t[1]){87}} }, /* concat_132 */
    {.id = 133, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){87}}, .output_tensors = {1, (const int32_t[1]){88}} } /* conversion_133 */
  },
  .n_nodes = 88
};
#endif

