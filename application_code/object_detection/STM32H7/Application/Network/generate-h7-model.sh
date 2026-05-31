#!/bin/bash

stedgeai generate -m st_ssd_mobilenet_v1_025_192_int8.tflite --target stm32h7
cp st_ai_output/network.h Inc/
cp st_ai_output/network_data.h Inc/
cp st_ai_output/network_details.h Inc/
cp st_ai_output/network.c Src/
cp st_ai_output/network_data.c Src/
cp st_ai_output/network_c_info.json .
cp st_ai_output/network_generate_report.txt .