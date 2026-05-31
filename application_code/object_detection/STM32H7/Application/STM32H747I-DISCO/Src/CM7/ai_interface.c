/**
 ******************************************************************************
 * @file    ai_interface.c
 * @author  MCD Application Team
 * @brief   Abstraction interface to AI generated code
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2019 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

/* Includes ------------------------------------------------------------------*/
#include "ai_interface.h"
#include <string.h>
#include "ai_datatypes_defines.h"

/* Private typedef -----------------------------------------------------------*/
/* Private defines -----------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
STAI_NETWORK_CONTEXT_DECLARE(network, STAI_NETWORK_CONTEXT_SIZE)

/**
 * @brief Returns the input format type
 * @retval stai_format Input format type: quantized (STAI_FORMAT_Q) or float (STAI_FORMAT_FLOAT32)
 */
stai_format ai_get_input_format(void)
{
  return STAI_NETWORK_IN_1_FORMAT;
}

/**
 * @brief Returns the output format type
 * @retval stai_format Output format type: quantized (STAI_FORMAT_Q) or float (STAI_FORMAT_FLOAT32)
 */
stai_format ai_get_output_format(void)
{
  return STAI_NETWORK_OUT_1_FORMAT;
}

/**
 * @brief Returns value of the input quantized format
 * @retval ai_size Input quantized format
 */
ai_size ai_get_input_quantized_format(void)
{
  stai_format fmt = ai_get_input_format();
  return STAI_FORMAT_GET_IBITS(fmt);
}

/**
 * @brief Returns the quantization scheme used to quantize the input layer of the neural network
 * @retval ai_size Quantization scheme: AI_FXP_Q, AI_UINT_Q, AI_SINT_Q
 */
uint32_t ai_get_input_quantization_scheme(void)
{
  ai_float scale = ai_get_input_scale();

  stai_format fmt = ai_get_input_format();
  ai_size sign = STAI_FORMAT_GET_SIGN(fmt);

  if(scale==0)
  {
    return AI_FXP_Q;
  }
  else
  {
    if(sign==0)
    {
      return AI_UINT_Q;
    }
    else
    {
      return AI_SINT_Q;
    }
  }
}

/**
 * @brief Returns the quantization scheme used to quantize the output layer of the neural network
 * @retval ai_size Quantization scheme: AI_FXP_Q, AI_UINT_Q, AI_SINT_Q
 */
uint32_t ai_get_output_quantization_scheme(void)
{
  ai_float scale = ai_get_output_scale();

  stai_format fmt = ai_get_output_format();
  ai_size sign = STAI_FORMAT_GET_SIGN(fmt);

  if(scale==0)
  {
    return AI_FXP_Q;
  }
  else
  {
    if(sign==0)
    {
      return AI_UINT_Q;
    }
    else
    {
      return AI_SINT_Q;
    }
  }
}


/**
 * @brief Returns value of the scale for the output quantized format
 * @retval ai_size Scale for output quantized format
 */
ai_float ai_get_output_fxp_scale(void)
{
  float scale;

  /* Retrieve format of the output tensor - index 0 */
  stai_format fmt = ai_get_output_format();

  /* Build the scale factor for conversion */
  scale = 1.0f / (0x1U << STAI_FORMAT_GET_FBITS(fmt));

  return scale;
}

/**
 * @brief Returns value of the scale for the input quantized format
 * @retval ai_size Scale for input quantized format
 */
ai_float ai_get_input_scale(void)
{
  return STAI_NETWORK_IN_1_SCALE;
}

/**
 * @brief Returns value of the zero point for the input quantized format
 * @retval ai_size Zero point for input quantized format
 */
ai_i32 ai_get_input_zero_point(void)
{
  return STAI_NETWORK_IN_1_ZERO_POINT;
}

/**
 * @brief Returns value of the scale for the output quantized format
 * @retval ai_size Scale for output quantized format
 */
ai_float ai_get_output_scale(void)
{
#ifdef  STAI_NETWORK_OUT_1_SCALE
  return STAI_NETWORK_OUT_1_SCALE;
#else
  return -1;
#endif
}


/**
 * @brief Returns value of the zero point for the output quantized format
 * @retval ai_size Zero point for output quantized format
 */
ai_i32 ai_get_output_zero_point(void)
{
#ifdef  STAI_NETWORK_OUT_1_ZERO_POINT
  return STAI_NETWORK_OUT_1_ZERO_POINT;
#else
  return -1;
#endif
}

/**
 * @brief Initializes the generated C model for a neural network
 * @param  activation_buffer Pointer to the activation buffer (i.e. working buffer used during NN inference)
 * @retval ai_handle
 */
void ai_init(uint8_t** activation_buffer, stai_ptr* inputs_buff_Ptr, stai_ptr* outputs_buff_Ptr)
{
  stai_return_code err;
  stai_size ai_input_size = 0;
  stai_size ai_output_size = 0;

  err = stai_runtime_init();
  assert(err == STAI_SUCCESS);

  err = stai_network_init(network);
  assert(err == STAI_SUCCESS);
  /* Create and initialize the c-model */
#if AI_ACTIVATION_BUFFERS_COUNT == 1
  const stai_ptr acts[] = { activation_buffer[0] };
#elif AI_ACTIVATION_BUFFERS_COUNT == 2
  const stai_ptr acts[] = { activation_buffer[0], activation_buffer[1] };
#elif AI_ACTIVATION_BUFFERS_COUNT == 3
  const stai_ptr acts[] = { activation_buffer[0], activation_buffer[1], activation_buffer[2] };
#endif

  err = stai_network_set_activations(network, acts, AI_ACTIVATION_BUFFERS_COUNT);
  assert(err == STAI_SUCCESS);

  err = stai_network_get_inputs(network, inputs_buff_Ptr, &ai_input_size);
  assert(err == STAI_SUCCESS);

  err = stai_network_get_outputs(network, outputs_buff_Ptr, &ai_output_size);
  assert(err == STAI_SUCCESS);
}

/**
 * @brief De-initializes the generated C model for a neural network
 */
void ai_deinit(void)
{
  stai_network_deinit(network);
}

/**
 * @brief  Run an inference of the generated C model for a neural network
 */
void ai_run(void)
{
  stai_return_code err;
  err = stai_network_run(network, STAI_MODE_SYNC);
  assert(err == STAI_SUCCESS);
}

