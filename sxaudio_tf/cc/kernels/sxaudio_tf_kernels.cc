/*  Copyright 2022 StreamLogic, LLC.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using tensorflow::errors::InvalidArgument;


extern void rFFT16(const int N, const int M, const int16_t *win,
		   uint16_t *fft, int16_t *fr, int16_t *fi);
extern void mel_spectrum_u16(int N, int M, int lo, int hi, uint16_t *spectrum);

static size_t log2i(size_t n) {
  return ( (n<2) ? 0 : 1+log2i(n/2));
}

class SpectrogramOp : public OpKernel {
 public:
  explicit SpectrogramOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &this->window_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_step", &this->window_step));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_channels", &this->num_channels));

    float upper_band_limit;
    float lower_band_limit;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("upper_band_limit", &upper_band_limit));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lower_band_limit", &lower_band_limit));
    freq_lo = (int)lower_band_limit;
    freq_hi = (int)upper_band_limit;

    N = window_size;
    M = log2i(N);

    fr_buf = (int16_t *)malloc(N*sizeof(int16_t));
    fi_buf = (int16_t *)malloc(N*sizeof(int16_t));
    fft_buf = (uint16_t *)malloc((N/2+1)*sizeof(int16_t));
  }

  ~SpectrogramOp() {
    free(fft_buf);
    free(fi_buf);
    free(fr_buf);
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* audio;
    OP_REQUIRES_OK(ctx, ctx->input("audio", &audio));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(audio->shape()),
                InvalidArgument("audio is not a vector"));

    auto audio_data =
        reinterpret_cast<const int16_t*>(audio->tensor_data().data());
    int audio_size = audio->NumElements();

    Tensor* output_tensor = nullptr;

    int num_frames = 0;
    int sampled_frames = 0;
    if (audio_size >= window_size) {
      num_frames = (audio_size - window_size) / window_step + 1;
    }
    TensorShape output_shape{ num_frames, num_channels };
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));
    auto output_flat = output_tensor->flat<uint16>();

    const int B = N/2 + 1;
    int index = 0;
    while (audio_size > window_size) {
      rFFT16(N, M, audio_data, fft_buf, fr_buf, fi_buf);
      audio_data += window_step;
      audio_size -= window_step;

      mel_spectrum_u16(num_channels+1, B, freq_lo, freq_hi, fft_buf);

      for (int i = 0; i < num_channels; ++i) {
	output_flat(index++) = fft_buf[i];
      }
    }
  }

 private:
  int N, M;
  int window_size;
  int window_step;
  int num_channels;
  int freq_lo;
  int freq_hi;

  int16_t *fr_buf;
  int16_t *fi_buf;
  uint16_t *fft_buf;
};

REGISTER_KERNEL_BUILDER(Name("Spectrogram").Device(DEVICE_CPU), SpectrogramOp);
