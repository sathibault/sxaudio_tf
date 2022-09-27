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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("Spectrogram")
    .Input("audio: int16")
    .Output("spectrogram: uint16")
    .Attr("window_size: int = 256")
    .Attr("window_step: int = 128")
    .Attr("num_channels: int = 32")
    .Attr("upper_band_limit: float = 8000.0")
    .Attr("lower_band_limit: float = 0.0")
    .SetShapeFn([](InferenceContext* ctx) {
	ShapeHandle input;
	TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 1, &input));

	int window_size;
	TF_RETURN_IF_ERROR(ctx->GetAttr("window_size", &window_size));
	int window_step;
	TF_RETURN_IF_ERROR(ctx->GetAttr("window_step", &window_step));

	int num_channels;
	TF_RETURN_IF_ERROR(ctx->GetAttr("num_channels", &num_channels));

	DimensionHandle num_frames = ctx->Dim(input, 0);
	TF_RETURN_IF_ERROR(ctx->Subtract(num_frames, window_size, &num_frames));
        TF_RETURN_IF_ERROR(
            ctx->Divide(num_frames, window_step, false, &num_frames));

	DimensionHandle num_features = ctx->MakeDim(num_channels);

	ShapeHandle output = ctx->MakeShape({num_frames, num_features});
	ctx->set_output(0, output);

	return Status::OK();
    });
