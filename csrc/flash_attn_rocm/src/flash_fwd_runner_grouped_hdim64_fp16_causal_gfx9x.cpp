// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
// HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "flash_fwd_runner_gfx9x.h"

namespace fwd_device_gemm {
template <>
void FlashFwdRunner::Run<true, 64, device_gemm_trait::Float16, true, true>(FlashFwdParams &params, hipStream_t &stream) {
  if(params.is_dropout){
    BOOL_SWITCH(is_deterministic_, kIsDeterministic, [&] {
      this->template run_<DeviceGemmGroupedHeadDim64,
                          device_gemm_trait::Float16, 
                          device_gemm_trait::kGemmSpecPadding,
                          device_gemm_trait::kMaskingSpecCausal,
                          kIsDeterministic>(params, stream);
    });
  }
  else{
    BOOL_SWITCH(is_deterministic_, kIsDeterministic, [&] {
      this->template run_<DeviceGemmGroupedHeadDim64NonDrop,
                          device_gemm_trait::Float16, 
                          device_gemm_trait::kGemmSpecPadding,
                          device_gemm_trait::kMaskingSpecCausal,
                          kIsDeterministic>(params, stream);
    });
  }
} // FlashFwdRunner::Run()

template <>
void FlashFwdRunner::Run<true, 64, device_gemm_trait::Float16, false, true>(FlashFwdParams &params, hipStream_t &stream) {
  if(params.is_dropout){
    BOOL_SWITCH(is_deterministic_, kIsDeterministic, [&] {
      this->template run_<DeviceGemmGroupedHeadDim64,
                          device_gemm_trait::Float16, 
                          device_gemm_trait::kGemmSpecDefault,
                          device_gemm_trait::kMaskingSpecCausal,
                          kIsDeterministic>(params, stream);
    });
  }
  else{
    BOOL_SWITCH(is_deterministic_, kIsDeterministic, [&] {
      this->template run_<DeviceGemmGroupedHeadDim64NonDrop,
                          device_gemm_trait::Float16, 
                          device_gemm_trait::kGemmSpecDefault,
                          device_gemm_trait::kMaskingSpecCausal,
                          kIsDeterministic>(params, stream);
    });
  }
} // FlashFwdRunner::Run()
} // namespace fwd_device_gemm