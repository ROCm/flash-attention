
# batch, hq, hk, sq, sk, d_head, causal, dropout
# LLaMA 3 8B: (1, 32, 8, 8192, 8192, 128, True, 0.0)
# LLaMA 3 70B: (1, 64, 8, 8192, 8192, 128, True, 0.0)

# increase BLOCK_M, perf better
# export AMD_TRITON_FWD_DECODE_BLOCK_M=64 && python bench.py -benchmark_fn flash_attn_with_kvcache --mode fwd

# increase num_stages, num_warps_fwd, and num_warps_reduce, perf worse
# export AMD_TRITON_FWD_DECODE_NUM_STAGES=2 && python bench.py -benchmark_fn flash_attn_with_kvcache --mode fwd
# export AMD_TRITON_FWD_DECODE_NUM_WARPS_FWD=4 && python bench.py -benchmark_fn flash_attn_with_kvcache --mode fwd
# export AMD_TRITON_FWD_DECODE_NUM_WARPS_REDUCE=8 && python bench.py -benchmark_fn flash_attn_with_kvcache --mode fwd

inp_config="-b 64 -hq 64 -hk 16 -sq 4096 -sk 16384 -d 64"

echo "" > res.out
for m in 16 32 64 128; do
    for n in 16 32 64 128; do
        echo "$m $n"
        echo "AMD_TRITON_FWD_DECODE_BLOCK_M=$m" >> res.out
        echo "AMD_TRITON_FWD_DECODE_BLOCK_N=$n" >> res.out
        export AMD_TRITON_FWD_DECODE_BLOCK_M=$m && \
        export AMD_TRITON_FWD_DECODE_BLOCK_N=$n && \
        python bench.py -benchmark_fn flash_attn_with_kvcache --mode fwd $inp_config >> res.out
        sleep 1
    done
done