import torch
import triton
import triton.language as tl
import pytest
import pdb
from flash_attn.flash_attn_triton_amd.utils import check_is_fp8

@triton.jit
def many_ops_triton(x_ptr,
                    y_ptr,
                    o_ptr,
                    M: tl.constexpr,
                    K: tl.constexpr,
                    N: tl.constexpr,
                    mult: tl.constexpr,
                    IMITATE_PYTORCH: tl.constexpr,
                    DTYPE: tl.constexpr,
                    DO_MULTIPLY: tl.constexpr,
                    DO_SIGMOID: tl.constexpr,
                    DO_COS: tl.constexpr,
                    DO_EXPONENT: tl.constexpr,
                    DO_SQRT: tl.constexpr
                ):
    """
    x_ptr: pointer to an (M, K) tensor [input]
    y_ptr: pointer to an (K, N) tensor [input]

    o_ptr: pointer to an (M, N) tensor [output]

    M: int matrix shape
    K: int matrix shape
    N: int matrix shape

    mult: multiplication factor for multiplication operation

    IMITATE_PYTORCH: {
        0: no casting after ops, 
        1: cast to original dtype after every op
    }
    DTYPE: {
        0: fp16, 
        1: fp32, 
        2: fp64
    }
    """
    # Set input dtype (we will cast back to this for the output)
    input_dtype = tl.float8e4b8 if DTYPE == 2 else tl.float16 if DTYPE==0 else tl.float32 if DTYPE==1 else None

    x_block_range = tl.arange(0, M)[:, None]*K + tl.arange(0, K)[None, :]
    y_block_range = tl.arange(0, K)[:, None]*N + tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + x_block_range)
    y = tl.load(y_ptr + y_block_range)

    # Multiply
    if DO_MULTIPLY:
        x = x * mult
        y = y * mult
        if IMITATE_PYTORCH:
            x = x.to(input_dtype)
            y = y.to(input_dtype)

    # Sigmoid
    if DO_SIGMOID:
        x = tl.sigmoid(x.to(tl.float32)) # +0.0 cause tl.sigmoid requires a fp32 and 0.0 is fp32 by default so if dtype if fp16 will become fp32
        y = tl.sigmoid(y.to(tl.float32))
        if IMITATE_PYTORCH:
            x = x.to(input_dtype)
            y = y.to(input_dtype)

    # Cos
    if DO_COS:
        x = tl.cos(x.to(tl.float32))     # +0.0 because requires fp32 or fp64
        y = tl.cos(y.to(tl.float32))
        if IMITATE_PYTORCH:
            x = x.to(input_dtype)
            y = y.to(input_dtype)

    # Exponentiate
    if DO_EXPONENT:
        log2_e = 1.4426950408889634  # log2(e)
        x = tl.exp2(log2_e * x)
        y = tl.exp2(log2_e * y)
        if IMITATE_PYTORCH:
            x = x.to(input_dtype)
            y = y.to(input_dtype)

    # Sqrt
    if DO_SQRT:
        x = tl.sqrt(x.to(tl.float32))    # +0.0 because requires fp32 or fp64
        y = tl.sqrt(y.to(tl.float32))
        if IMITATE_PYTORCH:
            x = x.to(input_dtype)
            y = y.to(input_dtype)

    # Matmul
    o_block_range = tl.arange(0, M)[:, None]*N + tl.arange(0, N)[None, :]
    o = tl.dot(x, y) # tl.dot always outputs input dtype. ALSO REQUIRES INPUT SHAPES M >= 16, N >= 16 and K >= 16

    if IMITATE_PYTORCH:
        x = x.to(input_dtype)
        y = y.to(input_dtype)

    # o = tl.dot(x, y, out_dtype=input_dtype) # FUSE CAST INTO DOT

    tl.store(o_ptr + o_block_range, o)

def many_ops_torch(x: torch.Tensor,
                   y: torch.Tensor,
                   out: torch.Tensor,
                   M: int,
                   K: int,
                   N: int,
                   mult: float,
                   DO_MULTIPLY: bool,
                   DO_SIGMOID: bool,
                   DO_COS: bool,
                   DO_EXPONENT: bool,
                   DO_SQRT: bool
                ):
    
    # import pdb; pdb.set_trace()
    
    # Multiply
    if DO_MULTIPLY:
        x = x * mult
        y = y * mult

    # Sigmoid
    if DO_SIGMOID:
        x = torch.sigmoid(x)
        y = torch.sigmoid(y)

    # Cos
    if DO_COS:
        x = torch.cos(x)
        y = torch.cos(y)

    # Exponentiate
    if DO_EXPONENT:
        x = torch.exp(x)
        y = torch.exp(y)

    # Sqrt
    if DO_SQRT:
        x = torch.sqrt(x)
        y = torch.sqrt(y)

    # Matmul
    out[:] = torch.matmul(x, y) # stores in place

@pytest.mark.parametrize("seed", [i for i in range(1)])  # seed for rand num generator
@pytest.mark.parametrize("M", [16, 32])
@pytest.mark.parametrize("K", [16, 32, 64]) # 64 seems to cause some issues
@pytest.mark.parametrize("N", [16, 32])
@pytest.mark.parametrize("mult", [0.7972]) # mult = [0, 2.99]
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float8_e4m3fnuz]) # torch.float32
@pytest.mark.parametrize("IMITATE_PYTORCH", [0]) # 0 = no casting (not imitating pytorch), 1 = cast after every op (imitating pytorch)
@pytest.mark.parametrize("DO_MULTIPLY", [0])  # Include multiplication
@pytest.mark.parametrize("DO_SIGMOID", [0])  # Include sigmoid
@pytest.mark.parametrize("DO_COS", [0])  # Include cosine
@pytest.mark.parametrize("DO_EXPONENT", [1])  # Include exponentiation
@pytest.mark.parametrize("DO_SQRT", [0])  # Include square root
def test_many_ops(seed, M, K, N, mult, dtype, IMITATE_PYTORCH, DO_MULTIPLY, DO_SIGMOID, DO_COS, DO_EXPONENT, DO_SQRT):
    """
    Test reproducability of PyTorch results with a Triton kernel implementing various math operations.

    Each operation can be individually enabled or disabled using the respective parameters. The test will compare
    the results from Triton and PyTorch to ensure they match within a specified tolerance.

    Args:
        seed (int): Random seed for reproducibility.
        M (int): Number of rows for the first input tensor.
        K (int): Number of columns for the first input tensor and rows for the second.
        N (int): Number of columns for the second input tensor.
        mult (float): Multiplication factor for the input tensors.
        dtype (torch type): the dtype of the tensors
        IMITATE_PYTORCH (int): If 1, cast tensors back to their original dtype after each operation, if 0 does not cast until very end.
        DO_MULTIPLY (int): If 1, include multiplication in the operations, if 0 does not.
        DO_SIGMOID (int): If 1, include sigmoid activation in the operations, if 0 does not.
        DO_COS (int): If 1, include cosine transformation in the operations, if 0 does not.
        DO_EXPONENT (int): If 1, include exponentiation in the operations, if 0 does not.
        DO_SQRT (int): If 1, include square root in the operations, if 0 does not.
    """

    # Misc parameters
    torch.set_printoptions(precision=6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)

    torch_type_to_id = {
        torch.float16: 0,
        torch.float32: 1,
        torch.float8_e4m3fnuz: 2
    }

    DTYPE = torch_type_to_id[dtype]

    x = torch.rand(M, K, dtype=torch.float32, device=device)
    y = torch.rand(K, N, dtype=torch.float32, device=device)
    x, y = x.to(dtype), y.to(dtype)

    grid = (1,)
    out = torch.zeros(M, N, dtype=dtype, device=device)

    if check_is_fp8(x):
        out_torch = torch.zeros(M, N, dtype=torch.float16, device=device)
    else:
        out_torch = torch.zeros(M, N, dtype=dtype, device=device)

    with torch.cuda.device(x.device):
        many_ops_triton[grid](x, y, out, M, K, N, mult, IMITATE_PYTORCH, DTYPE, DO_MULTIPLY, DO_SIGMOID, DO_COS, DO_EXPONENT, DO_SQRT)
        if check_is_fp8(x):
            many_ops_torch(x.to(torch.float16), y.to(torch.float16), out_torch, M, K, N, mult, DO_MULTIPLY, DO_SIGMOID, DO_COS, DO_EXPONENT, DO_SQRT)
        else:
            many_ops_torch(x, y, out_torch, M, K, N, mult, DO_MULTIPLY, DO_SIGMOID, DO_COS, DO_EXPONENT, DO_SQRT)

    # print("torch - triton", (out_torch-out))
    if check_is_fp8(x):
        out = out.to(torch.float16)

    print(f'absolute error: {(out-out_torch).abs().max().item()}, relative error: {((out-out_torch)/out).abs().max().item()}')

    assert torch.allclose(out, out_torch, atol=1e-6, rtol=1e-5), f'absolute error: {(out-out_torch).abs().max().item()}, relative error: {((out-out_torch)/out).abs().max().item()}' # tensors must match exactly