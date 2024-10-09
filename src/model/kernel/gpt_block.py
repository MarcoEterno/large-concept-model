from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


if __name__ == '__main__':
    def explore_block_speed():
        from src.model.config import GPTConfig, DecoderConfig
        import torch
        config = DecoderConfig

        device = "mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # set a seed for reproducibility
        torch.manual_seed(42)
        block = Block(config).to(device)
        x = torch.randn(20, 100, config.n_embd, device=device)
        x = block(x)
        # time the block function
        with torch.autograd.profiler.profile(use_device = 'cpu') as profiler:
            for i in range(100):
                x = block(x)
        print(profiler.key_averages().table(sort_by="self_cpu_time_total"))
        print(x.shape)

    explore_block_speed()

"""
/opt/conda/envs/home/bin/python /home/marco.eterno/large-concept-model/src/model/gpt_block.py 
Using device: cuda
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       cudaMalloc        48.50%      77.787ms        48.50%      77.787ms     163.762us           475  
                                      aten::addmm        13.07%      20.963ms        45.03%      72.214ms     180.535us           400  
                                 cudaLaunchKernel         6.64%      10.643ms         6.64%      10.643ms      10.643us          1000  
                          aten::native_layer_norm         4.57%       7.324ms         8.48%      13.593ms      67.967us           200  
                                      aten::empty         4.12%       6.609ms         7.54%      12.090ms      12.090us          1000  
                                       aten::view         4.04%       6.484ms         4.04%       6.484ms       4.052us          1600  
                                        aten::add         3.35%       5.368ms        11.44%      18.355ms      91.774us           200  
                                     aten::linear         2.72%       4.368ms        53.20%      85.332ms     213.331us           400  
                                       aten::gelu         2.35%       3.775ms        13.22%      21.203ms     212.033us           100  
                                  aten::transpose         1.78%       2.852ms         2.50%       4.005ms       3.338us          1200  
    aten::_scaled_dot_product_efficient_attention         1.30%       2.084ms         8.88%      14.238ms     142.379us           100  
               aten::_efficient_attention_forward         1.27%       2.039ms         7.13%      11.442ms     114.423us           100  
                                          aten::t         1.13%       1.805ms         2.04%       3.274ms       8.185us           400  
                                 aten::as_strided         0.96%       1.536ms         0.96%       1.536ms       1.024us          1500  
               aten::scaled_dot_product_attention         0.88%       1.414ms         9.76%      15.652ms     156.522us           100  
                                      aten::split         0.75%       1.205ms         1.70%       2.731ms      27.309us           100  
    cudaOccupancyMaxActiveBlocksPerMultiprocessor         0.61%     977.281us         0.61%     977.281us       2.443us           400  
                            cudaStreamIsCapturing         0.56%     895.917us         0.56%     895.917us       1.558us           575  
                                     aten::narrow         0.40%     649.379us         0.95%       1.526ms       5.085us           300  
                                    aten::reshape         0.40%     641.850us         1.69%       2.711ms       6.777us           400  
                                      aten::slice         0.31%     493.220us         0.55%     876.134us       2.920us           300  
                                 aten::layer_norm         0.29%     470.293us         8.77%      14.064ms      70.318us           200  
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 160.386ms

torch.Size([20, 100, 768])

SAME OUTPUT BUT WITH CUDA PROFILER:

/opt/conda/envs/home/bin/python /home/marco.eterno/large-concept-model/src/model/gpt_block.py 
Using device: cuda
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       cudaMalloc        30.87%      80.303ms        30.87%      80.303ms     169.059us       0.000us         0.00%       0.000us       0.000us           475  
                                  cudaEventRecord        18.43%      47.948ms        18.43%      47.948ms       2.788us       0.000us         0.00%       0.000us       0.000us         17200  
                                      aten::addmm         8.31%      21.615ms        28.52%      74.190ms     185.475us     220.960ms        74.55%     220.960ms     552.400us           400  
                                     aten::linear         6.15%      16.005ms        44.78%     116.498ms     291.244us       8.118ms         2.74%     241.901ms     604.753us           400  
                          aten::native_layer_norm         4.18%      10.869ms        10.68%      27.778ms     138.889us       8.397ms         2.83%      11.553ms      57.765us           200  
                                 cudaLaunchKernel         3.91%      10.170ms         3.91%      10.170ms      10.170us       0.000us         0.00%       0.000us       0.000us          1000  
                                  aten::transpose         3.84%       9.982ms         5.90%      15.351ms      12.793us       8.956ms         3.02%      12.859ms      10.716us          1200  
                            cudaDeviceSynchronize         3.12%       8.116ms         3.12%       8.116ms       8.116ms       0.000us         0.00%       0.000us       0.000us             1  
                                       aten::view         2.90%       7.545ms         2.90%       7.545ms       4.716us       5.046ms         1.70%       5.046ms       3.154us          1600  
                                      aten::empty         2.65%       6.887ms         4.87%      12.667ms      12.667us       3.222ms         1.09%       3.222ms       3.222us          1000  
                                        aten::add         2.08%       5.405ms         7.24%      18.828ms      94.139us       3.199ms         1.08%       3.199ms      15.995us           200  
               aten::_efficient_attention_forward         1.93%       5.024ms         6.13%      15.955ms     159.549us      12.779ms         4.31%      14.076ms     140.760us           100  
    aten::_scaled_dot_product_efficient_attention         1.83%       4.760ms        10.27%      26.726ms     267.265us       2.433ms         0.82%      20.786ms     207.860us           100  
                                          aten::t         1.60%       4.162ms         4.14%      10.772ms      26.929us       2.990ms         1.01%       7.292ms      18.230us           400  
                                       aten::gelu         1.45%       3.772ms         8.30%      21.595ms     215.947us       4.076ms         1.38%       4.076ms      40.760us           100  
                                    aten::reshape         1.17%       3.044ms         2.56%       6.661ms      16.652us       3.018ms         1.02%       4.303ms      10.758us           400  
                                      aten::split         1.11%       2.885ms         4.12%      10.726ms     107.264us       1.602ms         0.54%       7.057ms      70.570us           100  
                                     aten::narrow         0.92%       2.386ms         2.65%       6.885ms      22.950us       2.249ms         0.76%       5.455ms      18.183us           300  
                                      aten::slice         0.83%       2.150ms         1.36%       3.534ms      11.780us       2.221ms         0.75%       3.206ms      10.687us           300  
                                 aten::as_strided         0.74%       1.930ms         0.74%       1.930ms       1.287us       4.888ms         1.65%       4.888ms       3.259us          1500  
               aten::scaled_dot_product_attention         0.69%       1.796ms        11.09%      28.852ms     288.522us     748.000us         0.25%      21.534ms     215.340us           100  
                                 aten::layer_norm         0.62%       1.615ms        11.55%      30.055ms     150.275us       1.500ms         0.51%      13.053ms      65.265us           200  
    cudaOccupancyMaxActiveBlocksPerMultiprocessor         0.44%       1.154ms         0.44%       1.154ms       2.885us       0.000us         0.00%       0.000us       0.000us           400  
                            cudaStreamIsCapturing         0.25%     651.774us         0.25%     651.774us       1.134us       0.000us         0.00%       0.000us       0.000us           575  
-------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 260.175ms
Self CUDA time total: 296.402ms

torch.Size([20, 100, 768])

"""