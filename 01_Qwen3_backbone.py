import re
from pathlib import Path

import torch
import torch.nn as nn
from transformers.models.qwen3 import Qwen3ForCausalLM

# Qwen3-0.6B模型结构配置
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,  # 词表大小
    "max_position_embeddings": 40_960,  # 最大上下文长度
    "hidden_size": 1024,  # 模型隐藏向量维度
    "intermediate_size": 3072,  # FFN中间层维度
    "hidden_act": "silu",  # FFN中的激活函数，Qwen3使用SwiGLU结构
    "num_hidden_layers": 28,  # Transformer Block层数
    "num_attention_heads": 16,  # Query注意力头数
    "num_key_value_heads": 8,  # Key/Value注意力头数，GQA会让多个Q头共享KV头
    "head_dim": 128,  # 每个注意力头的维度
    "qk_norm": True,  # 是否需要对Query和Key进行归一化
    "rope_theta": 1_000_000.0,  # RoPE中的theta base
    "rms_norm_eps": 1e-6,  # RMSNorm中的epsilon
    "tie_word_embeddings": True,  # 输入embedding和输出lm head是否共享权重
    "torch_dtype": torch.bfloat16,  # 低精度dtype，用于降低显存占用
}


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 输入层：token_embedding
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], dtype=cfg["torch_dtype"])

        # transformer block
        # nn.ModuleList：model.parameters()会自动注册其中子模块的参数
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["num_hidden_layers"])]
        )
        self.final_norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.out_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False, dtype=cfg["torch_dtype"])

        if cfg["tie_word_embeddings"]:
            # 权重共享
            self.out_head.weight = self.tok_emb.weight

        # 没有传入head_dim时，默认使用hidden_size // num_attention_heads作为head_dim
        if cfg["head_dim"] is None:
            head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        else:
            head_dim = cfg["head_dim"]
        # 获取到每个位置的每组向量的旋转的余弦值和正弦值
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_theta"],
            context_length=cfg["max_position_embeddings"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # 追踪KV Cache 当前位置的索引

    def forward(self, in_idx, cache=None):
        # in_idx：shape:[batch_size, num_tokens]
        # 前向传播

        # 输入层：获取到token_embedding
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds  # shape: [batch_size, num_tokens, hidden_size / embed_dim]

        num_tokens = x.shape[1]
        if cache is not None and len(cache) > 0:  # decode阶段
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
                # 例如pos_start=20,pos_end=21时，只取第20个位置这一行，并允许它关注0到20号位置
            )[pos_start:pos_end, :pos_end]
        else:  # prefill阶段
            pos_start = 0
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
            self.current_pos = num_tokens

        # 扩展mask到[1, 1, query_len, key_value_len]，方便广播到每个batch和每个head
        mask = mask.unsqueeze(0).unsqueeze(0)

        for i, block in enumerate(self.trf_blocks):
            # prefill阶段，cache传进来的是空字典{}；decode阶段，cache里已经有每一层的KV Cache
            blk_cache = cache.get(i) if cache is not None else None
            # x : shape [batch_size, seq_len, embedding_dim]
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                # 对字典中每一层赋值K和V的Cache
                cache[i] = new_blk_cache

        # 输出前先进行层归一化
        x = self.final_norm(x)
        # 输出层
        # logits.shape: [batch_size, num_tokens, vocab_size]
        logits = self.out_head(x.to(self.cfg["torch_dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # GQA注意力层
        self.att = GroupedQueryAttention(
            d_in=cfg["hidden_size"],
            num_heads=cfg["num_attention_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["num_key_value_heads"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["torch_dtype"],
            rms_norm_eps=cfg["rms_norm_eps"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.norm2 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # 保存输入，用于后面进行残差连接（GQA的残差连接）
        shortcut = x
        # x : (8 * 68 * 1024)
        x = self.norm1(x)
        # attention
        # x : (8 * 68 * 1024)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos,
                                 cache=cache)  # Shape [batch_size, num_tokens, hidden_size]
        x = x + shortcut  # 进行残差连接

        # 保存输入，用于后面进行残差连接（FFN的残差连接）
        shortcut = x
        x = self.norm2(x)
        # FFN
        x = self.ff(x)
        x = x + shortcut  # 进行残差连接

        return x, next_cache


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], dtype=cfg["torch_dtype"], bias=False)
        self.up_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], dtype=cfg["torch_dtype"], bias=False)
        self.down_proj = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], dtype=cfg["torch_dtype"], bias=False)
        assert cfg["hidden_act"] == "silu", "This teaching implementation only supports silu for SwiGLU"

    def forward(self, x):
        gate = nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class GroupedQueryAttention(nn.Module):
    def __init__(
            self,
            d_in,  # embdding_dim  1024
            num_heads,  # 多头注意力头数  16
            num_kv_groups,  # kv cache 每组的头数 8
            head_dim=None,  # 多头注意力维度  128
            qk_norm=False,  # qwen自带的归一化操作
            dtype=None,  # 数据类型
            rms_norm_eps=1e-6  # rmsnorme 添加的常数
    ):
        super().__init__()
        # 需要满足：多头注意力满足kv分组
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        # 多头注意力头数  16
        self.num_heads = num_heads
        # kv cache 每组的头数 8
        self.num_kv_groups = num_kv_groups
        # 组内头数 2
        self.group_size = num_heads // num_kv_groups
        # 判断是 GQA 还是 MQA
        if head_dim is None:
            # 需要满足 输入总维度1024 // 组内头数 16 整除 不分组的话一个维度 64
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        # 多头注意力维度  128
        self.head_dim = head_dim
        # 多头注意力输出维度 16 * 128 = 2048
        self.d_out = num_heads * head_dim

        # q_proj层参数为全量的：d_in输入，d_out输出
        # 一个token向量，d_in=1024，经过q_proj运算之后，得到的向量维度：d_out
        # Qwen3-0.6B中：d_out = num_attention_heads * head_dim = 16 * 128 = 2048
        # 此时可以将2048维向量拆成16个Q头，每个Q头128维
        # 1024 映射成 2048维，用于后续拆分头
        # 要求 2048 // num_heads 16 = head_dim 头维度 128
        self.q_proj = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        # k_proj和v_proj的输出头数更少，d_in输入，num_kv_groups * head_dim输出
        # Qwen3-0.6B中：num_key_value_heads * head_dim = 8 * 128 = 1024
        # 这时可以将1024维向量拆成8个K头/V头，每个头128维
        # 1024 -> 8 * 128
        self.k_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        # 输出投影层：融合多头信息，将维度从2048，重新输出成1024维
        self.o_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # [batch_size, seq_len, embedding_dim]
        # (8,68,1024)
        b, num_tokens, _ = x.shape
        # 获取Q, K, V
        # (8,68,16*128)
        queries = self.q_proj(x)  # (b, num_tokens, num_heads * head_dim)
        # (8,64,8*128)
        keys = self.k_proj(x)  # (b, num_tokens, num_kv_groups * head_dim)
        # (8,64,8*128)
        values = self.v_proj(x)  # (b, num_tokens, num_kv_groups * head_dim)

        # Queries分成num_heads个头，每个头的维度为head_dim
        # 16头
        # 新queries shape: (b, self.num_heads,num_tokens , self.head_dim)
        # (8,16,68,128)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # Keys和Values分成num_kv_groups个组，每个组的维度为head_dim
        # # 新keys/values  shape: (b, self.num_kv_groups,num_tokens , self.head_dim)
        # (8,8,68,128)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # (8,8,68,128)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # 额外的归一化
        # qwen创新
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # 重点：对queries和keys应用旋转位置编码
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            # 把算出来的新的key和value和前面token的k和v拼起来
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        # 复制Keys和Values，使每组内的多个Query头共用同一组Keys和Values
        # (8,8,68,128) -> (8,16,68,128)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        # (8,8,68,128) -> (8,16,68,128)
        values = values.repeat_interleave(self.group_size, dim=1)
        # query, keys, values, shape: batch_size, num_heads, num_tokens, head_dim

        # 注意力得分计算
        # keys: [batch_size, num_heads, num_tokens , head_dim ]
        # keys。transpose: [batch_size, num_heads, head_dim , num_tokens ]
        # (8,16,68,128) @ (8,16,128,68)
        # (8,16,68,68)
        attn_scores = queries @ keys.transpose(2, 3)
        # 需要对score做因果掩码，使得每个token只能关注自己和前面的token
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
        # attn_weights: [batch_size, num_heads, query_len, query_len]
        # values: [batch_size, num_heads, query_len, head_dim]
        # (attn_weights @ values)结果：[batch_size, num_heads, query_len, head_dim]
        # context转置后shape: [batch_size, query_len, num_heads, head_dim]
        # reshape: 去掉num_heads: d_out: num_heads* head_dim
        # (8,16,68,68) @ (8,16,68,128) -> (8,16,68,128)->(8,68,16,128)->(8*68*16*128 2048)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        # o_proj降维：转换成(b, num_tokens, d_in(hidden_size))
        # o_proj： 1、将多个简单拼接的V向量，有机融合在一起， 2、将d_out的向量（num_heads * head_dim）转换回 hidden_size
        # (8*68*2048) -> (8*68*1024)
        return self.o_proj(context), next_cache


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """
    计算每个位置的余弦值和正弦值，用于旋转位置编码
    head_dim假设等于8：
    [0,1,2,3,4,5,6,7]
    常见二维旋转讲解里会写成相邻配对：[0,1],[2,3],[4,5],[6,7]
    我们这里的实现：[0,4],[1,5],[2,6],[3,7]
    因此，此处计算出，每个序列位置的token的每个维度处，cos和sin的值，后面直接使用该值进行计算
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 1. 生成频率下标：0, 2, 4, ..., head_dim-2，总数为head_dim//2
    # torch.arange: 生成一维张量，[)
    freq_indices = torch.arange(0, head_dim, 2, dtype=dtype)

    # 2. 转成浮点，并除以 head_dim，得到指数
    exponents = freq_indices.float() / head_dim

    # 3. 计算 theta_base 的这些指数次幂
    scales = theta_base ** exponents

    # 4. 取倒数，得到 inverse frequencies
    inv_freq = 1.0 / scales

    # 计算位置索引：[0,1,2,3,...,context_length-1] shape: (context_length,)
    positions = torch.arange(context_length, dtype=dtype)

    # 计算每个位置的每组旋转的角度
    # positions.unsqueeze(1): (context_length,1)
    # inv_freq.unsqueeze(0): (1,head_dim // 2)
    # angles: (context_length, head_dim // 2)
    # angles[0,0]=序列中第0个位置处，第0组分量的旋转角度，对应的就是m * theta_i
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # 将angles扩展到head_dim维度，shape: (context_length, head_dim)
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)
    # head_dim=8时，一行会变成：[angle_0, angle_1, angle_2, angle_3, angle_0, angle_1, angle_2, angle_3]
    # 这样刚好匹配apply_rope里的前半/后半配对方式

    # 预计算每个角度的余弦值和正弦值
    cos = torch.cos(angles)  # 得到序列当中每个位置，每个分量的余弦值
    sin = torch.sin(angles)  # 得到序列当中每个位置，每个分量的正弦值

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    """
    使用计算好的cos和sin来计算旋转位置编码之后的x，注意，此处获取分组向量不是使用紧挨着的两个值构造而成的，
    如果 head_dim = 8，此处的配对关系就是：
                    (0, 4)
                    (1, 5)
                    (2, 6)
                    (3, 7)
    而不是常见的：
                    (0, 1)
                    (2, 3)
                    (4, 5)
                    (6, 7)
    Args:
        x: 需要应用旋转位置编码的张量，shape为[batch_size, num_heads, seq_len, head_dim]
        cos: 预计算好的余弦值张量，shape为[max_position_embeddings, head_dim]
        sin: 预计算好的正弦值张量，shape为[max_position_embeddings, head_dim]
        offset: 当前这段token在完整序列中的起始位置
    Returns:
        应用RoPE后的张量，shape与x相同

    具体例子：
        假设某个token在某个head上的向量为：
            x = [x0, x1, x2, x3, x4, x5, x6, x7]
        当前position对应的角度为：
            angles = [a0, a1, a2, a3, a0, a1, a2, a3]
        那么cos和sin分别是：
            cos = [cos(a0), cos(a1), cos(a2), cos(a3), cos(a0), cos(a1), cos(a2), cos(a3)]
            sin = [sin(a0), sin(a1), sin(a2), sin(a3), sin(a0), sin(a1), sin(a2), sin(a3)]
        rotate_half(x)会先把x拆成前后两半：
            x1 = [x0, x1, x2, x3]
            x2 = [x4, x5, x6, x7]
        再拼成：
            rotate_half(x) = [-x4, -x5, -x6, -x7, x0, x1, x2, x3]
        最终计算：
            x_rotated = x * cos + rotate_half(x) * sin
        展开后就是：
            x0' = x0*cos(a0) - x4*sin(a0)
            x4' = x4*cos(a0) + x0*sin(a0)

            
            x1' = x1*cos(a1) - x5*sin(a1)
            x5' = x5*cos(a1) + x1*sin(a1)
        
        其他分量同理。它本质上是在二维平面(x0, x4)、(x1, x5)...里按位置角度做旋转。
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    if offset + seq_len > cos.shape[0]:
        raise ValueError(
            f"RoPE position out of range: offset={offset}, seq_len={seq_len}, "
            f"max_position_embeddings={cos.shape[0]}"
        )

    # 将x切分为前一半和后面一半
    x1 = x[..., : head_dim // 2]  # 前面一半：Shape: (batch_size, num_heads, seq_len, head_dim // 2)
    x2 = x[..., head_dim // 2:]  # 后面一半：Shape: (batch_size, num_heads, seq_len, head_dim // 2)

    # cos, sin : shape: [context_length,head_dim]
    # 调整sin和cos的形状：当前这段 token 取出对应位置的 cos 和 sin，并调整 shape 以便广播
    # offset: 10 seq_len: 2: [10,11]
    # offset: 12 seq_len:3 [12,13,14]
    # offset=0时，表示从序列开头开始取
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)

    # 应用旋转变换：rotate_half(x) = [-后半部分, 前半部分]
    rotated = torch.cat((-x2, x1), dim=-1)  # Shape: (batch_size, num_heads,seq_len, head_dim)
    # 二维旋转公式：原来的向量[a,b] ,旋转后的向量[a', b'] = [a cos - b sin, a sin + b cos]
    # 写成向量形式，就是：[a, b] * cos + [-b, a] * sin=[acos-bsin,bcos+asin]
    # 以(x0, x4)举例，旋转后会变成：
    # x0' = x0 * cos - x4 * sin
    # x4' = x0 * sin + x4 * cos
    # 同理：

    # x1' = x1 * cos - x5 * sin
    # x5' = x1 * sin + x5 * cos
    x_rotated = (x * cos) + (rotated * sin)

    # 返回旋转后的x
    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        return norm_x.to(input_dtype)


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path="tokenizer-base.json",
                 apply_chat_template=False,
                 add_generation_prompt=False,
                 add_thinking=False):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file '{tok_path}' not found. Please ensure it's available."
            )

        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # 对齐HF行为：chat模型使用<|im_end|>，base模型使用<|endoftext|>
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"  # 不插入<think>标签，只换行
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s


def generate_text(input_ids, model: Qwen3Model, tokenizer: Qwen3Tokenizer, max_len: int = 100):
    # 每次调用时，重置一下current_pos位置的值
    model.reset_kv_cache()
    generated_token = 0
    final_output = input_ids.clone()
    kv_cache = {}

    with torch.no_grad():
        # 1、prefill阶段
        # output_logits: shape:[batch_size,seq_len,vocab_size]
        output_logits = model(input_ids, cache=kv_cache)

        logits = output_logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        generated_token += 1

        # 2、decode阶段

        next_input = next_token_id.unsqueeze(-1)

        final_output = torch.cat([final_output, next_input], dim=-1)
        while generated_token < max_len:
            # 当前KV_Cache就不是空字典了，当前这个dict中的键，就是不同的TransformerBlock的层，值就是这一层所对应的KV Cache
            output_logits = model(next_input, kv_cache)

            logits = output_logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_input = next_token_id.unsqueeze(-1)

            final_output = torch.cat([final_output, next_input], dim=-1)

            generated_token += 1

    res_list = final_output[0].tolist()
    print(res_list)

    res = tokenizer.decode(res_list)
    print(res)
    return res


def main():
    """
    测试Qwen3模型的生成流程。
    注意：这里只初始化了模型结构，没有加载预训练权重，所以输出不代表真实Qwen3效果。
    """
    import torch
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=r"model/Qwen3-0.6B/tokenizer.json")
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.eval()
    model.to("cuda")
    input_ids = torch.tensor(tokenizer.encode("你好，今天天气真好啊")).unsqueeze(0).to("cuda")
    output = generate_text(input_ids, model, tokenizer)

    print(output)


if __name__ == "__main__":
    main()
