"""
@Author : Seren
@Date : 2026/7/1 11:48
@Description:
DPO训练SFT后的qwen3模型
"""
import math
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DPOConfig:
    epoch: int = 2,
    batch_size: int = 2,
    # base_model_dir: str = "../model/Qwen3-0.6B",
    # sft训练好的模型
    reference_model_dir: str = "../fine_tuning/sft",
    # dpo模型输出地址
    dpo_model_dir: str = "../fine_tuning/dpo",
    # dpo 偏好学习一般比较困难设置小一点
    lr: float = 1e-6,
    # 预热占余弦衰减的比例
    warmup_ratio: float = 0.1,
    # 取训练样本中的前train_data_size条，全部的太多，这里只取一部分
    train_data_size: int = 5000,
    log_dir = "../log/dpo_log",
    # 日志轮次
    log_iter: int = 100


class CosineDecayWithWarmup(LRScheduler):
    """
    带预热的余弦衰减学习率调度器
    """

    def __init__(self, optimizer, total_batch, warmup_ratio=0.1, last_epoch=-1):
        self.total_batch = total_batch
        self.warmup_ratio = warmup_ratio
        self.warmup_batch = int(total_batch * warmup_ratio)
        super(CosineDecayWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_batch:
            # 预热阶段：线性增长
            return [base_lr * self.last_epoch / self.warmup_batch for base_lr in self.base_lrs]
        else:
            # 余弦衰减阶段
            progress = (self.last_epoch - self.warmup_batch) / (self.total_batch - self.warmup_batch)
            # 防止 progress 超过 1（训练可能超出 total_batch）
            progress = min(progress, 1.0)
            decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * decay for base_lr in self.base_lrs]


def get_train_data(dpo_config, tokenizer):
    """
    加载数据，调用chat template方法，得到分词之后的input_ids；后面在train主循环当中会调用该方法，获取训练数据
    """
    from datasets import load_dataset

    train_data = load_dataset("../data/ultrafeedback_binarized")["train_prefs"]
    chosen_result_list = []
    rejected_result_list = []

    for i in range(dpo_config.train_data_size):
        chosen_message_list: List = train_data[i]["chosen"]
        chosen_message_list.insert(0, {"role": "system", "content": "you are a helpfule assistant"})

        chosen_result: Dict[str, List] = tokenizer.apply_chat_template(chosen_message_list, tokenize=True,
                                                                       truncation=True, max_length=2500)
        chosen_result_list.append(chosen_result["input_ids"])

        rejected_message_list: List = train_data[i]["rejected"]
        rejected_message_list.insert(0, {"role": "system", "content": "you are a helpfule assistant"})

        rejected_result: Dict[str, List] = tokenizer.apply_chat_template(rejected_message_list, tokenize=True,
                                                                         truncation=True, max_length=2500)
        rejected_result_list.append(rejected_result["input_ids"])

    return chosen_result_list, rejected_result_list


def create_answer_mask(input_ids, tokenizer):
    """
    创建answer mask，从input_ids当中找出assistant回答的部分，然后输出一个与input_ids相同shape的mask，
    后续将其与pad_mask进行逻辑与操作，得到最终的mask，用以计算损失
    """

    # 构建answer mask，输入的input_ids为批量 tokenize之后的数据，对于每一条数据，查找当中assistant回答的部分，将其设置为1

    # 1. 构造一个和input_ids相同shape的全0矩阵
    answer_mask = torch.zeros_like(input_ids)

    # 2. 遍历input_ids中的每一条数据，查找assistant回答的部分，将其设置为1
    eos_token_id = tokenizer.encode('<|im_end|>')[0]
    # input_ids.shape: batch_size, seq_len
    for idx, ids in enumerate(input_ids):
        # 获取到所有的eos_position
        eos_position: List = torch.where(ids == eos_token_id)[0].tolist()
        print(eos_position)
        # 排除第一个eos_position: 第一个对应的是system prompt
        eos_position = eos_position[1:]
        # 解析获得user_ends和assistant_ends
        user_ends, assistant_ends = _parse_conversation_turns(eos_position)
        # 设置answer mask
        _set_answer_masks(answer_mask[idx], user_ends, assistant_ends)

        # 结果返回:
    return answer_mask


def _parse_conversation_turns(eos_positions: List[int]):
    """
    输入eos_positions，输出user所对应的end位置和assistant所对应的end位置。

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么输入的eos_token_id为：[10,15,20,25]
    user_turns为从第一个开始取（具体索引位置需要加一，因为eos_token_id后面还有一个\n换行符），每隔一个取一次，assistant_turns为从第二个开始取，每隔一个取一次。

    输出结果为：
        user_turns:[11,21]
        assistant_ends:[16,26]
    """

    use_ends = [pos + 1 for pos in eos_positions[::2]]
    assistant_ends = [pos + 1 for pos in eos_positions[1::2]]

    return use_ends, assistant_ends


def _set_answer_masks(mask, user_ends, assistant_ends):
    """
    将mask当中，assistant回答的部分，设置为1（原地修改，不返回新的mask），其余部分保持为0

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么user_turns:[11,21]，assistant_ends:[16,26]

    user_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    assistant_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    要想获取到assistant的回答的起始位置，就需要再跳过\n,<|im_start|>,assistant 这三个token，所以需要加3.
    要想获取到assistant的回答的结束位置，就需要往前跳一个<|im_end|>，所以需要减1.
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    if num_user_turns == num_assistant_turns:
        for user_end, assistant_end in zip(user_ends, assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1

    elif num_user_turns == num_assistant_turns + 1:
        for user_end, assistant_end in zip(user_ends[:-1], assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1

        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1]
        last_answer_start = last_user_end + 3
        mask[last_answer_start:] = 1


def _compute_log_prob(output_logits, chosen_labels, assistant_mask):
    """
    output_logits: 模型预测输出ids :[batch_size,seq_len,vocab_size]
    [
        [  # 第0层（样本1）
            [0.2,  1.5, -0.3, 0.8,  0.1],   # 位置0
            [2.1,  0.4,  0.7, -0.5, 1.2],   # 位置1
            [0.0, -1.0,  1.8, 0.5,  0.9]    # 位置2
        ],
        [  # 第1层（样本2）
            [0.9, -0.2,  1.1, 2.3,  0.3],   # 位置0
            [1.7,  0.6,  0.1, 0.4,  2.5],   # 位置1
            [0.3,  1.2, -0.1, 1.9,  0.8]    # 位置2
        ]
    ]
    chosen_labels: 模型真实值 :[batch_size, seq_len]
    chosen_assistant_mask: prefill掩码 :[batch_size,seq_len]

    """
    # 先转换为概率
    # [batch_size,seq_len,vocab_size]
    output_probs = torch.log_softmax(output_logits, dim=-1)
    # [batch_size,seq_len]
    # 不要求[batch_size, seq_len]长度相同,可以取n-1个,但是下面需要取相同才能哈达玛积
    # 结果应该是:[batch_size,seq_len]
    output_labels = torch.gather(
        output_probs, dim=-1, index=chosen_labels
    )
    # 先乘以prefill掩码,哈达玛积.得到真实值出现的概率
    # [batch_size,seq_len]
    output_log_prob = output_labels * assistant_mask
    # 归一化之后的对数概率
    # (batch_size,)
    average_log_prob = torch.sum(output_log_prob, dim=-1) / assistant_mask.sum(dim=-1)

    return average_log_prob


def compute_loss(chosen_log_prob, rejected_log_prob, reference_chosen_log_prob, reference_rejected_log_prob, beta):
    """
    DPO计算损失的函数
    args:
        chosen_log_prob: 当前模型输出chosen的对数概率，shape: (batch_size,)
        rejected_log_prob：当前模型输出rejected的对数概率，shape(batch_size,)
        reference_chosen_log_prob: 参考模型输出chosen的对数概率，shape: (batch_size,)
        reference_rejected_log_prob：参考模型输出rejected的对数概率，shape: (batch_size,)
        beta: 用于控制模型区分chosen和rejected的强度
    """
    # (batch_size, )
    margin = (chosen_log_prob - rejected_log_prob) - (reference_chosen_log_prob - reference_rejected_log_prob)
    loss = -torch.nn.functional.logsigmoid(beta * margin)
    loss = loss.sum() / len(loss)

    return loss


def save_model_tokenizer(model, tokenizer, save_dir):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def train(dpo_config: DPOConfig):
    # 1.准备基础资料
    # 1.1 模型相关
    # 参考模型
    ref_model = AutoModelForCausalLM.from_pretrained(dpo_config.reference_model_dir)
    # 训练模型
    model = AutoModelForCausalLM.from_pretrained(dpo_config.reference_model_dir)
    # 参考模型要预测模式
    ref_model.eval()
    # 训练模型进入训练模式
    model.train()
    # 将两个模型放到cuda上
    model.to("cuda")
    ref_model.to("cuda")
    # tokenzier
    tokenizer = AutoTokenizer.from_pretrained(dpo_config.reference_model_dir)

    # 1.2 数据相关
    # 准备训练偏好数据集
    # 偏好、负面数据集
    chosen_train_data, rejected_train_data = get_train_data(dpo_config, tokenizer)
    # 1.3 训练相关
    # 优化器 用来梯度更新
    optimizer = AdamW(model.parameters(), lr=dpo_config.lr)
    # 调度器 用来调整学习率
    # cv里通常是epoch级别更新，llm通常是batch级别更新
    scheduler = CosineDecayWithWarmup(
        optimizer,
        total_batch=len(chosen_train_data),
        warmup_ratio=0.1
    )
    # 1.4 日志相关
    # tensorboard
    writer = SummaryWriter(log_dir=dpo_config.log_dir)
    # tqdm 总共的轮次
    # 防止最后一轮训练数据不足batch_size，确保训练数据能被batch_size整除 5001+2-1 //2 = 2500
    total_batch = (len(chosen_train_data) + dpo_config.batch_size - 1) // dpo_config.batch_size
    progress_bar = tqdm(total=total_batch)
    # 记录loss
    total_loss_list = []

    # 2.训练
    for epoch in range(dpo_config.epoch):
        for batch in tqdm(range(total_batch)):
            # 2.1 获取数据
            # 获取两个倾向数据
            # 2.1.1 获取正向数据chosen_input_ids和chosen_labels、chosen_assistant_mask
            # 对加载数据集进行batch的切分,长度seq_len
            current_chosen_batch_data = chosen_train_data[
                batch * dpo_config.batch_size: (batch + 1) * dpo_config.batch_size]
            # 2.1.1 同一个batch做padding
            # 获取当前batch中的最大长度
            current_chosen_max_len = max([len(data) for data in current_chosen_batch_data])
            # padding
            for data in current_chosen_batch_data:
                padding_length = current_chosen_max_len - len(data)
                data.extend([tokenizer.pad_token_id] * padding_length)

            # 截取target和 source
            # shape: [batch_size, seq_len]
            chosen_batch_data_tensor = torch.tensor(current_chosen_batch_data, dtype=torch.long).to("cuda")
            # [batch_size, seq_len-1]
            chosen_input_ids = chosen_batch_data_tensor[:, :-1]
            chosen_labels = chosen_batch_data_tensor[:, 1:]
            # 做prefill
            # 创造assistant掩码,用于覆盖模型生成的assistant
            # [batch_size,seq_len]
            chosen_assistant_mask = create_answer_mask(input_ids=chosen_input_ids, tokenizer=tokenizer)

            # 2.1.2 获取反向数据rejected_input_ids和rejected_labels,rejected_assistant_mask
            current_rjected_batch_data = rejected_train_data[
                batch * dpo_config.batch_size: (batch + 1) * dpo_config.batch_size]

            current_rejected_max_len = max([len(data) for data in current_rjected_batch_data])

            for data in current_rjected_batch_data:
                padding_length = current_rejected_max_len - len(data)
                data.extend([tokenizer.pad_token_id] * padding_length)

            # shape: batch_size, seq_len+1
            rejected_batch_data_tensor = torch.tensor(current_rjected_batch_data, dtype=torch.long).to("cuda")
            # shape: batch_size, seq_len
            rejected_input_ids = rejected_batch_data_tensor[:, :-1]
            # shape: batch_size, seq_len
            rejected_labels = rejected_batch_data_tensor[:, 1:]
            rejected_assistant_mask = create_answer_mask(input_ids=rejected_input_ids, tokenizer=tokenizer)

            # 2.2 获取模型输出
            # shape: batch_size, seq_len, vocab_size
            chosen_output_logits = model(chosen_input_ids).logits
            rejected_output_logits = model(rejected_input_ids).logits
            # 不计算梯度,参考模型前向传播
            with torch.no_grad():
                reference_chosen_output_logits = ref_model(chosen_input_ids).logits
                reference_rejected_output_logits = ref_model(rejected_input_ids).logits
            # 2.3 计算loss
            chosen_log_prob = _compute_log_prob(chosen_output_logits, chosen_labels, chosen_assistant_mask)
            rejected_log_prob = _compute_log_prob(rejected_output_logits, rejected_labels, rejected_assistant_mask)

            reference_chosen_log_prob = _compute_log_prob(reference_chosen_output_logits, chosen_labels,
                                                          chosen_assistant_mask)
            reference_rejected_log_prob = _compute_log_prob(reference_rejected_output_logits, rejected_labels,
                                                            rejected_assistant_mask)

            loss = compute_loss(chosen_log_prob, rejected_log_prob, reference_chosen_log_prob,
                                reference_rejected_log_prob, dpo_config.beta)
            total_loss_list.append(loss.item())
            # 2.4 反向传播
            loss.backward()
            # 2.5 优化器更新参数
            optimizer.step()
            # 2.6 调度器更新学习率
            scheduler.step()
            optimizer.zero_grad()
            # 2.6 模型保存
            save_model_tokenizer(model, tokenizer, dpo_config.save_dir)
            # 2.7 日志记录
            current_lr = scheduler.get_last_lr()
            writer.add_scalar("train/lr", scalar_value=current_lr, global_step=batch)
            should_log = batch % dpo_config.log_iter == 0 or batch == total_batch - 1
            if should_log:
                """
                记录一下损失
                """
                loss_list = total_loss_list[-dpo_config.log_iter:]
                average_loss = sum(loss_list) / len(loss_list)
                writer.add_scalar("train/loss", scalar_value=average_loss, global_step=batch)
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix(lr=f"{current_lr:.2e}", loss=f"{loss.item():.4f}")
        return model, tokenizer
