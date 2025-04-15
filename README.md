# bert-transformer
<meta charset="UTF-8">
BERT 凭借其双向编码结构，能够深度捕捉文本的上下文信息，生成极为丰富且准确的词向量表示。Transformer模型以其独特的自注意力机制，能够有效处理序列中的长距离依赖关系，在学习学生答题序列模式、挖掘知识关联方面具有显著优势。

将 BERT 与 Transformer 相结合应用于知识追踪。BERT 生成的词向量作为 Transformer 模型的输入，为其提供更具语义丰富性的特征，使得模型能够更好地融合问题文本信息与学生答题数据，进而实现学生答题情况的准确预测。

项目框架
===

├── dataset // 数据集存放目录

│ ├── 2017 (1.csv 

│ └── 数学题库 final.csv

├── bert.py // 负责 BERT 模型的加载、文本数据处理以及词向量生成等功能

├── transformer1.py // 定义 Transformer 架构的编码器、解码器以及相关组件

├── dataload.py // 实现数据加载和预处理逻辑

├── mask.py // 生成各种掩码（如填充掩码、后续掩码），用于处理序列数据填充值和防止模型看到未来信息

├── train.py // 包含模型训练主要逻辑，定义训练函数、计算损失并更新模型参数的 Python 脚本

├── main.py // 项目主程序文件，负责参数解析、数据加载、模型训练、评估和测试等全流程控制的 Python 脚本

└── README.md // 本项目说明文档，介绍项目整体情况、使用方法等内容
    
项目模块
==
BERT 模块
--
文本特征提取：利用预训练的 BERT 模型，对题目文本（Q_Title）进行深度语义理解与特征提取。通过 BERT 分词器将文本转化为模型可处理的输入格式，然后获取文本的词向量表示，这些向量蕴含了丰富的语义信息，为后续的模型学习提供基础。

Transformer 模块
--
编码器（Encoder）：负责对学生的答题序列数据进行编码。包括问题 ID、技能 ID、答题正确性、答题时间、题目生成的词向量等多维度信息，通过嵌入层将这些信息转化为向量表示，并结合位置编码，使模型能够捕捉序列中的位置信息。经过多层多头注意力机制和前馈神经网络的处理，提取出有效的特征表示。

解码器（Decoder）：从输入数据中提取目标问题序列、目标技能序列、目标问题编号等信息，经过嵌入和位置编码等处理后得到 dec_output。dec_output 与编码器的输出（enc_output）一起输入解码器中生成最终学生能否答对该题目的预测结果。在生成过程中，利用掩码机制防止模型提前看到未来的信息，保证预测的合理性。
