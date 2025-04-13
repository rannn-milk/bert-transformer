import argparse
import os

from torch.utils.data import DataLoader

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from bert import load_data,load_bert_model,get_bert_embeddings,TextDataset
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
from sklearn import metrics

from dataload import readdata
from train import train, test

from transformer1 import Encoder, Decoder
import torch.nn.functional as F
import torch.nn as nn
from mask import get_pad_mask, get_subsequent_mask


def main():
    '''
	The main function of the training script
	'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Script for KT experiment')

    parser.add_argument('--epoch_num', type=int, default=30,
                        help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of batch')
    parser.add_argument('--session_size', type=int, default=16,
                        help='number of sessions')
    parser.add_argument('--action_size', type=int, default=64,
                        help='number of actions in each session')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='embedding dimensions')
    parser.add_argument('--learning_rate', type=int, default=5e-5,
                        help='learning rate')
    parser.add_argument('--d_inner', type=int, default=2048,
                        help='FFN dimension')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--n_head', type=int, default=4,
                        help='number of head for multihead attention')
    parser.add_argument('--d_k', type=int, default=64,
                        help='k query dimensions')
    parser.add_argument('--d_v', type=int, default=64,
                        help='v query dimensions')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='dropout')
    parser.add_argument('--dataset', type=str, default='数学题库final',
                        help='dropout')

    params = parser.parse_args()
    dataset_name = f'./dataset/{params.dataset}.csv'

    df = pd.read_csv(dataset_name, low_memory=False)

    # Create Variable
    batch_size = params.batch_size
    session_size = params.session_size
    embedding_size = params.embedding_size
    action_size = params.action_size
    padding_correct_value = 2
    padding_node = 0

    num_problem = max(df.problemId.unique().tolist()) + 3  # (padding, EOS, Session EOS)
    num_skill = max(df.skill.unique().tolist()) + 3  # (padding, EOS, Session EOS)
    num_qd = max(df.Q_D.unique().tolist()) + 3  # (padding, EOS, Session EOS)
    num_stuno = max(df.StudentID.unique().tolist()) + 3  # (padding, EOS, Session EOS)

    # -- EOS value
    EOS_quesiton = num_problem - 2
    EOS_skill = num_skill - 2
    EOS_C_value = padding_correct_value + 1
    EOS_q_no = (num_qd- 2)
    EOS_stu_no = num_stuno - 2

    session_EOS_question = num_problem - 1
    session_EOS_skill = num_skill - 1
    session_EOS_c_value = EOS_C_value + 1
    SEOS_q_no = num_qd - 1
    SEOS_stu_no = num_stuno - 1

    BOS_c_Value = session_EOS_c_value + 1

    n_type_correctness = 6  # -- (0, 1, padding, EOS, Session EOS, BOS)

    # -- model parameter
    learning_rate = params.learning_rate
    d_inner = params.d_inner

    n_layers = params.n_layers
    n_head = params.n_head
    d_k = params.d_k
    d_v = params.d_v
    dropout = params.dropout

    epoch_num = params.epoch_num
    seed_no = 123

    column_name = 'Q_Title'  # 需要处理的列名
    model_path = './chinese-roberta-ext-large/'  # 模型路径
    batch_size = 4  # 批量大小
    max_length = 128  # BERT 最大长度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检测设备

    # Step 1: 加载数据
    data = load_data(dataset_name, column_name)
    # Step 2: 加载 BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # Step 3: 创建 Dataset 和 DataLoader
    dataset = TextDataset(data, tokenizer, column_name, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Step 4: 加载 BERT 模型
    model = load_bert_model(model_path)
    # Step 5: 获取 BERT 词向量
    embeddings = get_bert_embeddings(dataloader, model, device)
    # 输出前 5 个样本的词向量形状
    print(f"Generated embeddings shape: {embeddings.shape}")
    print("Example embeddings for the first 5 texts:")
    print(embeddings[:5])

    training_array, val_array, test_array = readdata(df=df, padding_node=padding_node,
                                                     padding_correct_value=padding_correct_value,
                                                     action_size=action_size, session_size=session_size,
                                                     EOS_quesiton=EOS_quesiton, EOS_skill=EOS_skill,
                                                     EOS_C_value=EOS_C_value, BOS_c_Value=BOS_c_Value,
                                                     session_EOS_question=session_EOS_question,
                                                     session_EOS_skill=session_EOS_skill,
                                                     session_EOS_c_value=session_EOS_c_value,
                                                     EOS_q_no=EOS_q_no, EOS_stu_no=EOS_stu_no,
                                                     SEOS_q_no=SEOS_q_no, SEOS_stu_no=SEOS_stu_no)

    training_array = torch.LongTensor(training_array)
    val_array = torch.LongTensor(val_array)
    test_array = torch.LongTensor(test_array)

    print('train size:', training_array.size())
    print('val size:', val_array.size())
    print('test size:', test_array.size())


    my_model = transformer2(embedding_size=embedding_size, d_model=embedding_size, d_inner=d_inner,
                            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            action_position=action_size,
                            session_position=session_size, n_type_correctness=n_type_correctness,
                            n_type_problem=num_problem,
                            n_type_skill=num_skill, session_head=n_head, session_layer=n_layers, n_type_qno=num_qd)
    my_model = my_model.to(device)
    input_array = training_array  # Or any other input data
    optimizer1 = torch.optim.Adam(my_model.parameters(), betas=(0.9, 0.999), lr=learning_rate, eps=1e-8)

    train_loss_list = []
    train_AUC_list = []
    val_AUC_list = []
    val_loss_list = []
    best_valid_auc = 0
    for epoch in range(epoch_num):
        loss, Plabel, Ground_true = train(my_model=my_model, optimizer=optimizer1, training_array=training_array,
                                          device=device, batch_size=batch_size, session_size=session_size,
                                          padding_correct_value=padding_correct_value, EOS_C_value=EOS_C_value,
                                          padding_node=padding_node, EOS_skill=EOS_skill)

        one_AUC = metrics.roc_auc_score(Ground_true.detach().numpy().astype(int), Plabel.detach().numpy())

        print('train_one_epoch: ', loss, 'train_AUG:', one_AUC)
        train_loss_list.append(loss)
        train_AUC_list.append(one_AUC)

        # scheduler.step()
        print('------------------------------')
        print('------------------------------')

        val_loss, val_Plabel, val_Ground_true = test(my_model=my_model, optimizer=optimizer1, test_array=val_array,
                                                     device=device, batch_size=batch_size, session_size=session_size,
                                                     padding_correct_value=padding_correct_value,
                                                     EOS_C_value=EOS_C_value,
                                                     padding_node=padding_node, EOS_skill=EOS_skill)

        val_AUC = metrics.roc_auc_score(val_Ground_true.detach().numpy().astype(int),
                                        val_Plabel.detach().numpy())
        print('------------------------------')
        print('Epoch: ', epoch, ' val AUC:', val_AUC)
        val_AUC_list.append(val_AUC)
        val_loss_list.append(val_loss)
        print('------------------------------------------')
        '''
        if val_AUC > best_valid_auc:
         #   path = os.path.join(model_path, 'val') + '_*'

            for i in glob.glob(path):
                os.remove(i)

            #best_valid_auc = val_AUC
            #best_epoch = epoch + 1
         '''
        '''torch.save({'epoch': epoch,
                        'model_state_dict': my_model.state_dict(),
                        'optimizer_state_dict': optimizer1.state_dict(),
                        'loss': loss,
                        },
                        os.path.join(model_path,  'val')+'_' + str(best_epoch)
                        )'''

        #if (val_loss - min(val_loss_list)) > 10000:
         #   break
    '''
    print('------------------------------')
    print('train_loss_list', train_loss_list)
    print('train_AUC_list', train_AUC_list)
    print('VAL AUC List:', val_AUC_list)
    print('val loss List:', val_loss_list)
    print('max_val_auc:', max(val_AUC_list))
    print('------------------------------')
    print('Begin to test.........')
    '''
    #checkpoint = torch.load(os.path.join(model_path, 'val') + '_' + str(best_epoch))
    #my_model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_Plabel, test_Ground_true = test(my_model=my_model, optimizer=optimizer1, test_array=test_array,
                                                    device=device, batch_size=batch_size, session_size=session_size,
                                                    padding_correct_value=padding_correct_value,
                                                    EOS_C_value=EOS_C_value,
                                                    padding_node=padding_node, EOS_skill=EOS_skill)

    test_AUC = metrics.roc_auc_score(test_Ground_true.detach().numpy().astype(int),
                                     test_Plabel.detach().numpy())

    print('Test_AUC', test_AUC)
    #print('Best_epoch', best_epoch)

class transformer2(nn.Module):
    def __init__(
            self, embedding_size=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=64, d_k=2, d_v=64, dropout=0.1, action_position=64, session_position=16,
            n_type_correctness=6, n_type_problem=1500, n_type_skill=200,
            session_head=8, session_layer=6, n_type_qno=2000, n_type_kno=2000):
        super().__init__()

        self.embedding_size = embedding_size
        self.d_model = d_model
        self.fc_layer1 = nn.Linear(embedding_size, 1)
        self.cor_pad_emb = nn.Embedding(n_type_correctness, embedding_size)

        self.encoder = Encoder(
            n_type_correctness=n_type_correctness, n_type_problem=n_type_problem,
            n_type_skill=n_type_skill, embedding_size=embedding_size,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            d_model=d_model, d_inner=d_inner, dropout=dropout,
            n_position=action_position, n_type_qno=n_type_qno, n_type_kno=n_type_kno)

        self.decoder = Decoder(
            embedding_size=int(embedding_size),
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            d_model=d_model, d_inner=d_inner, dropout=dropout,
            n_position=action_position,
            n_type_problem=n_type_problem, n_type_skill=n_type_skill,
            n_type_qno=n_type_qno, n_type_kno=n_type_kno)

    def forward(self, input_array, session_size, padding_correct_value, padding_node, EOS_skill,embeddings):
        session_input_array = []
        # 提取输入数组中，前 session_size 个会话的正确性值（即第三维的第2个切片）
        pad_mask = get_pad_mask(input_array[:, :session_size, 2], padding_correct_value)
        # 这一行将提取的输入数组的前 session_size 个元素传递给 Encoder
        enc_out = self.encoder(input_array=input_array[:, :session_size],emb=embeddings ,pad_mask=pad_mask)
        # 提取最后一个时间步的输出
        session_input_array = enc_out[:, :, -1, :]

        #-- Decoder
        # 提取所有批次（:）中指定会话（session_size）的第一个元素（索引为 0）
        target_problem_seq = input_array[:, session_size, 0]
        target_skill_seq = input_array[:, session_size, 1]
        target_qno = input_array[:, session_size, 3]
        target_kno = input_array[:, session_size, 4]

        # Get target mask
        trg_mask = get_subsequent_mask(target_problem_seq) & get_pad_mask(target_skill_seq,padding_node) & get_pad_mask(target_skill_seq,EOS_skill)

        dec_out = self.decoder(trg_problem_seq=target_problem_seq, trg_skill_seq=target_skill_seq,
                               enc_output=session_input_array, trg_mask=trg_mask, pad_mask=trg_mask, target_qno=target_qno,
                               target_kno=target_kno)
        dec_out = self.fc_layer1(dec_out)
        dec_out = nn.Sigmoid()(dec_out.squeeze(2))
        return dec_out


if __name__ == '__main__':
    main()
