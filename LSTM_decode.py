import torch
import numpy as np
import torch.nn as nn
from LSTM_model import M_LSTM
from BPE_handle.word22id import interchange, word2id, id2word, get_word, replace_word

PATH_MODEL = "./LSTM_train_state/train_state_73000.pth"
PATH_INDEX = "F:/PyTorch学习/BPE_handle/en_index.npy"


# PATH_INDEX = "./LSTM_file/en_index.npy"

def load(srcf):
    v = np.load(srcf, allow_pickle=True).item()  # 字典的导入
    return v


if __name__ == '__main__':
    en2index = load(PATH_INDEX)
    en = get_word(en2index)

    sentence = input("请输入测试语句:").strip().split()
    # print(sentence)

    sentence = replace_word(sentence, en)
    print(sentence)

    word2id_l = word2id(en2index, sentence)
    # print(word2id)

    model = M_LSTM(len(en2index))

    net_state_dict = torch.load(PATH_MODEL, map_location='cpu')
    model.load_state_dict(net_state_dict)
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    decode_step = len(sentence)
    predict_step = len(word2id_l)
    h_state= None
    cell = None
    result = []

    # 先将整句话过一遍decode
    for i in range(decode_step):
        words = torch.LongTensor([word2id_l[i]]).to(device)
        output, h_state, cell = model.decode(words, h_state, cell)

    for i in range(predict_step):
        words = torch.LongTensor([word2id_l[i]]).to(device)  # 每次取一个词的索引
        output, h_state, cell = model.decode(words, h_state, cell)  # 解码获得最可能的下一个词
        result.append(output.item())

    result_words = id2word(en2index, result)

    print("预测句子结果为:{}".format(" ".join(result_words).replace("@@ ", "")))
