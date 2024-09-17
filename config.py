# 存放模型训练、数据处理等配置信息
class DefaultConfig(object):
    # 数据集路径
    data_path = 'data/'
    # 模型保存路径
    model_path = 'checkpoints/'
    # 训练集路径
    train_path = data_path + 'train.txt'
    # 测试集路径
    test_path = data_path + 'test.txt'
    # 验证集路径
    val_path = data_path + 'val.txt'
    # 类别文件路径
    class_path = data_path + 'class.txt'
    # 类别数
    num_classes = 10
    # 词表大小
    vocab_size = 5000
    # 每个句子的长度
    seq_length = 200
    # batch大小
    batch_size = 64
    # epoch数
    num_epochs = 10
    # 学习率
    learning_rate = 0.001
    # 词向量维度
    embedding_dim = 128
    # 隐藏层大小
    hidden_dim = 128
    # dropout
    dropout = 0.5
    # 保存模型的步长
    save_per_step = 100
    # 打印信息的步长
    print_per_step = 20
    # 是否使用gpu
    use_gpu = True
    # gpu编号
    gpu = 0
    # 是否训练
    is_train = True