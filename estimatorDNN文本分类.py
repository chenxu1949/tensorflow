import tensorflow as tf
from tensorflow import keras

# 获取文本中5000个不同的词列表，再取它的下标代表每个词[0,1,2,3....4999]
# 训练集和测试集各2500个样本，每个样本的每个词按词列表的下标去排列
# 最后再将每个词embedding一下，转换成词向量，作为特征值
vocab_size = 5000
# 让每个样本的序列长度一致,多截少补0
sentence_size = 200

def get_train_test():
    """
    1.获取训练样本(数据集预处理)
    :return:
    """
    imbd = keras.datasets.imdb
    (x_train,y_train),(x_test,y_test) = imbd.load_data(num_words=vocab_size)

    # 样本序列长度固定
    x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=sentence_size,
                                                        padding='post',value=0)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=sentence_size,
                                                        padding='post',value=0)

    return (x_train,y_train),(x_test,y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = get_train_test()

    # 2.定义input_func
    def parser(x,y):
        features = {'feature':x}
        return features,y

    def train_input_func():
        dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        dataset = dataset.shuffle(buffer_size=25000)
        dataset = dataset.batch(batch_size=64)
        dataset = dataset.map(parser)
        # repeat为空的话就是一直训练
        dataset = dataset.repeat()
        return dataset

    def test_input_func():
        dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        dataset = dataset.batch(64)
        dataset = dataset.map(parser)
        return dataset

    # 3.指定特征列
    column = tf.feature_column.categorical_column_with_identity('feature',vocab_size)
    embedding_size = 50
    word_embedding_column = tf.feature_column.embedding_column(column,dimension=embedding_size)

    # 4.模型训练与评估
    classifier = tf.estimator.DNNClassifier(hidden_units=[100],
                                            feature_columns=[word_embedding_column],
                                            model_dir='./model/cnn_word_embedding/')
    classifier.train(input_fn=train_input_func,steps=1000)
    res = classifier.evaluate(input_fn=test_input_func)
    for key,value in res.items():
        print("{}:{}".format(key,value))
