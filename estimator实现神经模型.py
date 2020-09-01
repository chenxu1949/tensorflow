import tensorflow as tf
import functools

# tf.train.AdamOptimizer()   Adam优化算法能自适应调节学习率的大小

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]


train_file = "./data/adult.data"
test_file = "./data/adult.test"

def input_func(file,epoches,batch_size):
    """
    获取数据集
    :param file:
    :param epoches:重复训练的次数
    :param batch_size:mini-bacth梯度下降算法的每批样本个数
    :return:
    """
    def deal_with_scv(value):
       # 获取value
        data = tf.decode_csv(value,record_defaults=_CSV_COLUMN_DEFAULTS)
        feature_dict = dict(zip(_CSV_COLUMNS,data))
        labels = feature_dict.pop('income_bracket')
        classes = tf.equal(labels,'>50k')

        return feature_dict,classes

    dataset = tf.data.TextLineDataset(file)
    dataset = dataset.map(deal_with_scv)
    dataset = dataset.repeat(epoches)
    dataset = dataset.batch(batch_size)

    return dataset

def get_feature_column():
    """
    构造特征
    :return:
    """
    # 连续型特征（数值类）
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")

    numeric_columns = [age, education_num, capital_gain, capital_loss, hours_per_week]

    # 离散型特征（类别类）
    relationship = tf.feature_column.categorical_column_with_vocabulary_list("relationship",
                 ['Husband', 'Not-in-family', 'Wife','Own-child', 'Unmarried','Other-relative'])

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # 如果不清楚该特征下有什么类别，或不想一个个将所有类别列出来，用哈希分桶
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",
                                                                       hash_bucket_size=1000)

    # 神经网络模型的类别类特征要进行embedding转换成词向量
    categorical_columns = [tf.feature_column.embedding_column(relationship, dimension=6),
                           tf.feature_column.embedding_column(occupation, dimension=1000),
                           tf.feature_column.embedding_column(education, dimension=16),
                           tf.feature_column.embedding_column(marital_status, dimension=7),
                           tf.feature_column.embedding_column(workclass, dimension=9)]

    return numeric_columns + categorical_columns

def get_feature_column_v2():
    """
    进行分桶与特征交叉优化后
    :return:
    """
    # 连续型特征（数值类）
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")

    numeric_columns = [age, education_num, capital_gain, capital_loss, hours_per_week]

    # 离散型特征（类别类）
    relationship = tf.feature_column.categorical_column_with_vocabulary_list("relationship",
                                                                             ['Husband', 'Not-in-family', 'Wife',
                                                                              'Own-child', 'Unmarried',
                                                                              'Other-relative'])

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # 如果不清楚该特征下有什么类别，或不想一个个将所有类别列出来，用哈希分桶
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",
                                                                       hash_bucket_size=1000)

    # 神经网络模型的类别类特征要进行embedding转换成词向量
    # categorical_columns = [tf.feature_column.embedding_column(relationship, dimension=6),
    #                        tf.feature_column.embedding_column(occupation, dimension=1000),
    #                        tf.feature_column.embedding_column(education, dimension=16),
    #                        tf.feature_column.embedding_column(marital_status, dimension=7),
    #                        tf.feature_column.embedding_column(workclass, dimension=9)]
    categorical_columns = [relationship,education,marital_status,workclass,occupation]

    # 对连续型特征进行哈希分桶与特征交叉
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
    ]

    return numeric_columns + categorical_columns + crossed_columns

if __name__ == '__main__':

    # 没采用特征交叉的结果
    train_func = functools.partial(input_func,train_file,epoches=3,batch_size=32)
    test_func = functools.partial(input_func,test_file,epoches=1,batch_size=32)

    classifier = tf.estimator.DNNClassifier(feature_columns=get_feature_column(),
                                            hidden_units=[512,256],
                                            optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
                                                                             l1_regularization_strength=10,
                                                                             l2_regularization_strength=10))
    classifier.train(train_func)
    res = classifier.evaluate(test_func)
    for key,value in res.items():
        print("{}:{}".format(key,value))
    print("-------------")


    # 分桶交叉特征之后的结果
    train_func1 = functools.partial(input_func,train_file,epoches=3,batch_size=32)
    test_func1 = functools.partial(input_func,test_file,epoches=1,batch_size=32)

    classifier1 = tf.estimator.LinearClassifier(feature_columns=get_feature_column_v2())
    classifier1.train(train_func1)
    res1 = classifier.evaluate(test_func1)
    for key, value in res1.items():
        print("{}:{}".format(key, value))



