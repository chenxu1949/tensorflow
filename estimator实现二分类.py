import tensorflow as tf
import functools


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
    :return:
    """
    def deal_with_csv(value):
        data = tf.decode_csv(value,record_defaults=_CSV_COLUMN_DEFAULTS)

        # 构建返回的字典数据feature_dict, label
        feature_dict = dict(zip(_CSV_COLUMNS,data))
        labels = feature_dict.pop('income_bracket')
        classes = tf.equal(labels,'>50k')

        return feature_dict,classes

    dataset = tf.data.TextLineDataset(file)
    dataset =dataset.map(deal_with_csv)
    dataset = dataset.repeat(epoches)  # 整个循环多少次
    dataset = dataset.batch(batch_size)  # 一次提取多少行样本

    return dataset

def get_feature_column():
    """
    指定estimator输入的特征列类型
    :return:
    """
    # 连续型特征（数值类）
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")

    numeric_columns = [age,education_num,capital_gain,capital_loss,hours_per_week]

    # 离散型特征（类别类）
    relationship = tf.feature_column.categorical_column_with_vocabulary_list("relationship",
            ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])

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

    categorical_columns = [relationship, marital_status, workclass, occupation]

    return numeric_columns + categorical_columns

if __name__ == '__main__':

    # 构造线性二分类模型
    train_func = functools.partial(input_func,train_file,epoches=3,batch_size=32)
    test_func = functools.partial(input_func,test_file,epoches=1,batch_size=32)

    classifier = tf.estimator.LinearClassifier(feature_columns=get_feature_column(),
                                               optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
                                                                                l1_regularization_strength=10,
                                                                                l2_regularization_strength=10))
    classifier.train(train_func)

    # 训练结果和测试结果进行评估
    res = classifier.evaluate(test_func)
    for key,value in res.items():
        print("{}:{}".format(key,value))
