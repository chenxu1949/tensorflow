import tensorflow as tf

tf.python_io.TFRecordWriter()
def func():
    """
    tensorflow实现线性回归的优化
    实现对损失函数的优化，得到最接近0.8的w,和最接近0.7的b
    y = 0.8x + 0.7
    :return:
    """
    # 1.数据集准备
    # 特征值
    # varibale_scope命名空间是为了给重要的步骤模块化，最后在tensorboard显示时方便查看
    with tf.variable_scope("original_data"):
        x = tf.random_normal(shape=(10,1),mean=0,stddev=1)

        # 目标值(真实值)
        y_true = tf.matmul(x,[[0.8]]) + 0.7

    # 2.建立线性模型
    # y = wx + b
    # 3.随机初始化w1和b1(变量op)
    with tf.variable_scope("linear_model"):
        w = tf.Variable(initial_value=tf.random_normal(shape=(1,1)))
        b = tf.Variable(initial_value=tf.random_normal(shape=(1,1)))
        # 预测值
        y_predict = tf.matmul(x,w) + b

    # 4.损失函数（均方误差）
    with tf.variable_scope("loss"):
        error = tf.reduce_mean(tf.square(y_predict-y_true))

    # 5.梯度下降优化损失
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 收集变量(将变量收集起来，方便在tensorboard中显示)
    tf.summary.scalar("error",error)  # scalar收集低维变量，2维以下
    tf.summary.histogram("w",w)   # histogram收集高维变量
    tf.summary.histogram("b",b)

    # 合并变量
    merge = tf.summary.merge_all()

    # 初始化变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("随机初始化权重为：{}，偏置为：{}".format(w.eval(),b.eval()))

        #创建文件事件
        file_writer = tf.summary.FileWriter("./",graph=sess.graph)

        # 训练模型（跑优化器optimizer）
        for i in range(100):
            sess.run(optimizer)
            print("第{}步的误差为{}，权重为{}，偏置为{}".format(i,error.eval(),w.eval(),b.eval()))

            # 运行合并变量
            summary = sess.run(merge)
            file_writer.add_summary(summary,i)


func()


