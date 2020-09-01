import tensorflow as tf
def func():
    """
    Variable定义变量才能将内容持久化,在模型中可被训练
    :return:
    """
    # 定义变量
    a = tf.Variable(initial_value=30)
    b = tf.Variable(initial_value=20)
    c = tf.add(a,b)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(c))

func()
