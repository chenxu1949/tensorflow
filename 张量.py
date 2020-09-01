import tensorflow as tf

def func():
    """
    张量的基本操作
    :return:
    """
    a = tf.zeros(shape=[2,3,4],dtype=tf.float32)
    b = tf.random_normal(shape=[3,4,5],dtype=tf.float32,mean=0,stddev=1)
    c = tf.constant(5.0,dtype=tf.float32,shape=[2,3])

    # 修改张量形状必须要在定义op变量时用占位符
    # set_shape静态修改张量形状
    op_a = tf.placeholder(dtype=tf.int32,shape=[None,2])
    op_b = tf.placeholder(dtype=tf.int32,shape=[None,None])
    op_c = tf.placeholder(dtype=tf.int32,shape=[1,2])

    # 用set_shape修改形状，修改后就不能再改变了，就是说只能修改None
    op_a.set_shape([2,3])

    # 用reshape修改形状后只是添加了个新的张量
    op_c_new = tf.reshape(op_c,[2,1])

    print("修改后的op_a:",op_a.shape)
    print("op_c:",op_c.shape)
    print("op_c_new:",op_c_new.shape)

    # 开启会话
    with tf.Session() as sess:
        print(sess.run(a))
        print(sess.run(b))
        print(sess.run(c))



func()

