import tensorflow as tf

def func():
    """
    会话的基本操作
    :return:
    """
    # 占位符
    a = tf.placeholder(tf.int32)
    b = tf.placeholder(tf.int32)
    # a = tf.constant(1)
    # b = tf.constant(2)
    c = tf.add(a,b,name="add_c")
    

    with tf.Session() as sess:
        # 可以一次run多个op,需要多个参数去接
        # feed_dict可以给前面占位符填充数据
        res_a,res_b,res_c = sess.run([a,b,c],feed_dict={a:10,b:20})
        print("a的结果：{}".format(res_a))
        print("b的结果：{}".format(res_b))
        print("add结果：{}".format(res_c))
        # 也可以用eval()得到结果，和run的作用一样
        # print(c.eval())

func()


