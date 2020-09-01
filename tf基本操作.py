import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def func(x,y):
    """
    tensorflow加法
    :param x:
    :param y:
    :return:
    """
    a = tf.constant(x,name="const_a")
    b = tf.constant(y,name="const_b")
    c = tf.add(a,b,name="add_c")
    print("a的结果：{}".format(a))
    print("b的结果：{}".format(b))
    print("c的结果：{}".format(c))

    # 开启默认的图
    default_g = tf.get_default_graph()
    print("默认的图:{}".format(default_g))
    print(a.graph)
    print(b.graph)
    print(c.graph)

    # 新创建一个图
    new_g = tf.Graph()
    print("自定义图：{}".format(new_g))

    # 在一个新创建的图当中定义变量时要这样写
    with new_g.as_default():
        new_a = tf.constant(1,name="const_new_a")
        new_b = tf.constant(2,name="const_new_b")
        new_c = tf.add(new_a,new_b,name="add_new_c")

    print(new_a.graph)
    print(new_b.graph)
    print(new_c.graph)

    # 开启会话(会话想执行哪个图，graph参数就写哪个图；若想执行多个图，就需开启多个会话)
    with tf.Session() as sess:
        sum0 = sess.run(c)
        print("默认图的结果：{}".format(sum0))
    with tf.Session(graph=new_g) as sess:

        # 将图写入到指定目录，以提供给tensorbord
        # file_writer = tf.summary.FileWriter("./", graph=sess.graph)

        sum = sess.run(new_c)
        print("新图的结果:{}".format(sum))


func(10,20)






