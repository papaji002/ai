from tensorflow import compat
tf = compat.v1
tf.compat.v1.disable_eager_execution()
print(f"TensorFlow Version: {tf.__version__}")

print("\nAddition code:")
a = tf.constant(5)
b = tf.constant(3)
with tf.Session() as sess:
    sum = sess.run(a+b)
print(f"Result : {sum}")

print("\nString code:")
x = tf.constant('Bhola')
y = tf.constant(' Study')
with tf.Session() as sess:
    result = sess.run(x+y)
print(f"Result : {result}")

print("\nMetrics code:")
mat1 = tf.fill((5,5),8)
zeros = tf.zeros((5,5))
one = tf.ones((4,4))
myops = [mat1, zeros, one]
sess = tf.InteractiveSession()
for op in myops:
    print(sess.run(op),end='\n\n')

"""OUTPUT:-
    TensorFlow Version: 2.13.0

    Addition code:
    2023-09-22 11:06:37.119339: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-09-22 11:06:37.146973: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:375] MLIR V1 optimization pass is not enabled
    Result : 8

    String code:
    Result : b'Bhola Study'

    Metrics code:
    [[8 8 8 8 8]
    [8 8 8 8 8]
    [8 8 8 8 8]
    [8 8 8 8 8]
    [8 8 8 8 8]]

    [[0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0.]]

    [[1. 1. 1. 1.]
    [1. 1. 1. 1.]
    [1. 1. 1. 1.]
    [1. 1. 1. 1.]]
"""
