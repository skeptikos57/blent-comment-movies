import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("CUDA available:", tf.test.is_built_with_cuda())
print("GPU devices available:", tf.config.list_physical_devices('GPU'))

# Test simple pour vérifier que le GPU fonctionne
if tf.config.list_physical_devices('GPU'):
    print("\n✅ GPU détecté et fonctionnel!")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("Test de multiplication matricielle sur GPU:", c)
else:
    print("\n❌ Aucun GPU détecté")