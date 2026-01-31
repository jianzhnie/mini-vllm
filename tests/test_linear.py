import unittest

from minivllm.models.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    divide,
)


class TestLinearLayers(unittest.TestCase):

    def test_divide(self):
        self.assertEqual(divide(10, 2), 5)
        with self.assertRaises(ValueError):
            divide(10, 3)

    def test_column_parallel_linear_init(self):
        # Mock dist
        # Since we can't easily mock dist in this environment without multiple processes,
        # we'll test the logic that doesn't depend on dist or assumes rank 0/size 1 (default fallback).
        linear = ColumnParallelLinear(16, 32)
        self.assertEqual(linear.input_size, 16)
        self.assertEqual(linear.output_size, 32)
        self.assertEqual(linear.weight.shape, (32, 16))

    def test_qkv_linear_init(self):
        # hidden=16, head=4, heads=4, kv_heads=4
        linear = QKVParallelLinear(16, 4, 4, 4)
        # output = (4 + 2*4) * 4 = 12 * 4 = 48
        self.assertEqual(linear.weight.shape, (48, 16))

        # MQA: hidden=16, head=4, heads=4, kv_heads=1
        linear_mqa = QKVParallelLinear(16, 4, 4, 1)
        # output = (4 + 2*1) * 4 = 6 * 4 = 24
        self.assertEqual(linear_mqa.weight.shape, (24, 16))


if __name__ == '__main__':
    unittest.main()
