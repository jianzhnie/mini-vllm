# Multiprocessing 多进程编程指南


## Multiprocessing 之 Event 事件同步原语


在 Python 的 `multiprocessing` 模块中，`Event`（事件）是一个非常基础且重要的**进程间同步原语**。

你可以把它想象成一个**跨进程的“红绿灯”或“发令枪”**。

### 1. 如何通俗理解 `Event`？

在多进程编程中，进程之间是相互隔离的（拥有独立的内存空间）。如果你想让 进程 A 告诉 进程 B：“嘿，由于某些条件还没满足，你先暂停一下，等我准备好了你再继续”，你就需要一种同步机制。

`Event` 就是这样一个简单的**状态标志位**（Flag）：

* **本质：** 它内部维护一个布尔值（True 或 False）。
* **初始状态：** 默认为 `False`（红灯/大门关闭）。
* **交互方式：**
* **`wait()`**： 也就是“等红灯”。如果标志位是 `False`，调用这个方法的进程就会**阻塞**（停在这里不往下执行），直到变绿灯。
* **`set()`**： 也就是“变绿灯”。把标志位设为 `True`。此时，所有正在 `wait()` 的进程会瞬间被唤醒，继续执行。
* **`clear()`**： 也就是“变红灯”。把标志位重置为 `False`。
* **`is_set()`**： 只是看一眼现在是红灯还是绿灯（返回 True/False），不会阻塞。



### 2. 核心方法详解

| 方法                           | 说明                                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **`event.wait(timeout=None)`** | **最关键的方法**。如果 `Event` 内部标志为 `False`，程序运行到这就卡住（阻塞）。一旦变成 `True`，或者达到 `timeout` 时间，程序才继续往下走。 |
| **`event.set()`**              | 将内部标志设为 `True`。**唤醒**所有正在 `wait()` 的进程。                                                                                   |
| **`event.clear()`**            | 将内部标志设为 `False`。之后的 `wait()` 调用将再次阻塞。                                                                                    |
| **`event.is_set()`**           | 检查当前是否为 `True`。通常用于循环判断中。                                                                                                 |


### 3. 如何使用：代码示例

#### 场景一：发令枪模式（主进程控制子进程启动）

在这个例子中，子进程启动后不会立刻工作，而是等待主进程的“指令”。

```python
import multiprocessing
import time

def runner(name, event):
    print(f"{name} 正在起跑线上准备...")
    # 进程在这里阻塞，等待 event 被 set
    event.wait()
    print(f"砰！{name} 起跑了！")

if __name__ == "__main__":
    # 创建一个 Event 对象
    start_event = multiprocessing.Event()

    # 启动两个子进程
    p1 = multiprocessing.Process(target=runner, args=("选手A", start_event))
    p2 = multiprocessing.Process(target=runner, args=("选手B", start_event))

    p1.start()
    p2.start()

    print("主进程：正在检查跑道...")
    time.sleep(2) # 模拟准备工作

    print("主进程：发令！")
    # 设置 Event 为 True，瞬间唤醒所有等待的子进程
    start_event.set()

    p1.join()
    p2.join()

```

#### 场景二：优雅停止（作为终止信号）

这是 `Event` 最常用的场景之一：主进程告诉一直在跑循环的子进程“该下班了”。

```python
import multiprocessing
import time

def worker(stop_event):
    print("工人在努力搬砖...")
    # 只要 event 没有被 set，就一直工作
    while not stop_event.is_set():
        print("搬砖中... +1")
        # 模拟工作耗时，同时每秒检查一次
        # 这里用 wait(timeout) 比 time.sleep 更好，
        # 因为如果 sleep 期间收到停止信号，wait 会立即响应，不需要等 sleep 结束
        stop_event.wait(timeout=1)

    print("工人收到下班信号，清理现场，结束进程。")

if __name__ == "__main__":
    stop_event = multiprocessing.Event()

    p = multiprocessing.Process(target=worker, args=(stop_event,))
    p.start()

    # 让工人工作 3 秒
    time.sleep(3)

    print("主进程：通知工人下班（event.set()）")
    stop_event.set()

    p.join()
    print("主进程：所有工作结束")
```


### 4. 关键注意事项

1. **不是用来传数据的**：`Event` 只能传递简单的“是/否”信号。如果你需要传递复杂数据（如列表、对象），请使用 `Queue` 或 `Pipe`。
2. **一次性 vs 可重复**：`Event` 被 `set()` 后会一直保持 `True`，直到有人调用 `clear()`。如果你需要重复使用（例如：红灯->绿灯->红灯），记得在适当的时候调用 `clear()`，否则 `wait()` 就不起作用了（因为一直是绿灯）。
3. **多对多通信**：一个 `Event` 可以被多个进程同时 `wait()`。一旦 `set()`，**所有**等待的进程都会被唤醒（广播机制）。

### 总结

* **理解：** 它是进程间的红绿灯。
* **用途：**
  * 1.  控制启动顺序（等我准备好）。
  * 2.  控制优雅退出（通知子进程结束循环）.




## Multiprocessing 之 共享内存 (Shared Memory)


`multiprocessing.shared_memory` (Python 3.8+ 引入) 是 Python 多进程编程中用来提升性能的**大杀器**，特别是处理大量数据（如图像、矩阵、NumPy 数组）时。

### 1. 如何通俗理解 Shared Memory？

#### 传统多进程 (Queue/Pipe) —— “传真机模式”

通常，进程 A 想把一份大文件给进程 B，它需要把文件复制一份，通过管道传过去，进程 B 再接收。

* **缺点：** 发生了**数据的拷贝**（序列化/反序列化）。如果你有一个 1GB 的 NumPy 数组，传一次就要拷贝 1GB，非常慢且费内存。

#### Shared Memory —— “公共白板模式”

进程 A 申请了一块特殊的内存区域（公共白板），并把名字贴在墙上。进程 B 只要知道名字，就能直接看到这块白板。

* **优点：** **Zero-copy（零拷贝）**。进程 A 往白板上写字，进程 B 瞬间就能看到，不需要复制数据。
* **本质：** 直接映射一段物理内存到多个进程的虚拟地址空间。


### 2. 使用步骤（生命周期）

使用共享内存比 `Event` 或 `Queue` 要复杂，必须严格遵守**生命周期**，否则会导致内存泄漏。

1. **创建 (Create)**: 申请一块指定大小的内存，设定一个名字。
2. **挂载 (Attach)**: 其他进程通过名字找到这块内存，并连接上。
3. **读写 (Read/Write)**: 通过 `buffer` 直接操作二进制数据（通常配合 NumPy 使用）。
4. **关闭 (Close)**: 每个进程用完后，都要“关闭”连接（表示“我不再用它了”）。
5. **解绑 (Unlink)**: **这是最关键的一步！** 当**所有**进程都不用了，必须由**某一个**进程（通常是主进程）销毁这块内存。如果不销毁，这块内存会一直占用系统资源，直到电脑重启。



### 3. 代码示例

#### 场景一：基础用法（读写字节）

```python
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
import time

def worker(shm_name):
    # 2. 子进程挂载：通过名字连接到现有的共享内存
    existing_shm = SharedMemory(name=shm_name)

    # 通过 buffer 读取数据 (memoryview)
    # 注意：buffer 中的数据是二进制 bytes
    data = existing_shm.buf[:5]
    print(f"子进程读到了: {bytes(data).decode('utf-8')}")

    # 修改数据
    existing_shm.buf[:5] = b'World'

    # 4. 子进程关闭连接
    existing_shm.close()

if __name__ == "__main__":
    # 1. 主进程创建：分配 10 字节的内存
    shm = SharedMemory(create=True, size=10)

    # 初始化写入一些数据
    shm.buf[:5] = b'Hello'

    print(f"主进程写入: Hello")

    p = Process(target=worker, args=(shm.name,))
    p.start()
    p.join()

    # 读取子进程修改后的数据
    print(f"主进程再次读取: {bytes(shm.buf[:5]).decode('utf-8')}")

    # 4. 主进程关闭连接
    shm.close()

    # 5. 销毁内存（非常重要！）
    shm.unlink()

```



#### 场景二：进阶用法（配合 NumPy 高效处理大矩阵）

这是 Shared Memory 最核心的用途。如果你做 AI 或数据分析，这个模式非常有用。

```python
import numpy as np
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
import time

def modify_array(shm_name, shape, dtype):
    # 连接共享内存
    existing_shm = SharedMemory(name=shm_name)

    # **关键步骤**：将共享内存的 buffer 包装成 NumPy 数组
    # 这里不需要 copy，直接在原内存上操作
    np_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    print(f"子进程：开始修改数组...")
    # 直接修改数组，主进程会立即看到变化
    np_array[:] = np_array + 100

    existing_shm.close()

if __name__ == "__main__":
    # 创建一个较大的数组 (例如 10个元素)
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)

    # 1. 创建共享内存
    # nbytes 是数组占用的字节数
    shm = SharedMemory(create=True, size=data.nbytes)

    # 2. 创建一个基于共享内存的 NumPy 数组
    shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)

    # 将原始数据复制进去
    shm_array[:] = data[:]

    print(f"主进程：原始数据 {shm_array}")

    # 启动子进程，传入内存名、形状和类型
    p = Process(target=modify_array, args=(shm.name, data.shape, data.dtype))
    p.start()
    p.join()

    print(f"主进程：子进程结束后的数据 {shm_array}")

    shm.close()
    shm.unlink()

```



### 4. 核心注意事项（避坑指南）

1. **资源泄露风险 (Resource Leak)**：
如果你忘记调用 `unlink()`，或者程序中途崩溃了没有执行到 `unlink()`，这块内存会一直留在操作系统里（在 Linux 上通常可以在 `/dev/shm` 目录下看到）。
* *解决办法：* 尽量使用 `try...finally` 块来保证 `unlink()` 一定会被执行。


2. **数据竞争 (Race Condition)**：
共享内存**不包含**锁机制。如果进程 A 正在写数组的第 1 个位置，进程 B 同时也在写第 1 个位置，数据就会乱套。
* *解决办法：* 必须配合 `multiprocessing.Lock` 或 `Event` 一起使用来控制访问顺序。


3. **基本类型限制**：
`SharedMemory` 处理的是底层的 `bytes`。它最适合处理**固定大小**的连续内存块（如 NumPy 数组）。如果你想共享复杂的 Python 对象（如列表、字典），使用 `SharedMemory` 会非常痛苦（需要自己处理序列化），此时建议使用 `multiprocessing.Manager`（虽然慢，但方便）。

### 总结

| 特性         | Queue / Pipe            | Shared Memory           |
| ------------ | ----------------------- | ----------------------- |
| **原理**     | 复制数据 (Pickle)       | 映射同一块物理内存      |
| **速度**     | 慢 (随数据量增加而变慢) | **极快** (零拷贝)       |
| **适用场景** | 传递指令、小数据、消息  | 图像处理、大型矩阵计算  |
| **安全性**   | 进程安全 (自带锁)       | **不安全** (需手动加锁) |
| **易用性**   | 简单                    | 复杂 (需管理生命周期)   |

**下一步建议：**
既然你已经了解了“同步信号”(`Event`) 和 “高性能数据共享”(`SharedMemory`)，如果要处理**多个**子进程并发执行任务并汇总结果，你可能不想手动管理 `Process` 的启动和停止。此时，**进程池 (`multiprocessing.Pool`)** 是最高效的工具。你想了解 `Pool` 的使用吗？
