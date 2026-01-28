import multiprocessing
import time


def worker(stop_event):
    print('工人在努力搬砖...')
    # 只要 event 没有被 set，就一直工作
    while not stop_event.is_set():
        print('搬砖中... +1')
        # 模拟工作耗时，同时每秒检查一次
        # 这里用 wait(timeout) 比 time.sleep 更好，
        # 因为如果 sleep 期间收到停止信号，wait 会立即响应，不需要等 sleep 结束
        stop_event.wait(timeout=1)

    print('工人收到下班信号，清理现场，结束进程。')


if __name__ == '__main__':
    stop_event = multiprocessing.Event()

    p = multiprocessing.Process(target=worker, args=(stop_event, ))
    p.start()

    # 让工人工作 3 秒
    time.sleep(3)

    print('主进程：通知工人下班（设置 Event）')
    stop_event.set()

    p.join()
    print('主进程：所有工作结束')
