from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import time
from dask.distributed import progress


cluster = SLURMCluster(queue='par-multi', cores=10, memory='1000',
                       job_mem='2000')


def slow_increment(x):
    time.sleep(1)
    x = x + 1
    return x


cluster.scale(10)
client = Client(cluster)

futures = client.map(slow_increment, range(1000))
progress(futures)