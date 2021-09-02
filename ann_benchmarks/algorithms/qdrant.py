import time

from qdrant_client import QdrantClient
from qdrant_openapi_client.models.models import Distance, CollectionStatus, StorageOperationsAnyOf1, \
    StorageOperationsAnyOf1UpdateCollection, OptimizersConfigDiff, HnswConfig, HnswConfigDiff, SearchParams

from ann_benchmarks.algorithms.base import BaseANN


class Qdrant(BaseANN):

    def __init__(self, metric, index_args):
        self.name = "qdrant"
        self.collection_name = "benchmark_collection"
        self.client = QdrantClient()
        self.ef_construction = index_args["efConstruction"]
        self.m = index_args["M"]
        self.ef_search = 128

    def set_query_arguments(self, ef):
        self.ef_search = ef

    def fit(self, X):
        print("Konifugracja: ef_construction: {}, M: {}, ef_search: {}".format(self.ef_construction, self.m, self.ef_search))
        self.upload_data(data=X, vector_size=X.shape[1],ef_construct=self.ef_construction, m=self.m)
        self.wait_collection_green()
        self.enable_indexing()
        time.sleep(0.5)

        print(self.client.openapi_client.collections_api.get_collection(self.collection_name).dict())

        wait_for_index_time = self.wait_collection_green()
        print("Waited for index: ", wait_for_index_time)

        print(self.client.openapi_client.collections_api.get_collection(self.collection_name).dict())

    def query(self, q, n):
        res = self.client.search(
            self.collection_name,
            query_vector=q,
            top=n,
            append_payload=False,
            search_params=SearchParams(hnsw_ef=self.ef_search)
        )
        return [x.id for x in res]

    def batch_query(self, X, n):
        pass

    def get_batch_results(self):
        pass

    def upload_data(self, data, vector_size, ef_construct, m, parallel=4):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vector_size=vector_size,
            distance=Distance.EUCLID,
            hnsw_config=HnswConfigDiff(
                ef_construct=ef_construct,
                m=m
            ),
            optimizers_config=OptimizersConfigDiff(
                flush_interval_sec=10,
                indexing_threshold=1000000000,  # Disable indexing before all points are added
                memmap_threshold=1000000000,
                payload_indexing_threshold=1000000000,
                max_segment_number=4
            )
        )

        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=data,
            payload=None,
            ids=None,
            parallel=parallel
        )

    def wait_collection_green(self):
        wait_time = 10.0
        total = 0
        collection_info = self.client.openapi_client.collections_api.get_collection(self.collection_name)
        while collection_info.result.status != CollectionStatus.GREEN:
            time.sleep(wait_time)
            total += wait_time
            collection_info = self.client.openapi_client.collections_api.get_collection(self.collection_name)
        return total

    def enable_indexing(self):

        self.client.openapi_client.collections_api.update_collections(
            StorageOperationsAnyOf1(update_collection=StorageOperationsAnyOf1UpdateCollection(
                name=self.collection_name,
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000,
                    max_segment_number=6
                )
            ))
        )