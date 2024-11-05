from pymongo import MongoClient

# MongoDB 클라이언트와 데이터베이스 초기화
class MongoDBClient:
    def __init__(self, uri="mongodb://localhost:27017", db_name="vector_db", collection_name="vector_store"):
        self.client = MongoClient(uri)
        self.db_name = db_name
        self.collection_name = collection_name
        self.db = self._initialize_db()
        self.collection = self._initialize_collection()

    def _initialize_db(self):
        # 데이터베이스가 없으면 생성
        if self.db_name not in self.client.list_database_names():
            print(f"'{self.db_name}' 데이터베이스가 생성되었습니다.")
        return self.client[self.db_name]

    def _initialize_collection(self):
        # 컬렉션이 없으면 생성
        if self.collection_name not in self.db.list_collection_names():
            print(f"'{self.collection_name}' 컬렉션이 생성되었습니다.")
        
        return self.db[self.collection_name]

    def get_collection(self):
        # 컬렉션을 반환
        return self.collection
    
    def collection_has_data(self):
        # 컬렉션의 문서 수를 세어 데이터 존재 여부 확인
        return self.collection.count_documents({}) > 0

    def close(self):
        # 애플리케이션 종료 시 호출하여 클라이언트 연결 종료
        self.client.close()
        print("MongoDB 클라이언트 연결이 종료되었습니다.")


if __name__ == "__main__":
    # MongoDBClient 인스턴스를 애플리케이션에서 전역적으로 사용
    mongo_client = MongoDBClient()
    collection = mongo_client.get_collection()

    # 필요할 때 `collection`을 사용해 MongoDB 작업 수행
    # 예: collection.insert_one({"name": "sample"})

    # 애플리케이션 종료 시 연결 종료
    # mongo_client.close()
