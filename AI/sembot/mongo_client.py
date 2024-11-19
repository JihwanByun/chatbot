from pymongo import MongoClient

def singleton(class_):
    class class_w(class_):
      _instance = None
      _sealed = False
      def __new__(class_, *args, **kwargs):

        if class_w._instance is None:
            class_w._instance = super(class_w, class_).__new__(class_)
            class_w._instance._sealed = False
        return class_w._instance
      def __init__(self, *args, **kwargs):
        if self._sealed:
          return
        super(class_w, self).__init__(*args, **kwargs)
        self._sealed = True
    class_w.__name__ = class_.__name__
    return class_w


# MongoDB 클라이언트와 데이터베이스 초기화
@singleton
class MongoDBClient:
    def __init__(self, uri="mongodb://root:ssafy@k11s102.p.ssafy.io:27017/?authSource=admin", db_name="sembot", collection_name="regulations"):
        self.client = MongoClient(uri)
        self.db_name = db_name
        self.collection_name = collection_name

    def find_regulations(self):
        # 모든 규정 데이터를 조회해서 리턴한다.
        return self.client[self.db_name][self.collection_name].find()

    def close(self):
        # 애플리케이션 종료 시 호출하여 클라이언트 연결 종료
        self.client.close()
        print("MongoDB 클라이언트 연결이 종료되었습니다.")