from django.db import models

# Create your models here.
import pymongo
import json


def SavaData(dataName, dictList, host, port, database):

    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)

    n = 0
    if dataName in db.list_collection_names():
        n += 1
        dataName = dataName[:dataName.rfind('.')] + str(n) + dataName[dataName.rfind('.'):]
        while dataName in db.list_collection_names():
            n += 1
            dataName = dataName[:dataName.rfind('.') - 1] + str(n) + dataName[dataName.rfind('.'):]

    coll = db.create_collection(name=dataName)

    return dataName, coll.insert_many(dictList)

def ReadData(dataName, host, port, database):
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)

    coll = db.get_collection(name=dataName)
    doc_list = [it for it in coll.find(projection={"_id": False})]
    # doc_json = json.dumps(doc_list)
    return doc_list

def ReadColl(host, port, database):
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)
    return db.list_collection_names()

def DropColl(collection, host, port, database):
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)
    return db.drop_collection(collection)

class Data(models.Model):
    data_name = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
