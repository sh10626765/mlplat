from django.db import models

# Create your models here.
import pymongo
import json


def SavaData(dataName, dictList, overwrite, host, port, database):
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)

    n = 0
    if dataName in db.list_collection_names():
        if not overwrite:
            n += 1
            dataName = dataName[:dataName.rfind('.')] + str(n) + dataName[dataName.rfind('.'):]
            while dataName in db.list_collection_names():
                n += 1
                dataName = dataName[:dataName.rfind('.') - 1] + str(n) + dataName[dataName.rfind('.'):]
        else:
            db.drop_collection(dataName)

    coll = db.create_collection(name=dataName)

    return dataName, coll.insert_many(dictList)


def UpdateData(number, newData, dataName, host, port, database):
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)

    coll = db.get_collection(dataName)
    return coll.find_one_and_update({'NO': number}, {'$set': newData})


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


def RmDoc(doc_filter, collection, host, port, database):
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)
    coll = db.get_collection(collection)
    return coll.remove(doc_filter)


class Data(models.Model):
    data_name = models.CharField(max_length=200, primary_key=True)
    data_abstract = models.CharField(max_length=200, blank=True, null=True)
    data_keywords = models.CharField(max_length=200, blank=True, null=True)
    data_field = models.CharField(max_length=200, default='')
    data_interest = models.CharField(max_length=200, default='')
    sample_num = models.IntegerField(default=0)
    dim_num = models.IntegerField(default=0)

    submitter = models.CharField(max_length=200, default='')
    collater = models.CharField(max_length=200, default='')
    submitter_organization = models.CharField(max_length=200, blank=True, null=True)
    submitter_email = models.EmailField(default='')
    submitter_phone = models.CharField(max_length=200, blank=True, null=True)
    submitter_address = models.CharField(max_length=200, blank=True, null=True)

    origin = models.CharField(max_length=200, default='')
    origin_type = models.CharField(max_length=200, default='')
    origin_decision = models.CharField(max_length=200, blank=True, null=True)
    origin_platenumber = models.CharField(max_length=200, blank=True, null=True)
    origin_materialname = models.CharField(max_length=200, blank=True, null=True)
    origin_expcondition = models.CharField(max_length=200, blank=True, null=True)
    origin_exparguments = models.CharField(max_length=200, blank=True, null=True)
    origin_expdevice = models.CharField(max_length=200, blank=True, null=True)

    pub_date = models.DateTimeField('date published')


class MachineLearningMethods(models.Model):
    method_name = models.CharField(max_length=200, primary_key=True)
