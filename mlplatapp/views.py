from django.shortcuts import render, render_to_response, HttpResponse, HttpResponseRedirect, redirect, reverse
from django.http import FileResponse, StreamingHttpResponse
from django.utils import timezone
from . import utils, models

# Create your views here.
import pandas as pd
import os
import json
import pymongo
from .feature_select import BPSO_FK, GA_FK

HOST = 'localhost'
PORT = 27017
DATABASE = 'materialsData'
si = 0  # 暂时，用于记录第一个数值型数据的列下标


def test(request):
    return render(request, 'testhtml.html')


def edit(request, name, number):
    if request.method == 'POST':
        val_list = request.POST.getlist('material_attr')  # 获取form表单数据皆为字符串，须对数字进行转换
        eval_val_list = [eval(item) if utils.is_number(item) else item for item in val_list]

        data = models.ReadData(name, HOST, PORT, DATABASE)
        key_list = []
        for k, v in data[0].items():
            key_list.append(k)
        newdata = dict(zip(key_list, eval_val_list))

        update_result = models.UpdateData({'NO': number}, {'$set': newdata}, name, HOST, PORT, DATABASE)

        return redirect(reverse('qualitycontrol', kwargs={'data_name': name}))

    if request.method == 'GET':
        # 根据数据集的名称name，编号number，查询到需要改动的数据
        data = models.ReadData(name, HOST, PORT, DATABASE)

        item = None
        for it in data:
            if it['NO'] == number:
                item = it
                break

        return render(request, 'modify_data.html', {'dataitem': item, 'dataname': name, 'col_start': si})


def setverbose(request, name, number):
    if request.method == 'GET':
        data = models.ReadData(name, host=HOST, port=PORT, database=DATABASE)

        excelproc = utils.excelProcessor(data)
        corr = excelproc.get_pearson_r()
        attr_name = excelproc.col_name[:-1]
        corr = [(attr_name[i[0]], attr_name[i[1]], i[2]) for i in corr]  # 将属性下标转为属性名
        corr.sort(key=lambda x: x[2], reverse=True)

        return render(request, 'set_verbose.html', {
            'dataname': name,
            'attrpair': corr[number]
        })

    if request.method == 'POST':
        ck_attr1 = request.POST.get('attr1')
        ck_attr2 = request.POST.get('attr2')

        cl = pymongo.MongoClient(host=HOST, port=PORT)
        db = cl.get_database(DATABASE)
        coll = db.get_collection('verbose_attr_' + name)

        if ck_attr1:
            coll.update_one({'name': ck_attr1}, {'$setOnInsert': {'name': ck_attr1}}, upsert=True)
        if ck_attr2:
            coll.update_one({'name': ck_attr2}, {'$setOnInsert': {'name': ck_attr2}}, upsert=True)

        return redirect(reverse('qualitycontrol', kwargs={'data_name': name}))


def index(request):
    return render(request, 'homepage.html')


def show(request):
    if request.method == 'POST':
        pass

    if request.method == 'GET':
        # Data objects 有models中定义的属性
        model_info = models.Data.objects.all()  # return type: QuerySet of Data object
        # materialsData 中存在的集合
        data_info = models.ReadColl(host=HOST, port=PORT, database=DATABASE)

        # 判断materialsData中的集合信息是否与Data表中信息一致
        # 一致则保留，否则视作冗余信息，删除
        for model_item in model_info:
            if model_item.data_name in data_info:
                continue
            else:
                models.Data.objects.filter(data_name=model_item.data_name, pub_date=model_item.pub_date).delete()

        for data_item in data_info:
            if models.Data.objects.filter(data_name=data_item):
                continue
            else:
                models.DropColl(data_item, host=HOST, port=PORT, database=DATABASE)
                models.DropColl('stat_data_quality_' + data_item, host=HOST, port=PORT,
                                database=DATABASE)  # 质量检测过程的中间结果，使用过就没用了，删除
                models.DropColl('algo_data_quality_' + data_item, host=HOST, port=PORT, database=DATABASE)
                models.DropColl('eudist_data_quality_' + data_item, host=HOST, port=PORT, database=DATABASE)

        return render(request, 'show_data.html', {'dataset': models.Data.objects.all()})


def data(request, name):
    if request.method == 'GET':
        data = models.ReadData(name, host=HOST, port=PORT, database=DATABASE)
        # return HttpResponse(data)
        return render(request, 'data_detail.html', {'data': data})


def upload(request):
    global si
    if request.method == 'POST':
        fileinput = request.FILES.get('input-excel')  # read file from <input name="input-excel">
        fileinputname = fileinput.name  # get file name
        excelproc = utils.excelProcessor(fileinput)  # preprocess the file uploaded
        si = excelproc.col_start

        if excelproc.has_blank_cell():  # if any blank cell exists, interrupt the uploading
            return HttpResponse('Blank Cells Exist!')

        # 将表格数据存入数据库，返回表格数据在数据库中的集合名和文档_id
        # 数据库文档不允许字段名中存在'.','$'字符，须过滤
        data_name_in_db, res = models.SavaData(fileinputname, excelproc.get_data(), False,
                                               host=HOST, port=PORT, database=DATABASE)

        # 数据质量检测结果存入数据库
        # 数据质量检测结果仅作为中间数据存储，数据库不保留这些信息
        modeldata = models.Data(data_name=data_name_in_db, pub_date=timezone.now())
        modeldata.save()

        return redirect(reverse('qualitycontrol', kwargs={
            'data_name': data_name_in_db,
            # 'stat_quality_name': stat_data_quality_name,
            # 'algo_quality_name': algo_data_quality_name,
            # 'eudist_quality_name': eudist_data_quality_name,
        }))

    if request.method == 'GET':
        return render(request, 'upload_data.html')


def qualitycontrol(request, data_name):  # , stat_quality_name, algo_quality_name, eudist_quality_name
    if request.method == 'POST':
        pass

    if request.method == 'GET':
        data = models.ReadData(data_name, host=HOST, port=PORT, database=DATABASE)

        excelproc = utils.excelProcessor(data)
        corr = excelproc.get_pearson_r()
        attr_name = excelproc.col_name[:-1]
        corr = [(attr_name[i[0]], attr_name[i[1]], i[2]) for i in corr]  # 将属性下标转为属性名
        corr.sort(key=lambda x: x[2], reverse=True)

        stat_data_quality = []  # 对表格数据进行质量检测，得到基本统计信息
        for key, val in excelproc.statistics_data_check().to_dict().items():
            stat_data_quality.append({key.replace('.', '').replace('$', ''): val})

        stat_data_quality_name, res = models.SavaData('stat_data_quality_' + data_name, stat_data_quality, True,
                                                      host=HOST, port=PORT, database=DATABASE)

        algo_data_quality = []  # 检测表格数据质量，得到算法检测结果
        for key, val in excelproc.algorithm_data_check().items():
            algo_data_quality.append({key: val})
        algo_data_quality_name, res = models.SavaData('algo_data_quality_' + data_name, algo_data_quality, True,
                                                      host=HOST, port=PORT, database=DATABASE)

        eudist_data_quality = []  # 检测表格数据质量，得到算法检测结果
        for key, val in excelproc.eudist_data_check().items():
            eudist_data_quality.append({'NO': key, 'count': val})
        eudist_data_quality_name, res = models.SavaData('eudist_data_quality_' + data_name, eudist_data_quality, True,
                                                        host=HOST, port=PORT, database=DATABASE)

        stat = models.ReadData(stat_data_quality_name, host=HOST, port=PORT, database=DATABASE)
        algo = models.ReadData(algo_data_quality_name, host=HOST, port=PORT, database=DATABASE)
        eudist = models.ReadData(eudist_data_quality_name, host=HOST, port=PORT, database=DATABASE)

        stat_dict = {}
        for i in stat:
            stat_dict.update(i)
        algo_dict = {}
        for i in algo:
            algo_dict.update(i)
        return render(request, 'quality_control.html', {
            'dataname': data_name,
            'data': data,
            'stat': json.dumps(stat_dict),
            'algo': json.dumps(algo_dict),
            'eudist': json.dumps(eudist),
            'pearsonr': corr,
            'col_start': si,
        })


def download(request, name):
    if request.method == 'GET':
        data = models.ReadData(name, host=HOST, port=PORT, database=DATABASE)
        dataj = pd.read_json(json.dumps(data))
        WFP = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'excel_file'),
                           str(name))
        # WFP = '././excel_file/' + str(name)
        dataj.to_excel(WFP, index=False)
        # res = FileResponse(file)
        # res['Content-Type'] = 'application/octet-stream'
        # res['Content-Disposition'] = 'attachment;filename="' + str(name) + '"'
        return FileResponse(open(WFP, 'rb'))


def dataprocess(request, data_name):
    if request.method == 'POST':
        pass
    if request.method == 'GET':
        return HttpResponse('this is data pre-process page for {}'.format(data_name))


def featureselection(request, data_name):
    if request.method == 'POST':
        check_list = request.POST.getlist('checkbox_list')
        features_to_retain = [{'name': feature_name} for feature_name in check_list]
        ftr_name, res = models.SavaData('custom_retain_features_' + data_name, features_to_retain, True, HOST, PORT,
                                        DATABASE)
        verbose_feature = models.ReadData('verbose_attr_' + data_name, HOST, PORT, DATABASE)

        origin_data = models.ReadData(data_name, HOST, PORT, DATABASE)
        origin_data = utils.excelProcessor(origin_data)

        features_to_retain_idx = [origin_data.col_name[:-1].index(e['name']) for e in features_to_retain]
        verbose_feature_idx = [origin_data.col_name[:-1].index(e['name']) for e in verbose_feature]
        # for e in features_to_retain:
        #     features_to_retain_idx.append(origin_data.col_name[:-1].index(e['name']))
        print(features_to_retain_idx, verbose_feature_idx)

        index_of_sets = [[1, 3, 6, 14, 24, 28, 29, 38, 43, 48],
                         [0, 5, 8, 15, 16, 25, 30, 31, 39, 44],
                         [2, 10, 11, 17, 18, 26, 32, 33, 40, 45],
                         [9, 12, 13, 19, 20, 27, 34, 35, 41, 46],
                         [4, 7, 21, 22, 23, 36, 37, 42, 47, 49]]

        bpso_fk = BPSO_FK.BPSO_FK(origin_data, 100, 0.01, 40, features_to_retain_idx, verbose_feature_idx, index_of_sets)
        res, rmse, r2 = bpso_fk.evolve()

        print(check_list)
        print(res, rmse, r2)
        return HttpResponse('Success! You have chosen {}.<br>Result is {}.<br>RMSE is {}.<br>R2 is {}'.format(check_list, res, rmse, r2))
    if request.method == 'GET':
        data = models.ReadData(data_name, HOST, PORT, DATABASE)  # 读原始数据
        excelproc = utils.excelProcessor(data)

        verbose_attrs = models.ReadData('verbose_attr_' + data_name, HOST, PORT, DATABASE)  # 读标记的冗余属性
        features_to_select = excelproc.col_name[:-1]  # 全部属性
        for verbose_attr in verbose_attrs:  # 从筛选冗余属性后的属性中，选择自定义保留属性
            features_to_select.remove(verbose_attr['name'])

        return render(request, 'custom_retain_features.html',
                      {
                          'dataname': data_name,
                          'featurenames': features_to_select
                      })


def machinelearning(request, data_name):
    if request.method == 'POST':
        check_list = request.POST.getlist('checkbox_list')
        print(check_list)
        return HttpResponse('Success! You have chosen {}'.format(check_list))
    if request.method == 'GET':
        ml_methods = models.MachineLearningMethods.objects.all()
        return render(request, 'machine_learning.html',
                      {
                          'dataname': data_name,
                          'ml_methods': ml_methods,
                      })
