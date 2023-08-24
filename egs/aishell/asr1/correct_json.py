import json
import os,sys

json_path = '/work/m11115119/espnet/egs/aishell/asr1/dump/train/deltafalse/data_bert.json'
json_path1 = '/work/m11115119/espnet/egs/aishell/asr1/dump/train/deltafalse/data_bert_tmp.json'


with open(json_path,'r',encoding='utf-8') as f:
#定义为只读模型，并定义名称为f
    params = json.load(f)
    #修改内容
    print(type(params))
    # print(params.get('feat',default=None))
    for utt_key, utt_value in params.items():
        for ted_key, ted_value in utt_value.items():
            for input_key, input_value in ted_value.items():
                if input_key == "input":
                    for index, i in enumerate(input_value):
                        for a, b in i.items():
                            if a == 'feat':
                                # print(b)
                                tmp = params[utt_key][ted_key][input_key][index][a]
                                print(params[utt_key][ted_key][input_key][index][a])
                                print(tmp)
                                tmp2 = tmp.replace('/data/espnet/egs/aishell/asr1/dump', '/tmp/dump')
                                print(tmp2)
                                params[utt_key][ted_key][input_key][index][a]=tmp2
                                print(params[utt_key][ted_key][input_key][index][a])
            
    # print(params.values())
    #打印
    dict = params
    #将修改后的内容保存在dict中
f.close()
#关闭json读模式

with open(json_path1,'w',encoding='utf-8') as r:
    #定义为写模式，名称定义为r
    
        json.dump(dict,r,ensure_ascii=False,indent=4)
        #将dict写入名称为r的文件中
        
r.close()
