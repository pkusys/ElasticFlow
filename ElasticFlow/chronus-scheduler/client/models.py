from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import utils

m_tensors = [[1.1,2.3,2.3,2.3,4.5,9.0,9.0,9.0,9.0,9.0,9.0,9.0,392.0,64.0,15.6],
[1.1,2.3,2.3,4.5,9.0,9.0,9.0,9.0,9.0,392.0,64.0,15.6],
[1.1,2.3,4.5,9.0,9.0,9.0,392.0,64.0,15.6],
[1.2,2.5,5.1,3.4,144.0,64.0,15.6],
[2.0,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,8.0,2.0,9.0,4.0,4.0,9.0,4.0,4.0,9.0,4.0,7.8],
[2.0,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,8.0,2.0,9.0,4.0,4.0,9.0,4.0,4.0,9.0,4.0,7.8],
[2.0,2.3,2.3,2.3,2.3,2.3,2.3,8.0,2.0,9.0,4.0,4.0,9.0,4.0,4.0,9.0,4.0,7.8],
[1.3,5.1,1.5,2.0,1.5,1.1,1.5,1.1,1.3,1.5,1.5,1.1,1.5,1.1,1.3,1.5,1.5,1.1,1.5,1.1,1.3,1.5,1.5,1.1,1.5,1.1,1.3,1.5,1.5,1.1,1.5,1.1,1.3,1.5,1.5,1.1,1.5,1.1,1.3,1.5,1.5,1.1,1.5,1.1,1.3,1.5,1.3,1.8,2.2,3.5,1.5,1.5,2.3,1.1,1.1,2.3,2.0,2.6,1.5,1.5,1.5,1.5,2.3,1.1,1.1,2.3,2.0,2.6,1.5,1.5,1.5,1.5,2.3,1.1,1.1,2.3,2.0,2.6,1.5,1.5,5.9],
[3.8,2.1,1.3,1.6,1.9,1.7,1.7,2.2,5.9,1.7,1.7,2.5,3.0,1.7,1.7,3.5,5.9,1.7,1.7,1.5,7.8],
[3.8,2.1,1.3,1.6,1.9,1.7,1.7,2.2,5.9,1.7,1.7,2.5,3.0,1.7,1.7,3.5,5.9,1.7,1.7,1.5,7.8],
[3.8,2.1,1.3,1.6,1.9,1.7,1.7,2.2,5.9,1.7,1.7,2.5,3.0,1.7,1.7,3.5,5.9,1.7,1.7,1.5,7.8]]



m_names = ['vgg16', 'resnet50', 'inception3', 'bert', 'gpt2', 'deepspeech2']
# m_mem = [0.60, 0.55, 0.45, 0.13, 0.85, 0.70, 0.50, 0.85, 0.80]
m_mem = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

worker_mem = 5
ps_mem = 8
per_worker_mem = 0.2


def get_model(model_name):
    '''
    get model tensor information by model_name
    return a dict{name, tensors(list)}
    '''
    if model_name == 'vgg16':
        m_idx = 0
    elif model_name == 'resnet50':
        m_idx = 1
    elif model_name == 'inception3':
        m_idx = 2
    elif model_name == 'bert':
        m_idx = 3
    elif model_name == 'gpt2':
        m_idx = 4
    elif model_name == 'deepspeech2':
        m_idx = 5
    else:
        # m_idx = random.randint(0,5)
        m_idx = 1
        #utils.print_fn('No model match, pick %s' % m_names[m_idx])

    ret = {'name':m_names[m_idx], 'ind':m_idx, 'tensors':m_tensors[m_idx], 'mem_util':m_mem[m_idx]}
    return ret

def get_model_with_scale(model_name, model_scale):
    '''
    get model tensor information by model_name
    and extend the number of tensors with model_scale
    return a dict{name, tensors(list)}
    '''
    ret = get_model(model_name)
    ret['tensors'] = ret['tensors'] * int(model_scale)
    total_size = 0.0
    for i in ret['tensors']:
        total_size += i 
    ret['total_size'] = round(total_size, 1) #float x.x
    return ret

def get_max_gpu(job_dict):
    if job_dict['model_name'] == 'deepspeech2':
        return job_dict['batch_size']
    if job_dict['model_name'] == 'gpt2':
        if job_dict['batch_size'] == 128:
            return 16
        if job_dict['batch_size'] == 256:
            return 32
    if job_dict['model_name'] == 'bert':
        if job_dict['batch_size'] == 64:
            return 8
        return 16
    if job_dict['model_name'] == 'inception3':
        if job_dict['batch_size'] == 64:
            return 16
        if job_dict['batch_size'] == 128:
            return 32
    if job_dict['model_name'] == 'resnet50':
        if job_dict['batch_size'] == 64:
            return 32
        return min(job_dict['batch_size'], 64)
    if job_dict['model_name'] == 'vgg16':
        if job_dict['batch_size'] == 64:
            return 32
    return min(job_dict['batch_size'], 64)


# if __name__ == '__main__':
#     # print('Hello world %d' % 2)
#     print(get_model_with_scale('vgg11', 2))