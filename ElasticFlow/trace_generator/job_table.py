from job_template import JobTemplate

def deepspeech2(batch_size):
    model = 'None (batch size %d)' % (batch_size)
    # TODO
    command = 'python3 main.py -j 8 -a resnet50 -b %d' % (batch_size)
    command += ' %s/imagenet/'
    working_directory = 'image_classification/imagenet'
    num_steps_arg = '--num_minibatches'
    return JobTemplate(model=model, command=command, batch_size=batch_size,
                       working_directory=working_directory, name='deepspeech2',
                       num_steps_arg=num_steps_arg, distributed=True)

def vgg16(batch_size):
    model = 'VGG-16 (batch size %d)' % (batch_size)
    # TODO
    command = 'python3 main.py -j 8 -a resnet50 -b %d' % (batch_size)
    command += ' %s/imagenet/'
    working_directory = 'image_classification/imagenet'
    num_steps_arg = '--num_minibatches'
    return JobTemplate(model=model, command=command, batch_size=batch_size,
                       working_directory=working_directory, name='vgg16',
                       num_steps_arg=num_steps_arg, distributed=True)


def resnet50(batch_size):
    model = 'ResNet-50 (batch size %d)' % (batch_size)
    command = 'python3 main.py -j 8 -a resnet50 -b %d' % (batch_size)
    command += ' %s/imagenet/'
    working_directory = 'image_classification/imagenet'
    num_steps_arg = '--num_minibatches'
    return JobTemplate(model=model, command=command, batch_size=batch_size,
                       working_directory=working_directory, name='resnet50',
                       num_steps_arg=num_steps_arg, distributed=True)

def inception3(batch_size):
    model = 'None (batch size %d)' % (batch_size)
    command = 'python3 main.py -j 8 -a resnet50 -b %d' % (batch_size)
    command += ' %s/imagenet/'
    working_directory = 'image_classification/imagenet'
    num_steps_arg = '--num_minibatches'
    return JobTemplate(model=model, command=command, batch_size=batch_size,
                       working_directory=working_directory, name='inception3',
                       num_steps_arg=num_steps_arg, distributed=True)

def gpt2(batch_size):
    model = 'GPT-2 (batch size %d)' % (batch_size)
    command = 'python3 main.py -j 8 -a resnet50 -b %d' % (batch_size)
    command += ' %s/imagenet/'
    working_directory = 'image_classification/imagenet'
    num_steps_arg = '--num_minibatches'
    return JobTemplate(model=model, command=command, batch_size=batch_size,
                       working_directory=working_directory, name='gpt2',
                       num_steps_arg=num_steps_arg, distributed=True)

def bert(batch_size):
    model = 'GPT-2 (batch size %d)' % (batch_size)
    command = 'python3 main.py -j 8 -a resnet50 -b %d' % (batch_size)
    command += ' %s/imagenet/'
    working_directory = 'image_classification/imagenet'
    num_steps_arg = '--num_minibatches'
    return JobTemplate(model=model, command=command, batch_size=batch_size,
                       working_directory=working_directory, name='bert',
                       num_steps_arg=num_steps_arg, distributed=True)


JobTable = []


for batch_size in [64, 128, 256]:
    JobTable.append(resnet50(batch_size))
for batch_size in [64, 128, 256]:
    JobTable.append(vgg16(batch_size))
for batch_size in [64, 128]:
    JobTable.append(inception3(batch_size))
JobTable.append(gpt2(128))
JobTable.append(gpt2(256))
JobTable.append(bert(128))
JobTable.append(bert(64))
JobTable.append(deepspeech2(64))
JobTable.append(deepspeech2(32))