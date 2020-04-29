import os

def weights_path(_file_, _root_num, dirname):
    basepath = os.path.dirname(_file_)
    backs = [".."]*_root_num
    model_dir = os.path.abspath(os.path.join(basepath, *backs, dirname))
    return model_dir

def check_instance(name, val, ins):
    assert isinstance(val, ins), '{} must be {}'.format(name, ins.__name__)
    return val