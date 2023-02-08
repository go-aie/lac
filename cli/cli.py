# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import urllib.request

import paddle
from paddlenlp.taskflow.models import BiGruCrf


resource_filenames = dict(
    model_state='model_state.pdparams',
    tag='tag.dic',
    q2b='q2b.dic',
    word='word.dic',
)

url_prefix = 'https://bj.bcebos.com/paddlenlp/taskflow/lexical_analysis/lac'


def download(path):
    if not os.path.exists(path):
        os.makedirs(path)

    for filename in resource_filenames.values():
        local_path = os.path.join(path, filename)
        remote_url = os.path.join(url_prefix, filename)
        if not os.path.exists(local_path):
            print('Downloading {} from {}'.format(filename, remote_url))
            response = urllib.request.urlopen(remote_url)
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(response, f)


def convert_dygraph_to_static(path):
    """Convert the dygraph model to a static model.
    """
    paddle.disable_static()
    model = construct_dygraph(path)
    to_static(model, path)


def construct_dygraph(path):
    emb_dim = 128
    hidden_size = 128
    word_vocab_len = 58224
    tag_vocab_len = 57

    model_instance = BiGruCrf(emb_dim, hidden_size, word_vocab_len, tag_vocab_len)
    state_dict = paddle.load(os.path.join(path, resource_filenames['model_state']))
    model_instance.set_dict(state_dict)

    model = model_instance
    model.eval()
    return model


def to_static(model, path):
    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype='int64', name='token_ids'),
        paddle.static.InputSpec(shape=[None], dtype='int64', name='length'),
    ]
    print('Converting to the inference model costs a little time.')
    static_model = paddle.jit.to_static(model, input_spec=input_spec)

    inference_model_path = os.path.join(path, 'static', 'inference')
    paddle.jit.save(static_model, inference_model_path)
    print('The inference model save in the path: {}'.format(inference_model_path))


def main():
    parser = argparse.ArgumentParser(prog='lac')
    subparsers = parser.add_subparsers(dest='command', title='commands')

    download_parser = subparsers.add_parser('download',
                                        help='Download LAC resources and convert the raw LAC model to an inference model')
    download_parser.add_argument('--path', default='./lac',
                                 help='directory for LAC resource files (default: %(default)s)')

    args = parser.parse_args()
    if args.command == 'download':
        download(args.path)
        convert_dygraph_to_static(args.path)


if __name__ == '__main__':
    main()