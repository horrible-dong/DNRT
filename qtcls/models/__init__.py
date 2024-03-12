# Copyright (c) QIU Tian. All rights reserved.

from .vit_example import *

__vars__ = vars()


def build_model(args):
    from termcolor import cprint
    from .. import datasets

    model_lib = args.model_lib.lower()
    model_name = args.model.lower()

    if 'num_classes' in args.model_kwargs.keys():
        cprint(f"Warning: Do NOT set 'num_classes' in 'args.model_kwargs'. "
               f"Now fetching the 'num_classes' registered in 'qtcls/datasets/__init__.py'.", 'light_yellow')

    try:
        num_classes = datasets.num_classes[args.dataset.lower()]
    except KeyError:
        print(f"KeyError: 'num_classes' for the dataset '{args.dataset.lower()}' is not found. "
              f"Please register your dataset's 'num_classes' in 'qtcls/datasets/__init__.py'.")
        exit(1)

    args.model_kwargs['num_classes'] = num_classes

    if model_lib == 'default':
        try:
            model = __vars__[model_name](**args.model_kwargs)
        except KeyError:
            print(f"KeyError: Model '{model_name}' is not found.")
            exit(1)

        return model

    if model_lib == 'timm':
        ...

    raise ValueError(f"Model lib '{model_lib}' is not found.")
