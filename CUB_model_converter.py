import torch
import torch.nn as nn
import torchvision
from CUB.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY
from CUB.template_model import End2EndModel
import CUB.converter_model as converter_model
import argparse


def get_concept_bottleneck_model(model_1_path, model_2_path, use_sigmoid=False, use_relu=False, n_class_attr=2):
    """
    This function will load and return a concept bottleneck model. If only model_1_path
    is provided it is assumed this is for a joint model. If both model paths are provided
    then model_1_path is assumed to be the XtoC model and model_2_path is for the CtoY model.

    args:
        model_1_path (string): File path to Pytorch pth file
        model_2_path (string): File path to Pytorch pth file

    returns:
        Pytorch model
    """
    if model_1_path is not None and model_2_path is None:  # Model_1_path is a joint model
        return torch.load(model_1_path, map_location=torch.device('cpu'))
    elif model_1_path is not None and model_2_path is not None: # Model_1_path is XtoC model and Model_2_path is CtoY model
        XtoC_model = torch.load(model_1_path, map_location=torch.device('cpu'))
        CtoY_model = torch.load(model_2_path, map_location=torch.device('cpu'))

        return End2EndModel(XtoC_model, CtoY_model, use_relu, use_sigmoid, n_class_attr)
    else:
        raise ValueError("Check the correct model path variables are set")


def get_converted_model(XtoCtoY_model, num_classes, n_attributes, expand_dim, all_fc):
    """
    Initilises a new model and load in weights form a concept bottleneck model.
    The ModuleList layer from the concept bottleneck model is replaced by a linear layer

    Note: This only works if expanded_dim == 0 for the time being

    args:
        XtoCtoY_model (Pytorch model): Pytorch model which weights are loaded from
        num_classes (int): Number of classifications (CtoY model output)
        n_attributes (int): Number of concepts (XtoC model output)
        expand_dim (bool): Size of an additional linear layer in CtoY model (if layer not required, set to 0)
        all_fc (ModuleList): ModuleList layer which is converted to a linear layer

    returns:
        Pytorch model
    """
    pretrained = False
    bottleneck = True
    new_XtoC_model = converter_model.vgg16_bn(pretrained=pretrained, num_classes=num_classes, n_attributes=n_attributes, bottleneck=bottleneck, expand_dim=expand_dim, all_fc=all_fc)
    new_XtoC_model.load_state_dict(XtoCtoY_model.first_model.state_dict(), strict=False)
    XtoCtoY_model.first_model = new_XtoC_model
    return XtoCtoY_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace the ModuleList layer in a concept bottleneck model with a linear layer')

    parser.add_argument(
        '--model_1_path',
        type=str,
        help='model path'
    )
    parser.add_argument(
        '--model_2_path',
        type=str,
        default=None,
        help='model path'
    )
    parser.add_argument(
        '--model_out_path',
        type=str,
        help='output model save path'
    )
    parser.add_argument(
        '--expand_dim',
        type=int,
        default=0,
        help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only'
    )
    parser.add_argument(
        '-n_class_attr',
        type=int,
        default=2,
        help='whether attr prediction is a binary or triary classification'
    )
    parser.add_argument(
        '-use_relu',
        action='store_true',
        help='Whether to include relu activation before using attributes to predict Y. '
             'For end2end & bottleneck model'
    )
    parser.add_argument(
        '-use_sigmoid',
        action='store_true',
        help='Whether to include sigmoid activation before using attributes to predict Y. '
             'For end2end & bottleneck model'
    )
    args = parser.parse_args()

    XtoCtoY_model = get_concept_bottleneck_model(model_1_path=args.model_1_path, model_2_path=args.model_2_path, use_sigmoid=args.use_sigmoid, use_relu=args.use_relu, n_class_attr=args.n_class_attr)
    XtoCtoY_model = get_converted_model(XtoCtoY_model, num_classes=200, n_attributes=len(XtoCtoY_model.first_model.all_fc), expand_dim=args.expand_dim, all_fc=XtoCtoY_model.first_model.all_fc)
    torch.save(XtoCtoY_model.state_dict(), args.model_out_path)
