import torch
import torch.nn as nn
import torchvision
from CUB.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY
import CUB.converter_model as converter_model
import argparse


def get_concept_bottleneck_model(model_path):
    """
    Load and return a standard concept bottleneck model from a given file path.

    args:
        model_path (string): File path to Pytorch pth file

    returns:
        Pytorch model
    """
    return torch.load(model_path, map_location=torch.device('cpu'))


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
    args = parser.parse_args()

    XtoCtoY_model = get_concept_bottleneck_model(args.model_1_path)
    XtoCtoY_model = get_converted_model(XtoCtoY_model, num_classes=200, n_attributes=len(XtoCtoY_model.first_model.all_fc), expand_dim=args.expand_dim, all_fc=XtoCtoY_model.first_model.all_fc)
    torch.save(XtoCtoY_model.state_dict(), args.model_out_path)
