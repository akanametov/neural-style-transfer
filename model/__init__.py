import torch
import torch.nn as nn
from .utils import NormalizationLayer, ContentLossLayer, StyleLossLayer

class Compiler():
    def __init__(self, baseModel, contentLayerNames, styleLayerNames, device='cuda:0'):
        self.baseModel = baseModel.to(device)
        self.contentLayerNames = contentLayerNames
        self.styleLayerNames = styleLayerNames
        
    
    def compile(self, contentImage, styleImage, device='cuda:0'):
        contentImage = contentImage.to(device)
        styleImage = styleImage.to(device)
        contentLayers=[]
        styleLayers=[]
        model = nn.Sequential()
        model.add_module('norm', NormalizationLayer())
        i = 0
        for layer in self.baseModel.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
                
            model.add_module(name, layer)
            
            if name in self.contentLayerNames:
                target = model(contentImage).detach()
                layer = ContentLossLayer(target)
                model.add_module("content{}".format(i), layer)
                contentLayers.append(layer)

            if name in self.styleLayerNames:
                target = model(styleImage).detach()
                layer = StyleLossLayer(target)
                model.add_module("style{}".format(i), layer)
                styleLayers.append(layer)
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLossLayer) or isinstance(model[i], StyleLossLayer):
                break
        model = model[:(i + 1)]
        return model, contentLayers, styleLayers
