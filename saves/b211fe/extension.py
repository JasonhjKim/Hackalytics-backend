

def transforms():
    transform = T.Compose([T.Resize((256,256)),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
                        ])
    
    return transform

def get_model():
        model = torchvision.models.vgg19(pretrained=True)
        # add Linear classifier layer
        in_features = model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    return model
