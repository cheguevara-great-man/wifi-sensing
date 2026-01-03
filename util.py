from dataset import *
from UT_HAR_model import *
from NTU_Fi_model import *
from widar_model import *
from self_supervised_model import *
import torch
import os
from Widar_digit_model import Widar_digit_amp_model, Widar_digit_conj_model

def load_data_n_model(dataset_name, model_name, root,sample_rate=1.0,sample_method='uniform_nearest',interpolation_method='linear',use_energy_input = 1,use_mask_0 = 0):
    classes = {'UT_HAR_data':7,'NTU-Fi-HumanID':14,'NTU-Fi_HAR':6,'Widar':22,'Widar_digit_amp': 10,
'Widar_digit_conj': 10,}
    if dataset_name == 'UT_HAR_data':
        print('using dataset: UT-HAR DATA')
        data = UT_HAR_dataset(root, sample_rate=sample_rate, sample_method=sample_method,interpolation_method=interpolation_method, use_energy_input=use_energy_input, use_mask_0=use_mask_0)
        train_set = torch.utils.data.TensorDataset(data['X_train'],data['y_train'])
        test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'],data['X_test']),0),torch.cat((data['y_val'],data['y_test']),0))
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=256,shuffle=True, drop_last=True) # drop_last=True
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=512,shuffle=False)
        if model_name == 'MLP':
            print("using model: MLP")
            model = UT_HAR_MLP()
            train_epoch = 200
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = UT_HAR_LeNet()
            train_epoch = 200 #40
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = UT_HAR_ResNet18()
            train_epoch = 200 #70
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = UT_HAR_ResNet50()
            train_epoch = 200 #100
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = UT_HAR_ResNet101()
            train_epoch = 200 #100
        elif model_name == 'RNN':
            print("using model: RNN")
            model = UT_HAR_RNN()
            train_epoch = 3000
        elif model_name == 'GRU':
            print("using model: GRU")
            model = UT_HAR_GRU()
            train_epoch = 200
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = UT_HAR_LSTM()
            train_epoch = 200
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = UT_HAR_BiLSTM()
            train_epoch = 200
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = UT_HAR_CNN_GRU()
            train_epoch = 200 #20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = UT_HAR_ViT()
            train_epoch = 200 #100
        return train_loader, test_loader, model, train_epoch

    
    elif dataset_name == 'NTU-Fi-HumanID':
        print('using dataset: NTU-Fi-HumanID')
        num_classes = classes['NTU-Fi-HumanID']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/train_amp/',sample_rate=sample_rate,sample_method=sample_method,interpolation_method=interpolation_method ,use_energy_input = use_energy_input,use_mask_0 = use_mask_0), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/test_amp/',sample_rate=sample_rate,sample_method=sample_method,interpolation_method=interpolation_method ,use_energy_input = use_energy_input,use_mask_0 = use_mask_0), batch_size=64, shuffle=False)
        if model_name == 'MLP':
            print("using model: MLP")
            model = NTU_Fi_MLP(num_classes)
            train_epoch = 50 #15
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = NTU_Fi_LeNet(num_classes)
            train_epoch = 50 #20
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = NTU_Fi_ResNet18(num_classes)
            train_epoch = 50 #30
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = NTU_Fi_ResNet50(num_classes)
            train_epoch = 50 #40
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = NTU_Fi_ResNet101(num_classes)
            train_epoch = 50
        elif model_name == 'RNN':
            print("using model: RNN")
            model = NTU_Fi_RNN(num_classes)
            train_epoch = 75
        elif model_name == 'GRU':
            print("using model: GRU")
            model = NTU_Fi_GRU(num_classes)
            train_epoch = 50 #40
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = NTU_Fi_LSTM(num_classes)
            train_epoch = 50
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = NTU_Fi_BiLSTM(num_classes)
            train_epoch = 50
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = NTU_Fi_CNN_GRU(num_classes)
            train_epoch = 200 #20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = NTU_Fi_ViT(num_classes=num_classes)
            train_epoch = 50
        return train_loader, test_loader, model, train_epoch
    
    
    elif dataset_name == 'NTU-Fi_HAR':
        print('using dataset: NTU-Fi_HAR')
        num_classes = classes['NTU-Fi_HAR']
        #train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/train_amp/'), batch_size=64, shuffle=True)
        NUM_WORKERS = 2  # 你可以尝试 4, 8, 16 等，#为了设置进程为单线程，减少cpu占用
        train_loader = torch.utils.data.DataLoader(
            dataset=CSI_Dataset(root + 'NTU-Fi_HAR/train_amp/',sample_rate=sample_rate ,sample_method=sample_method,interpolation_method=interpolation_method ,use_energy_input = use_energy_input,use_mask_0 = use_mask_0),  # 别忘了使用你修正后的数据路径！
            batch_size=64,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,  # 如果你的数据可以完全放入内存，这个选项可以加速到GPU的传输
            #persistent_workers=True  # 如果PyTorch版本>=1.8，建议开启，可以避免每个epoch都重建worker进程
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=CSI_Dataset(root + 'NTU-Fi_HAR/test_amp/',sample_rate=sample_rate,sample_method=sample_method,interpolation_method=interpolation_method ,use_energy_input = use_energy_input,use_mask_0 = use_mask_0),  # 别忘了使用你修正后的数据路径！
            batch_size=64,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            #persistent_workers=True
        )
        #test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/test_amp/'), batch_size=64, shuffle=False)
        if model_name == 'MLP':
            print("using model: MLP")
            model = NTU_Fi_MLP(num_classes)
            train_epoch = 30 #10
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = NTU_Fi_LeNet(num_classes)
            train_epoch = 30 #10
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = NTU_Fi_ResNet18(num_classes)
            train_epoch = 30
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = NTU_Fi_ResNet50(num_classes)
            train_epoch = 30 #40
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = NTU_Fi_ResNet101(num_classes)
            train_epoch = 30
        elif model_name == 'RNN':
            print("using model: RNN")
            model = NTU_Fi_RNN(num_classes)
            train_epoch = 70
        elif model_name == 'GRU':
            print("using model: GRU")
            model = NTU_Fi_GRU(num_classes)
            train_epoch = 30 #20
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = NTU_Fi_LSTM(num_classes)
            train_epoch = 30 #20
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = NTU_Fi_BiLSTM(num_classes)
            train_epoch = 30 #20
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = NTU_Fi_CNN_GRU(num_classes)
            train_epoch = 100 #20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = NTU_Fi_ViT(num_classes=num_classes)
            train_epoch = 30
        return train_loader, test_loader, model, train_epoch

    elif dataset_name == 'Widar_digit_amp':
        # Expect directory: <root>/Widar_digit/{amp,conj,meta}/...
        #ds_root = os.path.join(root, 'Widar_digit')
        #train_set = Widar_digit_amp_dataset(ds_root, split='train')
        train_set = Widar_digit_amp_dataset(
            root_dir=root,
            split="train",
            sample_rate=sample_rate,
            sample_method=sample_method,
            interpolation_method=interpolation_method,
            use_mask_0=use_mask_0,
        )
        test_set = Widar_digit_amp_dataset(
            root_dir=root,
            split="test",
            sample_rate=sample_rate,
            sample_method=sample_method,
            interpolation_method=interpolation_method,
            use_mask_0=use_mask_0,
        )
        train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=False, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader  = DataLoader(dataset=test_set,  batch_size=128, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

        # infer T/F from the first sample
        x0, _ = train_set[0]
        T, F = int(x0.shape[-2]), int(x0.shape[-1])

        train_epoch = 200

        model = Widar_digit_amp_model(model_name, num_classes=10, T=T)

        return train_loader,test_loader,model,train_epoch

    elif dataset_name == 'Widar_digit_conj':
        train_set = Widar_digit_conj_dataset(
            root_dir=root,
            split="train",
            sample_rate=sample_rate,
            sample_method=sample_method,
            interpolation_method=interpolation_method,
            use_mask_0=use_mask_0,
        )
        test_set = Widar_digit_conj_dataset(
            root_dir=root,
            split="test",
            sample_rate=sample_rate,
            sample_method=sample_method,
            interpolation_method=interpolation_method,
            use_mask_0=use_mask_0,
        )
        train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=False, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader  = DataLoader(dataset=test_set,  batch_size=128, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

        x0, _ = train_set[0]
        T, F = int(x0.shape[-2]), int(x0.shape[-1])

        train_epoch = 200

        model = Widar_digit_conj_model(model_name, num_classes=10, T=T)

        return train_loader,test_loader,model,train_epoch

    elif dataset_name == 'Widar':
        print('using dataset: Widar')
        num_classes = classes['Widar']
        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/train/'), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/test/'), batch_size=128, shuffle=False)
        if model_name == 'MLP':
            print("using model: MLP")
            model = Widar_MLP(num_classes)
            train_epoch = 30 #20
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = Widar_LeNet(num_classes)
            train_epoch = 100 #40
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = Widar_ResNet18(num_classes)
            train_epoch = 100
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = Widar_ResNet50(num_classes)
            train_epoch = 100 #40
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = Widar_ResNet101(num_classes)
            train_epoch = 100
        elif model_name == 'RNN':
            print("using model: RNN")
            model = Widar_RNN(num_classes)
            train_epoch = 500
        elif model_name == 'GRU':
            print("using model: GRU")
            model = Widar_GRU(num_classes)
            train_epoch = 200 
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = Widar_LSTM(num_classes)
            train_epoch = 200 #20
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = Widar_BiLSTM(num_classes)
            train_epoch = 200
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = Widar_CNN_GRU(num_classes)
            train_epoch = 200 #20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = Widar_ViT(num_classes=num_classes)
            train_epoch = 200
        return train_loader, test_loader, model, train_epoch


def load_unsupervised_data_n_model(model_name,root):
    HAR_train_dataset=CSI_Dataset(root+'NTU-Fi_HAR/train_amp/')
    HAR_test_dataset=CSI_Dataset(root+'NTU-Fi_HAR/test_amp/')
    unsupervised_train_dataset = torch.utils.data.ConcatDataset([HAR_train_dataset,HAR_test_dataset])
    unsupervised_train_loader = torch.utils.data.DataLoader(dataset=unsupervised_train_dataset, batch_size=64, shuffle=True)
    supervised_train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root+'NTU-Fi-HumanID/test_amp/'), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root+'NTU-Fi-HumanID/train_amp/'), batch_size=64, shuffle=False)
    if model_name == 'MLP':
        print("using model: MLP_Parrallel")
        model = MLP_Parrallel()
    elif model_name == 'LeNet':
        print("using model: CNN_Parrallel")
        model = CNN_Parrallel()
    elif model_name == 'ResNet18':
        print("using model: ResNet18_Parrallel")
        model = ResNet18_Parrallel()
    elif model_name == 'ResNet50':
        print("using model: ResNet50_Parralle")
        model = ResNet50_Parrallel()
    elif model_name == 'ResNet101':
        print("using model: ResNet101_Parrallel")
        model = ResNet101_Parrallel()
    elif model_name == 'RNN':
        print("using model: RNN_Parrallel")
        model = RNN_Parrallel()
    elif model_name == 'GRU':
        print("using model: GRU_Parrallel")
        model = GRU_Parrallel()
    elif model_name == 'LSTM':
        print("using model: LSTM_Parrallel")
        model = LSTM_Parrallel()
    elif model_name == 'BiLSTM':
        print("using model: BiLSTM_Parrallel")
        model = BiLSTM_Parrallel()
    elif model_name == 'CNN+GRU':
        print("using model: CNN_GRU_Parrallel")
        model = CNN_GRU_Parrallel()
    elif model_name == 'ViT':
        print("using model: ViT_Parrallel")
        model = ViT_Parrallel()
    return unsupervised_train_loader, supervised_train_loader, test_loader, model