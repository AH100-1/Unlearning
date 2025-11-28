import argparse, torch
from unlearn_kd_class.model import get_model, get_unlearn_model
from unlearn_kd_class.dataset import get_cifar10, make_unlearn_retain_split, get_unlearn_dataloaders, get_dataloader
from unlearn_kd_class.utils.unlearn_distill import unlearn_distill, UKDConfig
from unlearn_kd_class.metric import get_membership_attack_prob, eval_retain_acc, eval_forget_acc
import pandas as pd
from unlearn_kd_class.utils.seed import set_seed
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from unlearn_kd_class.utils.distiller import distill, KDConfig
from unlearn_kd_class.utils.train import train_teacher
from unlearn_kd_class.utils.seed import set_seed
from unlearn_kd_class.utils.test import test

from unlearn_kd_class.metric import get_membership_attack_prob, eval_retain_acc, eval_forget_acc

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# con1 : fullmodel_fulldataset -> stu_model->fulldatset
# con2 : fullmodel_fulldatset -> stu_model->retain_set
# con3 : fullmodel_retainset -> stu_model->retain_set

def control_1(teacher_name, student_name, train_loader, test_loader, unlearn_train_loader, retain_train_loader, unlearn_test_loader, retain_test_loader):
    global device
    teacher_path =f'unlearn_kd_class/ckpt/teachers/ckpt_{teacher_name}.pt' 
    student_path =f'unlearn_kd_class/ckpt/students/ckpt_{student_name}_{teacher_name}_kd.pt'

    teacher_model = get_model(teacher_name, num_classes=10).to(device)
    teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
    student_model = get_model(student_name, num_classes=10).to(device)
    student_model.load_state_dict(torch.load(student_path, map_location=device))

    #student eval
    retain_train_loss, retain_train_acc = eval_retain_acc(student_model, retain_train_loader, device)
    forget_train_loss, forget_train_acc = eval_forget_acc(student_model, unlearn_train_loader, device)

    retain_test_loss, retain_test_acc = eval_retain_acc(student_model, retain_test_loader, device)
    forget_test_loss, forget_test_acc = eval_forget_acc(student_model, unlearn_test_loader, device)
    mia_rate = get_membership_attack_prob(retain_train_loader, unlearn_test_loader, retain_test_loader, student_model, device)

    print(f'{teacher_name}_{student_name}_con1 : fullmodel_fulldataset -> stu_model->fulldataset')
    print(f'retain_train_acc : {retain_train_acc:.2f}, forget_train_acc : {forget_train_acc:.2f}, mia_rate : {mia_rate:.2f}')
    print(f'retain_test_acc : {retain_test_acc:.2f}, forget_test_acc : {forget_test_acc:.2f}')


def control_2(teacher_name, student_name, train_loader, test_loader, unlearn_train_loader, retain_train_loader, unlearn_test_loader, retain_test_loader):
    global device
    teacher_save_path =f'unlearn_kd_class/ckpt/teachers/ckpt_{teacher_name}.pt' 
    # student_path =f'unlearn_kd_class/ckpt/students/ckpt_{student_model}_{teacher_model}_kd.pt'
    student_save_path = f'/data/khw/unlearn_kd_class/ckpt/retain_models/ckpt_{student_name}_retainset_from_{teacher_name}_fullset.pt'
    teacher_model = get_model(teacher_name, num_classes=10).to(device)
    teacher_model.load_state_dict(torch.load(teacher_save_path, map_location=device))
    student_model = get_model(student_name, num_classes=10).to(device)

    

    KD_cfg = KDConfig(T=2.0, alpha=0.7)
    _ = distill(student_model, teacher_model, retain_train_loader, retain_test_loader, device, epochs=50, cfg=KD_cfg, optimizer=None, scheduler=None, save=student_save_path)

    student_model.load_state_dict(torch.load(student_save_path, map_location=device))

    retain_train_loss, retain_train_acc = eval_retain_acc(student_model, retain_train_loader, device)
    forget_train_loss, forget_train_acc = eval_forget_acc(student_model, unlearn_train_loader, device)

    #student eval
    retain_test_loss, retain_test_acc = eval_retain_acc(student_model, retain_test_loader, device)
    forget_test_loss, forget_test_acc = eval_forget_acc(student_model, unlearn_test_loader, device)
    mia_rate = get_membership_attack_prob(retain_train_loader, unlearn_test_loader, retain_test_loader, student_model, device)

    print(f'{teacher_name}_{student_name}_con1 : fullmodel_fulldataset -> stu_model->retain_dataset')
    print(f'retain_train_acc : {retain_train_acc:.2f}, forget_train_acc : {forget_train_acc:.2f}, mia_rate : {mia_rate:.2f}')
    print(f'retain_test_acc : {retain_test_acc:.2f}, forget_test_acc : {forget_test_acc:.2f}')


def control_3(teacher_name, student_name, train_loader, test_loader, unlearn_train_loader, retain_train_loader, unlearn_test_loader, retain_test_loader):
    global device

    teacher_save_path = f'/data/khw/unlearn_kd_class/ckpt/retain_models/ckpt_{teacher_name}_retainset.pt'
    # student_path =f'unlearn_kd_class/ckpt/students/ckpt_{student_model}_{teacher_model}_kd.pt'
    student_save_path = f'/data/khw/unlearn_kd_class/ckpt/retain_models/ckpt_{student_name}_retainset_from_{teacher_name}_retainset.pt'

    teacher_model = get_unlearn_model(teacher_name, num_classes=10).to(device)
    student_model = get_model(student_name, num_classes=10).to(device)

    _ =  train_teacher(teacher_model, retain_train_loader, retain_test_loader, device, epochs=100, lr=0.1, wd=5e-4, momentum=0.9, save_path=teacher_save_path)
    teacher_model.load_state_dict(torch.load(teacher_save_path, map_location=device))

    KD_cfg = KDConfig(T=2.0, alpha=0.7)
    _ = distill(student_model, teacher_model, retain_train_loader, retain_test_loader, device, epochs=50, cfg=KD_cfg, optimizer=None, scheduler=None, save=student_save_path)
    student_model.load_state_dict(torch.load(student_save_path, map_location=device))

    #student eval
    retain_train_loss, retain_train_acc = eval_retain_acc(student_model, retain_train_loader, device)
    forget_train_loss, forget_train_acc = eval_forget_acc(student_model, unlearn_train_loader, device)
    retain_test_loss, retain_test_acc = eval_retain_acc(student_model, retain_test_loader, device)
    forget_test_loss, forget_test_acc = eval_forget_acc(student_model, unlearn_test_loader, device)
    mia_rate = get_membership_attack_prob(retain_train_loader, unlearn_test_loader, retain_test_loader, student_model, device)
    print(f'{teacher_name}_{student_name}_con1 : fullmodel_retaindataset -> stu_model->retain_dataset')
    print(f'retain_train_acc : {retain_train_acc:.2f}, forget_train_acc : {forget_train_acc:.2f}, mia_rate : {mia_rate:.2f}')
    print(f'retain_test_acc : {retain_test_acc:.2f}, forget_test_acc : {forget_test_acc:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--teacher_model', type=str, default='resnet18', choices=['r18', 'r34', 'r50'], help='model architecture (default: resnet18)')
    parser.add_argument('--student_model', type=str, default='resnet18', choices=['r18', 'r34', 'r50'], help='model architecture (default: resnet18)')
    parser.add_argument('--unlearn_class', type=int, default=0, help='class index to unlearn (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=128, help='test batch size (default: 256)')
    args = parser.parse_args()


    set_seed(args.seed)

    # Load CIFAR-10 dataset
    train_dataset, test_dataset = get_cifar10()
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size)
    test_loader = get_dataloader(test_dataset, batch_size=args.test_batch_size)

    # Split dataset into unlearn and retain sets
    unlearn_train_set, retain_train_set = make_unlearn_retain_split(train_dataset, args.unlearn_class)
    unlearn_test_set, retain_test_set = make_unlearn_retain_split(test_dataset, args.unlearn_class)

    unlearn_train_loader, retain_train_loader = get_unlearn_dataloaders(unlearn_train_set, retain_train_set, args.batch_size)
    unlearn_test_loader, retain_test_loader = get_unlearn_dataloaders(unlearn_test_set, retain_test_set, args.test_batch_size)

    control_1(args.teacher_model, args.student_model, train_loader, test_loader, unlearn_train_loader, retain_train_loader, unlearn_test_loader, retain_test_loader)
    control_2(args.teacher_model, args.student_model, train_loader, test_loader, unlearn_train_loader, retain_train_loader, unlearn_test_loader, retain_test_loader)
    control_3(args.teacher_model, args.student_model, train_loader, test_loader, unlearn_train_loader, retain_train_loader, unlearn_test_loader, retain_test_loader)
