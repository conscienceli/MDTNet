import torch
from utils.dataset import gen_train_loaders
import time
import copy
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
import os
import sklearn.metrics

def train_model(model_name, model, dataloaders_all,device, optimizer, loss_func, scheduler, num_epochs=25):
    existing_epoches = 0

    need_mse = False
    need_acc = False
    need_acc_single_class = False
    if model_name == 'type':
        need_acc = True
        topk = (1,3,)
    elif model_name == 'correct':
        need_acc_single_class = True
        topk = (1,)
    elif model_name.find('level_cls') >= 0:
        need_acc = True
        need_mse = True
        topk = (1,)
    elif model_name == 'level':
        need_mse = True
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    
    dataloaders = {
        'train': dataloaders_all[0],
        'val': dataloaders_all[1]
    }

    if os.path.isfile(f'./saved_models/{model_name}.pth.tar'):
            print(f"=> loading checkpoint './saved_models/{model_name}.pth.tar'")
            checkpoint = torch.load(f'./saved_models/{model_name}.pth.tar')
            existing_epoches = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint  (epoch {existing_epoches})")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + existing_epoches, num_epochs - 1 + existing_epoches))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_samples = 0
            
            if need_acc:
                maxk = max(topk)
                corrects = [0 for i in topk]
                totals = [0 for i in topk]
            
            if need_acc_single_class:
                correct = 0
                total = 0

            if need_mse:
                se = 0
                totals_se = 0
            
            Y_true = []
            Y_pred = []
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                if model_name == 'level_cls':
                    labels = labels.to(device, dtype=torch.int64)             
                else:
                    labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train' and model_name.find('inception') >= 0:
                        outputs = model(inputs)[0]
                    else:
                        outputs = model(inputs)
                    loss = loss_func(outputs, labels.long())

                    if need_acc:
                        _, pred_acc = outputs.topk(maxk, 1, True, True)
                        pred_acc = pred_acc.t()
                        correct = pred_acc.eq(labels)

                        for k_id, k in enumerate(topk):
                            correct_k = correct[:k].view(-1).float().sum(0)
                            corrects[k_id] += correct_k
                            totals[k_id] += len(labels)

                    if need_acc_single_class:
                        outputs_category = outputs > 0.0
                        correct += outputs_category.eq(labels > 0).sum()
                        total += len(labels)

                    if len(outputs) > 1 and need_mse:
                        preds_mse = outputs.max(1)[1]
                        labels_mse = labels
                        se += ((preds_mse-labels_mse)*(preds_mse-labels_mse)).sum()
                        totals_se += len(labels)

                    if need_mse and not need_acc:
                        y_pred = outputs.cpu().detach().numpy()
                        Y_pred.extend(y_pred.reshape(y_pred.shape[0]).tolist())
                        y_true = labels.cpu().detach().numpy()
                        Y_true.extend(y_true.reshape(y_true.shape[0]).tolist())


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
            
            if need_acc:
                for k_id, k in enumerate(topk):
                    print(f'accuracy on {phase}, top{k}: {100 * corrects[k_id] / totals[k_id]: .2f}%', end='\t')
            if need_mse and need_acc:
                print(f'mse on {phase}: {float(se) / float(totals_se) : .2f}')
            if need_mse and not need_acc:
                print(f'mse on {phase}: {sklearn.metrics.mean_squared_error(Y_true, Y_pred) : .2f}')
            if need_acc_single_class:
                print(f'accuracy on {phase}: {100 * correct / total: .2f}%')


            epoch_loss = loss / epoch_samples
            print(phase, " loss: ", epoch_loss.item())

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                is_best = True
                best_loss = epoch_loss
            else:
                is_best = False

            save_checkpoint({
                'epoch': existing_epoches + epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
                }, is_best, filename=model_name)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_checkpoint(state, is_best, filename):
    os.makedirs('./saved_models', exist_ok=True)
    torch.save(state, './saved_models/' + filename+'.pth.tar')
    if is_best:
        print("saving best model")
        shutil.copyfile('./saved_models/'+filename+'.pth.tar', './saved_models/'+filename+'_best.pth.tar')
