import torch
from utils.dataset import gen_train_loaders
import time
import copy
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def train_model(model_name, model, dataloaders_all,device, optimizer, loss_func, scheduler, num_epochs=25):
    need_mse = False
    need_acc = False
    if model_name == 'type':
        need_acc = True
        topk = (1,3,)
    elif model_name == 'level_cls':
        need_acc = True
        need_mse = True
        topk = (1,2,)
    elif model_name == 'level':
        need_mse = True
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    
    dataloaders = {
        'train': dataloaders_all[0],
        'val': dataloaders_all[1]
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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

            if need_mse:
                se = 0
                totals_se = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)

                    if need_acc:
                        _, pred_acc = outputs.topk(maxk, 1, True, True)
                        pred_acc = pred_acc.t()
                        correct = pred_acc.eq(labels.max(1)[1].view(1, -1).expand_as(pred_acc))

                        for k_id, k in enumerate(topk):
                            correct_k = correct[:k].view(-1).float().sum(0)
                            corrects[k_id] += correct_k
                            totals[k_id] += len(labels)

                    if len(outputs) > 1 and need_mse:
                        preds_mse = outputs.max(1)[1]
                        labels_mse = labels.max(1)[1]
                        se += ((preds_mse-labels_mse)*(preds_mse-labels_mse)).sum()
                        totals_se += len(labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
            
            if need_acc:
                for k_id, k in enumerate(topk):
                    print(f'accuracy on {phase}, top{k}: {100 * corrects[k_id] / totals[k_id]: .2f}%', end='\t')
            if need_mse:
                print(f'mse on {phase}: {float(se) / float(totals_se) : .2f}')


            epoch_loss = loss / epoch_samples
            print(phase, " loss: ", epoch_loss.item())

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'best_state_{model_name}.pth') 

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model