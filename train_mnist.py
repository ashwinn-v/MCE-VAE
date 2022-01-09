from numpy.lib.twodim_base import tri
import optuna 
import numpy as np
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from MCEVAE import MCEVAE
from utils import load_checkpoint
import argparse
from prettytable import PrettyTable 

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN

  
'''
funcs:
   train(..):
        return
        torch.save(state, 'models/' + modelname)
        np.save('losses/trainloss_' + modelname.replace("_checkpoint", ""), train_loss_record)
        np.save('losses/testloss_' + modelname.replace("_checkpoint", ""), test_loss_record)

        Called funcs:
            calcloss
                return loss, RE/x.shape[0], divergence_var_tau, divergence_c
            train_epoch
                return train_loss/c, train_reco_loss/c, train_div_var_tau/c, train_div_c/c
            test_epoch
                return train_loss/c, train_reco_loss/c, train_div_var_tau/c, train_div_c/c
        
'''


def calc_loss(model, x, x_init, beta=1., n_sampel=4):
    # print('x is ', x.size())
    x_hat, z_var_q, z_var_q_mu, z_var_q_logvar, \
    z_c_q, z_c_q_mu, z_c_q_logvar, z_c_q_L, tau_q, tau_q_mu, tau_q_logvar, x_rec, M = model(x)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    x = x.view(-1, model.in_size).to(device)
    x_hat = x_hat.view(-1, model.in_size)
    x_rec = x_rec.view(-1, model.in_size)
        
    if model.rec_loss == 'mse':
        RE = torch.sum((x - x_hat)**2)
        if model.tau_size > 0 and model.training_mode == 'supervised':
            RE_INV = torch.sum((x_rec - x_init)**2)
        elif model.tau_size > 0 and model.training_mode == 'unsupervised':
            RE_INV = torch.FloatTensor([0.]).to(device)
            for jj in range(25):
                with torch.no_grad():
                    x_arb = model.get_x_ref(x.view(-1,1,int(np.sqrt(model.in_size)),int(np.sqrt(model.in_size))), tau_q)
                    z_aug_arb = model.aug_encoder(x_arb)
                    z_c_q_mu_arb, z_c_q_logvar_arb, _ = model.q_z_c(z_aug_arb)
                    z_c_q_arb = model.reparameterize(z_c_q_mu_arb, z_c_q_logvar_arb).to(device)
                    z_var_q_mu_arb, z_var_q_logvar_arb = model.q_z_var(z_aug_arb)
                    z_var_q_arb = model.reparameterize(z_var_q_mu_arb, z_var_q_logvar_arb).to(device)
                    x_init, _ = model.reconstruct(z_var_q_arb, z_c_q_arb)
                    x_init = x_init.view(-1, model.in_size).to(device)
                    x_init = torch.clamp(x_init, 1.e-5, 1-1.e-5)
                RE_INV = RE_INV + torch.sum((z_var_q_arb - z_var_q)**2)
                RE_INV = RE_INV + torch.sum((z_c_q_arb - z_c_q)**2) 
                RE_INV = RE_INV + torch.sum((x_rec - x_init)**2)
            RE_INV = RE_INV/25.0
        else:
            RE_INV = torch.FloatTensor([0.]).to(device)
    elif model.rec_loss == 'bce':
        x_hat = torch.clamp(x_hat, 1.e-5, 1-1.e-5)
        x = torch.clamp(x, 1.e-5, 1-1.e-5)
        x_init = torch.clamp(x_init, 1.e-5, 1-1.e-5)
        x_rec = torch.clamp(x_rec, 1.e-5, 1-1.e-5)
        RE = -torch.sum((x*torch.log(x_hat) + (1-x)*torch.log(1-x_hat)))
        if model.tau_size > 0 and model.training_mode == 'supervised':
            x_init = x_init.view(-1, model.in_size).to(device)
            RE_INV = -torch.sum((x_init*torch.log(x_rec) + (1-x_init)*torch.log(1-x_rec)))
        elif model.tau_size > 0 and model.training_mode == 'unsupervised':
            RE_INV = torch.FloatTensor([0.]).to(device)
            for jj in range(25):
                with torch.no_grad():
                    x_arb = model.get_x_ref(x.view(-1,1,int(np.sqrt(model.in_size)),int(np.sqrt(model.in_size))), tau_q)
                    z_aug_arb = model.aug_encoder(x_arb)
                    z_c_q_mu_arb, z_c_q_logvar_arb, _ = model.q_z_c(z_aug_arb)
                    z_c_q_arb = model.reparameterize(z_c_q_mu_arb, z_c_q_logvar_arb).to(device)
                    z_var_q_mu_arb, z_var_q_logvar_arb = model.q_z_var(z_aug_arb)
                    z_var_q_arb = model.reparameterize(z_var_q_mu_arb, z_var_q_logvar_arb).to(device)
                    x_init, _ = model.reconstruct(z_var_q_arb, z_c_q_arb)
                    x_init = x_init.view(-1, model.in_size).to(device)
                    x_init = torch.clamp(x_init, 1.e-5, 1-1.e-5)
                RE_INV = RE_INV + torch.sum((z_var_q_arb - z_var_q)**2)
                RE_INV = RE_INV + torch.sum((z_c_q_arb - z_c_q)**2) 
                RE_INV = RE_INV - torch.sum((x_init*torch.log(x_rec) + (1-x_init)*torch.log(1-x_rec)))
            RE_INV = RE_INV/25.0
        else:
            RE_INV = torch.FloatTensor([0.]).to(device)
    else:
        raise NotImplementedError

    if z_var_q.size()[0] == 0:
        log_q_z_var, log_p_z_var = torch.FloatTensor([0.]).to(device), torch.FloatTensor([0.]).to(device)
    else:
        log_q_z_var = -torch.sum(0.5*(1 + z_var_q_logvar))
        log_p_z_var = -torch.sum(0.5*(z_var_q**2 )) 
        
    if tau_q.size()[0] == 0:
        log_q_tau, log_p_tau = torch.FloatTensor([0.]).to(device), torch.FloatTensor([0.]).to(device)
    else:
        log_q_tau = -torch.sum(0.5*(1 + tau_q_logvar))
        log_p_tau = -torch.sum(0.5*(tau_q**2 ))
    if z_c_q.size()[0] == 0:
        log_q_z_c, log_p_z_c = torch.FloatTensor([0.]).to(device), torch.FloatTensor([0.]).to(device)
    else:
        log_q_z_c = -torch.sum(0.5*(1 + z_c_q_logvar/model.latent_z_c + \
                                       (model.latent_z_c -1)*z_c_q**2/model.latent_z_c))
        log_p_z_c = -torch.sum(0.5*(z_c_q**2 )) + torch.sum(z_c_q)/model.latent_z_c

    likelihood = - (RE + RE_INV)/x.shape[0]
    divergence_c = (log_q_z_c - log_p_z_c)/x.shape[0]
    divergence_var_tau = (log_q_z_var - log_p_z_var)/x.shape[0]  + (log_q_tau - log_p_tau)/x.shape[0]


    loss = - likelihood + beta * divergence_var_tau + divergence_c
    return loss, RE/x.shape[0], divergence_var_tau, divergence_c


def train_epoch(data, model, optim, epoch, num_epochs, N, beta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    train_loss = 0
    train_reco_loss= 0
    train_div_var_tau = 0
    train_div_c = 0
    c = 0
    for (x, x_init) in data:
        b = x.size(0)
        x = x.view(-1, 1, int(np.sqrt(model.in_size)), int(np.sqrt(model.in_size))).to(device).float()
        optim.zero_grad()
        loss, reco_loss, divergence_var_tau, divergence_c = calc_loss(model, x, x_init, beta = beta)
        loss.backward()
        optim.step()
        c += 1
        train_loss += loss.item()
        train_reco_loss += reco_loss.item()
        train_div_var_tau += divergence_var_tau.item()
        train_div_c += divergence_c.item()
        template = '# [{}/{}] training {:.1%}, ELBO={:.5f}, Reco Error={:.5f}, Disent KL={:.5f}, Ent KL={:.5f}'
        line = template.format(epoch + 1, num_epochs, c / N, train_loss/c, train_reco_loss/c, train_div_var_tau/c, train_div_c/c)
        print(line, end = '\r', file=sys.stderr)
    print(' ' * 80, end = '\r', file=sys.stderr)
    return train_loss/c, train_reco_loss/c, train_div_var_tau/c, train_div_c/c


def test_epoch(data, model, beta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    train_loss = 0
    train_reco_loss = 0
    train_div_var_tau = 0
    train_div_c = 0
    c = 0
    for (x, x_init) in data:
        b = x.size(0)
        x = x.view(-1, 1, int(np.sqrt(model.in_size)), int(np.sqrt(model.in_size))).to(device).float()
        with torch.no_grad():
            loss, reco_loss, divergence_var_tau, divergence_c = calc_loss(model, x, x_init, beta = beta)
        c += 1
        train_loss += loss.item()
        train_reco_loss += reco_loss.item()
        train_div_var_tau += divergence_var_tau.item()
        train_div_c += divergence_c.item()
    return train_loss/c, train_reco_loss/c, train_div_var_tau/c, train_div_c/c


def load_fold(n):
    mnist_SE2 = np.load('/content/Saver/se2_fold'+str(n)+'.npy')
    mnist_SE2_test = np.load('/content/Saver/se2_test_fold'+str(n)+'.npy')[:1000]
    mnist_SE2_init = np.load('/content/Saver/se2_init_fold'+str(n)+'.npy')
    mnist_SE2_init_test = np.load('/content/Saver/se2_init_test_fold'+str(n)+'.npy')[:1000]
    print('preparing dataset')
    batch_size = int(args.nBatch)
    trans_dataset = torch.utils.data.TensorDataset(torch.from_numpy(mnist_SE2), torch.from_numpy(mnist_SE2_init))
    trans_loader = torch.utils.data.DataLoader(trans_dataset, batch_size=batch_size)
    trans_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(mnist_SE2_test),
                                                        torch.from_numpy(mnist_SE2_init_test))
    trans_test_loader = torch.utils.data.DataLoader(trans_test_dataset, batch_size=batch_size)

    return trans_loader, trans_test_loader


def train(n,params, args, 
          tr_mode='new', beta = 1.0, save_model= False):
    c = '/content/'
    transformation = str(args.transformation).lower()
    
    myTable = PrettyTable(['Fold', 'train_loss (ELBO)', 'train_RE (Reco Error)', 'train_div_var_tau (Disent KL)', 'train_div_c (Ent KL)','test_loss', 'test_RE', 'test_div_var_tau', 'test_div_c']) 
    
    in_size = aug_dim = 28*28
    mode = transformation.upper()
    num_epochs = int(args.nEpochs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tag = str(args.tag)
    model = MCEVAE(in_size=in_size,
                    aug_dim=aug_dim,
                    latent_z_c=int(args.nCat),
                    latent_z_var=int(args.nVar),
                    mode=mode, 
                    invariance_decoder='gated', 
                    rec_loss=str(args.loss_type), 
                    div='KL',
                    in_dim=1, 
                    out_dim=1, 
                    hidden_z_c=params['hidden_z_c'],
                    hidden_z_var=params['hidden_z_var'],
                    hidden_tau=params['hidden_tau'], 
                    activation=nn.Sigmoid,
                    training_mode=str(args.training_mode),
                    device = device,
                    tag = tag).to(device)
    lr = 1e-3
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_data, test_data = load_fold(n)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    modelname = "fold_{}_model_{}_{}_dEnt_{}_ddisEnt_{}_{}_{}_{}_checkpoint".format(str(n), model.mode, 
                                                                            model.invariance_decoder,
                                                                            model.latent_z_c,
                                                                            model.latent_z_var,
                                                                            model.tag,
                                                                            model.training_mode,
                                                                            model.rec_loss)
    # print(modelname)
    if tr_mode == 'resume' and os.path.exists('models/' + modelname):
        print("Loading old model")
        model, optim, epoch = load_checkpoint(model, optim, 'models/' + modelname)
        train_loss_record = np.load('losses/trainloss_' + modelname.replace("_checkpoint", "") + ".npy")
        test_loss_record = np.load('losses/testloss_' + modelname.replace("_checkpoint", "") + ".npy")
        n_trainrecord_old = len(train_loss_record)
        n_testrecord_old = len(test_loss_record)
        train_loss_record = np.append(train_loss_record, np.zeros(num_epochs))
        test_loss_record  = np.append(test_loss_record, np.zeros(num_epochs))
    else:
        n_trainrecord_old = 0
        n_testrecord_old = 0
        train_loss_record = np.zeros(num_epochs)
        test_loss_record = np.zeros(num_epochs)
    print("Training for fold", n)
    print('training...')
    N = len(train_data)
    vv = len(test_data)
    print('Training data  ',N)
    print('Test data  ',vv)
    RE_best = 10000
    output = sys.stdout
    for epoch in range(num_epochs):
        train_loss, train_RE, train_div_var_tau, train_div_c = train_epoch(train_data, model, 
                                                                           optim, epoch, num_epochs, N, beta)
        line = '\t'.join([str(epoch + 1), 'train', str(train_loss), str(train_RE), str(train_div_var_tau), str(train_div_c)])
        print(line, file=output)
        output.flush()
        train_loss_record[n_trainrecord_old + epoch] = train_RE
        test_loss, test_RE, test_div_var_tau, test_div_c = test_epoch(test_data, model, beta) 
        line = '\t'.join([str(epoch + 1), 'test', str(test_loss), str(test_RE), str(test_div_var_tau), str(test_div_c)])
        print(line, file=output)
        output.flush()
        test_loss_record[n_testrecord_old + epoch] = test_RE
        if abs(RE_best) > abs(train_RE):
            RE_best = train_RE
            state = {'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'optimizer': optim.state_dict()}
            if save_model:
                torch.save(state, 'models/' + modelname)
    #         torch.save(state, 'models/' + modelname)
    # print('saving...')
    # np.save('losses/trainloss_' + modelname.replace("_checkpoint", ""), train_loss_record)
    # np.save('losses/testloss_' + modelname.replace("_checkpoint", ""), test_loss_record)
    ptt=[n,train_loss, train_RE, train_div_var_tau, train_div_c,test_loss, test_RE, test_div_var_tau, test_div_c]
    myTable.add_row(ptt)
    return RE_best

def finaltrain(params, args, 
          tr_mode='new', beta = 1.0, save_model= False):
    c = '/content/'
    transformation = str(args.transformation).lower()
    
    myTable = PrettyTable(['Fold', 'train_loss (ELBO)', 'train_RE (Reco Error)', 'train_div_var_tau (Disent KL)', 'train_div_c (Ent KL)','test_loss', 'test_RE', 'test_div_var_tau', 'test_div_c']) 
    
    in_size = aug_dim = 28*28
    mode = transformation.upper()
    num_epochs = int(args.nEpochs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tag = str(args.tag)
    model = MCEVAE(in_size=in_size,
                    aug_dim=aug_dim,
                    latent_z_c=int(args.nCat),
                    latent_z_var=int(args.nVar),
                    mode=mode, 
                    invariance_decoder='gated', 
                    rec_loss=str(args.loss_type), 
                    div='KL',
                    in_dim=1, 
                    out_dim=1, 
                    hidden_z_c=params['hidden_z_c'],
                    hidden_z_var=params['hidden_z_var'],
                    hidden_tau=params['hidden_tau'], 
                    activation=nn.Sigmoid,
                    training_mode=str(args.training_mode),
                    device = device,
                    tag = tag).to(device)
    lr = 1e-3
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mnist_SE2 = np.load('/content/DAT/mnist_se2.npy')
    mnist_SE2_test = np.load('/content/DAT/mnist_se2_test.npy')[:1000]
    mnist_SE2_init = np.load('/content/DAT/mnist_se2_init.npy')
    mnist_SE2_init_test = np.load('/content/DAT/mnist_se2_init_test.npy')[:1000]
    print('preparing dataset for final training')
    batch_size = int(args.nBatch)
    trans_dataset = torch.utils.data.TensorDataset(torch.from_numpy(mnist_SE2), torch.from_numpy(mnist_SE2_init))
    trans_loader = torch.utils.data.DataLoader(trans_dataset, batch_size=batch_size)
    trans_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(mnist_SE2_test),
                                                        torch.from_numpy(mnist_SE2_init_test))
    trans_test_loader = torch.utils.data.DataLoader(trans_test_dataset, batch_size=batch_size)
    
    train_data = trans_loader
    test_data = trans_test_loader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    modelname = "model_{}_{}_dEnt_{}_ddisEnt_{}_{}_{}_{}_checkpoint".format(model.mode, 
                                                                            model.invariance_decoder,
                                                                            model.latent_z_c,
                                                                            model.latent_z_var,
                                                                            model.tag,
                                                                            model.training_mode,
                                                                            model.rec_loss)
    # print(modelname)
    if tr_mode == 'resume' and os.path.exists('models/' + modelname):
        print("Loading old model")
        model, optim, epoch = load_checkpoint(model, optim, 'models/' + modelname)
        train_loss_record = np.load('losses/trainloss_' + modelname.replace("_checkpoint", "") + ".npy")
        test_loss_record = np.load('losses/testloss_' + modelname.replace("_checkpoint", "") + ".npy")
        n_trainrecord_old = len(train_loss_record)
        n_testrecord_old = len(test_loss_record)
        train_loss_record = np.append(train_loss_record, np.zeros(num_epochs))
        test_loss_record  = np.append(test_loss_record, np.zeros(num_epochs))
    else:
        n_trainrecord_old = 0
        n_testrecord_old = 0
        train_loss_record = np.zeros(num_epochs)
        test_loss_record = np.zeros(num_epochs)
    print("Training with best parameters")
    print('training...')
    N = len(train_data)
    vv = len(test_data)
    print('Training data  ',N)
    print('Test data  ',vv)
    RE_best = 10000
    output = sys.stdout
    for epoch in range(num_epochs):
        train_loss, train_RE, train_div_var_tau, train_div_c = train_epoch(train_data, model, 
                                                                           optim, epoch, num_epochs, N, beta)
        line = '\t'.join([str(epoch + 1), 'train', str(train_loss), str(train_RE), str(train_div_var_tau), str(train_div_c)])
        print(line, file=output)
        output.flush()
        train_loss_record[n_trainrecord_old + epoch] = train_RE
        test_loss, test_RE, test_div_var_tau, test_div_c = test_epoch(test_data, model, beta) 
        line = '\t'.join([str(epoch + 1), 'test', str(test_loss), str(test_RE), str(test_div_var_tau), str(test_div_c)])
        print(line, file=output)
        output.flush()
        test_loss_record[n_testrecord_old + epoch] = test_RE
        if abs(RE_best) > abs(train_RE):
            RE_best = train_RE
            state = {'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'optimizer': optim.state_dict()}
            if save_model:
                torch.save(state, 'models/' + modelname)
    #         torch.save(state, 'models/' + modelname)
    # print('saving...')
    # np.save('losses/trainloss_' + modelname.replace("_checkpoint", ""), train_loss_record)
    # np.save('losses/testloss_' + modelname.replace("_checkpoint", ""), test_loss_record)
    return RE_best

def train_evaluate(params):
    narr = [1,2,3,4]
    all_train_losses = []
    for i in narr:
        print("Fold ", i)
        RE_best = train(i,params,args, save_model=False)
        all_train_losses.append(RE_best)
    return np.mean(all_train_losses)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformation", help = "Transformation type: use so2 or se2", default = "so2")
    parser.add_argument("--loss_type", help = "Reconstruction Loss type: use mse or bce", default = "bce")
    parser.add_argument("--nCat", help = "Number of Categories", default = 10)
    parser.add_argument("--nVar", help = "Number of Variational Latent Dimensions", default = 3)
    parser.add_argument("--nBatch", help = "Batch size", default = 100)
    parser.add_argument("--nEpochs", help = "Number of Epochs", default = 60)
    # parser.add_argument("--nHiddenCat", help = "Number of Nodes in Hidden Layers for Categorical Latent Space", default = 512)
    # parser.add_argument("--nHiddenVar", help = "Number of Nodes in Hidden Layers for Variational Latent Space", default = 512)
    # parser.add_argument("--nHiddenTrans", help = "Number of Nodes in Hidden Layers for Transformational Latent Space", default = 32)
    parser.add_argument("--tag", help = "tag for model name", default = "default")
    parser.add_argument("--training_mode", help = "Training mode: use supervised or unsupervised", default = "supervised")
    parser.add_argument("--beta", help = "Beta for beta-VAE training", default = 1.0)     
    args = parser.parse_args()
    


    transformation = str(args.transformation).lower()
    
    myTable = PrettyTable(['Fold', 'train_loss (ELBO)', 'train_RE (Reco Error)', 'train_div_var_tau (Disent KL)', 'train_div_c (Ent KL)','test_loss', 'test_RE', 'test_div_var_tau', 'test_div_c']) 
    
    in_size = aug_dim = 28*28
    mode = transformation.upper()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "hidden_z_c", "type": "range", "bounds": [200,512], "log_scale": False},
            {"name": "hidden_z_var", "type": "range", "bounds": [200,512], "log_scale": False},
            {"name": "hidden_tau", "type": "range", "bounds": [32,128], "log_scale": False},
        ],
        evaluation_function=train_evaluate,
        total_trials = 1,
    )
    print(best_parameters)
    
    # scr = finaltrain(trial_.params,args,save_model=True)
    # print('Final train Reconstruction error:',scr) 