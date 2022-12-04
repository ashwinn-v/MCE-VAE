import numpy as np
import torch.nn.functional as F
from MCEVAE import MCEVAE
from utils import load_checkpoint
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import sys

# from model_lenet import RegressionModel, RegressionTrain
# from model_resnet import MnistResNet, RegressionTrainResNet

from min_norm_solvers import MinNormSolver

import pickle
import pdb


def get_d_paretomtl_init(grads,value,weights,i):
    """ 
    calculate the gradient direction for ParetoMTL initialization 
    """
    
    flag = False
    nobj = value.shape
   
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
   
    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).float()
    else:
        vec =  torch.matmul(w[idx],grads)
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  torch.sum(torch.stack([sol[j] * w[idx][j ,0] for j in torch.arange(0, torch.sum(idx))]))
    weight1 =  torch.sum(torch.stack([sol[j] * w[idx][j ,1] for j in torch.arange(0, torch.sum(idx))]))
    weight = torch.stack([weight0,weight1])
   
    
    return flag, weight


def get_d_paretomtl(grads,value,weights,i):
    """ calculate the gradient direction for ParetoMTL """
    
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights 
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
    

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).float()


    vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight = torch.stack([weight0,weight1])
    
    return weight


def circle_points(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles

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
    a = [train_reco_loss/c, train_div_var_tau/c, train_div_c/c]
    task_loss = torch.stack(a)
    pdb.set_trace()
    return task_loss


def train(dataset, base_model, niter, npref, init_weight, pref_idx):

    # generate #npref preference vectors      
    n_tasks = 3
    ref_vec = torch.tensor(circle_points([1], [npref])[0]).float()
    
    # load dataset 

    # # MultiMNIST: multi_mnist.pickle
    # if dataset == 'mnist':
    #     with open('data/multi_mnist.pickle','rb') as f:
    #         trainX, trainLabel,testX, testLabel = pickle.load(f)  
    
    # # MultiFashionMNIST: multi_fashion.pickle
    # if dataset == 'fashion':
    #     with open('data/multi_fashion.pickle','rb') as f:
    #         trainX, trainLabel,testX, testLabel = pickle.load(f)  
    
    
    # # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    # if dataset == 'fashion_and_mnist':
    #     with open('data/multi_fashion_and_mnist.pickle','rb') as f:
    #         trainX, trainLabel,testX, testLabel = pickle.load(f)   

    # trainX = torch.from_numpy(trainX.reshape(120000,1,36,36)).float()
    # trainLabel = torch.from_numpy(trainLabel).long()
    # testX = torch.from_numpy(testX.reshape(20000,1,36,36)).float()
    # testLabel = torch.from_numpy(testLabel).long()
    
    
    # train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    # test_set  = torch.utils.data.TensorDataset(testX, testLabel)
    
    
    # batch_size = 256
    # train_loader = torch.utils.data.DataLoader(
    #                  dataset=train_set,
    #                  batch_size=batch_size,
    #                  shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #                 dataset=test_set,
    #                 batch_size=batch_size,
    #                 shuffle=False)
    
    # print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    # print('==>>> total testing batch number: {}'.format(len(test_loader))) 
    
    
    # # define the base model for ParetoMTL  
    # if base_model == 'lenet':
    #     model = RegressionTrain(RegressionModel(n_tasks), init_weight)
    # if base_model == 'resnet18':
    #     model = RegressionTrainResNet(MnistResNet(n_tasks), init_weight)
   
    
    # if torch.cuda.is_available():
    #     model.cuda()


    # # choose different optimizer for different base model
    # if base_model == 'lenet':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90], gamma=0.5)
    
    # if base_model == 'resnet18':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    
    
    # store infomation during optimization
    weights = []
    task_train_losses = []
    train_accs = []
    
    transformation = "se2"
    loss_type = "bce"
    nCat = 10
    nVar = 3
    nBatch = 100
    nEpochs = 60
    nHiddenCat = 512
    nHiddenVar = 512
    nHiddenTrans = 32
    tag = "default"
    training_mode = "supervised"
    beta = 1.0

    print('loading data...')
    c = '/content/'
    transformation = str(transformation).lower()
    mnist_SE2 = np.load('/Users/ashwinv/Desktop/data/mnist_se2_train.npy')
    mnist_SE2_test = np.load('/Users/ashwinv/Desktop/data/mnist_se2_test.npy')[:1000]
    mnist_SE2_init = np.load('/Users/ashwinv/Desktop/data/mnist_se2_init_train.npy')
    mnist_SE2_init_test = np.load('/Users/ashwinv/Desktop/data/mnist_se2_init_test.npy')[:1000]

    print('preparing dataset')
    batch_size = int(nBatch)
    trans_dataset = torch.utils.data.TensorDataset(torch.from_numpy(mnist_SE2), torch.from_numpy(mnist_SE2_init))
    trans_loader = torch.utils.data.DataLoader(trans_dataset, batch_size=batch_size)
    trans_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(mnist_SE2_test),
                                                        torch.from_numpy(mnist_SE2_init_test))
    trans_test_loader = torch.utils.data.DataLoader(trans_test_dataset, batch_size=batch_size)
    in_size = aug_dim = 28*28
    mode = transformation.upper()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = str(tag)
    model = MCEVAE(in_size=in_size,
                     aug_dim=aug_dim,
                     latent_z_c=int(nCat),
                     latent_z_var=int(nVar),
                     mode=mode, 
                     invariance_decoder='gated', 
                     rec_loss=str(loss_type), 
                     div='KL',
                     in_dim=1, 
                     out_dim=1, 
                     hidden_z_c=int(nHiddenCat),
                     hidden_z_var=int(nHiddenVar),
                     hidden_tau=int(nHiddenTrans), 
                     activation=nn.Sigmoid,
                     training_mode=str(training_mode),
                     device = device,
                     tag = tag).to(device)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # print the current preference vector
    print('Preference Vector ({}/{}):'.format(pref_idx + 1, npref))
    print(ref_vec[pref_idx].cpu().numpy())

    # run at most 2 epochs to find the initial solution
    # stop early once a feasible solution is found 
    # usually can be found with a few steps
    for t in range(2):
      
        model.train()
    # for (it, batch) in enumerate(trans_loader):
        # pdb.set_trace()
        # X = batch[0]
        # ts = batch[1]
        # if torch.cuda.is_available():
        #     X = X.cuda()
        #     ts = ts.cuda()

        grads = {}
        losses_vec = []
        N = len(trans_loader)
        epoch = t
        # obtain and store the gradient value
        for i in range(n_tasks):
            optimizer.zero_grad()
            task_loss= train_epoch(trans_loader,model, optimizer, epoch, nEpochs, N, beta)
            losses_vec.append(task_loss[i].data)
            
            task_loss[i].backward()
            
            grads[i] = []
            
            # can use scalable method proposed in the MOO-MTL paper for large scale problem
            # but we keep use the gradient of all parameters in this experiment
            for param in model.parameters():
                if param.grad is not None:
                    grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            
        
        grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
        grads = torch.stack(grads_list)
        
        # calculate the weights
        losses_vec = torch.stack(losses_vec)
        flag, weight_vec = get_d_paretomtl_init(grads,losses_vec,ref_vec,pref_idx)
        
        # early stop once a feasible solution is obtained
        if flag == True:
            print("fealsible solution is obtained.")
            break
        
        # optimization step
        optimizer.zero_grad()
        for i in range(len(task_loss)):
            task_loss = model(X, ts)
            if i == 0:
                loss_total = weight_vec[i] * task_loss[i]
            else:
                loss_total = loss_total + weight_vec[i] * task_loss[i]
        
        loss_total.backward()
        optimizer.step()
                
        # else:
        # # continue if no feasible solution is found
        #     continue
        # # break the loop once a feasible solutions is found
        # break
                
        

    # run niter epochs of ParetoMTL 
    for t in range(niter):
        
        # scheduler.step()
      
        model.train()
        # for (it, batch) in enumerate(train_loader):
        
        X = batch[0]
        ts = batch[1]
        if torch.cuda.is_available():
            X = X.cuda()
            ts = ts.cuda()

        # obtain and store the gradient 
        grads = {}
        losses_vec = []
        
        for i in range(n_tasks):
            optimizer.zero_grad()
            task_loss= train_epoch(trans_loader,model, optimizer, epoch, nEpochs, N, beta)
            losses_vec.append(task_loss[i].data)
            
            task_loss[i].backward()
        
            # can use scalable method proposed in the MOO-MTL paper for large scale problem
            # but we keep use the gradient of all parameters in this experiment              
            grads[i] = []
            for param in model.parameters():
                if param.grad is not None:
                    grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            
            
        grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
        grads = torch.stack(grads_list)
        
        # calculate the weights
        losses_vec = torch.stack(losses_vec)
        weight_vec = get_d_paretomtl(grads,losses_vec,ref_vec,pref_idx)
        
        normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
        weight_vec = weight_vec * normalize_coeff
        
        # optimization step
        optimizer.zero_grad()
        for i in range(len(task_loss)):
            task_loss = model(X, ts)
            if i == 0:
                loss_total = weight_vec[i] * task_loss[i]
            else:
                loss_total = loss_total + weight_vec[i] * task_loss[i]
        
        loss_total.backward()
        optimizer.step()


    #     # calculate and record performance
    #     if t == 0 or (t + 1) % 2 == 0:
            
    #         model.eval()
    #         with torch.no_grad():
  
    #             total_train_loss = []
    #             train_acc = []
        
    #             correct1_train = 0
    #             correct2_train = 0
                
    #             for (it, batch) in enumerate(train_loader):
                   
    #                 X = batch[0]
    #                 ts = batch[1]
    #                 if torch.cuda.is_available():
    #                     X = X.cuda()
    #                     ts = ts.cuda()
        
    #                 valid_train_loss = model(X, ts)
    #                 total_train_loss.append(valid_train_loss)
    #                 output1 = model.model(X).max(2, keepdim=True)[1][:,0]
    #                 output2 = model.model(X).max(2, keepdim=True)[1][:,1]
    #                 correct1_train += output1.eq(ts[:,0].view_as(output1)).sum().item()
    #                 correct2_train += output2.eq(ts[:,1].view_as(output2)).sum().item()
                    
                    
    #             train_acc = np.stack([1.0 * correct1_train / len(train_loader.dataset),1.0 * correct2_train / len(train_loader.dataset)])
        
    #             total_train_loss = torch.stack(total_train_loss)
    #             average_train_loss = torch.mean(total_train_loss, dim = 0)
                
            
    #         # record and print
    #         if torch.cuda.is_available():
                
    #             task_train_losses.append(average_train_loss.data.cpu().numpy())
    #             train_accs.append(train_acc)
                
    #             weights.append(weight_vec.cpu().numpy())
                
    #             print('{}/{}: weights={}, train_loss={}, train_acc={}'.format(
    #                     t + 1, niter,  weights[-1], task_train_losses[-1],train_accs[-1]))                 
               

    # torch.save(model.model.state_dict(), './saved_model/%s_%s_niter_%d_npref_%d_prefidx_%d.pickle'%(dataset, base_model, niter, npref, pref_idx))

    

def run(dataset = 'mnist',base_model = 'lenet', niter = 100, npref = 5):
    """
    run Pareto MTL
    """
    
    init_weight = np.array([0.5 , 0.5 ])
    
    for i in range(npref):
        
        pref_idx = i 
        train(dataset, base_model, niter, npref, init_weight, pref_idx)
        


run(dataset = 'mnist', base_model = 'lenet', niter = 100, npref = 5)
#run(dataset = 'fashion', base_model = 'lenet', niter = 100, npref = 5)
#run(dataset = 'fashion_and_mnist', base_model = 'lenet', niter = 100, npref = 5)

#run(dataset = 'mnist', base_model = 'resnet18', niter = 20, npref = 5)
#run(dataset = 'fashion', base_model = 'resnet18', niter = 20, npref = 5)
#run(dataset = 'fashion_and_mnist', base_model = 'resnet18', niter = 20, npref = 5