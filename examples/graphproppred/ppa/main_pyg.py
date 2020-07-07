import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN, GAT

from tqdm import tqdm
import argparse
import time
import numpy as np

import torch.nn.functional as F
from util import AverageMeter
from progress.bar import Bar as Bar

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

multicls_criterion = torch.nn.CrossEntropyLoss()

def add_virtualnode(graph, device):
    # Batch(batch=[742], edge_attr=[12186, 7], edge_index=[2, 12186], x=[742], y=[4, 1])
    batch_size = graph.y.shape[0]
    ## create the virtual node.
    old_node_num = graph.batch.shape[0]
    new_node_num = old_node_num + batch_size

    vn_node_num = []
    for b in range(0, batch_size):
        vn_node_num.append(list(graph.batch[:]).count(b))

    # new node batch.
    vn_batch = torch.cat([graph.batch[:], torch.tensor([m for m in range(0,batch_size)], dtype=torch.long)]) #.to(device)

    # new node type.
    vn_x = torch.cat([graph.x[:], torch.tensor([1 for _ in range(0, batch_size)], dtype=torch.long)]) #.to(device)

    edge_fea_num = graph.edge_attr.shape[1] 
    n_edge_attr = []
    n_edges_1 = []
    n_edges_2 = []
    s_idx = 0
    for b in range(0, batch_size):
        root_id = old_node_num + b
        
        for s in range(vn_node_num[b]):
            n_edges_1.append(root_id)
            n_edges_2.append(s_idx + s)
            n_edge_attr.append([0.0 for _ in range(0,edge_fea_num)] + [1.0, 0.0])

            n_edges_1.append(s_idx + s)
            n_edges_2.append(root_id)
            n_edge_attr.append([0.0 for _ in range(0,edge_fea_num)] + [0.0, 1.0])

        s_idx = s_idx + vn_node_num[b]

    # new edge index.
    vn_edge_index = torch.cat([graph.edge_index, torch.tensor([n_edges_1, n_edges_2], dtype=torch.long)], dim=1) #.to(device)

    _edge_attr = F.pad(input=graph.edge_attr, pad=(0,2,0,0), mode='constant', value=0.0)

    # new edge attr.
    vn_edge_attr = torch.cat([_edge_attr, torch.tensor(n_edge_attr)]) #.to(device)

    # new edge label.
    #vn_y = graph.y#.to(device)
    graph.x = vn_x
    graph.batch = vn_batch
    graph.edge_index = vn_edge_index
    graph.edge_attr = vn_edge_attr

    return graph # vn_x, vn_batch, vn_edge_index, vn_edge_attr, vn_y



def train(model, device, loader, optimizer, is_virtual_node = False):
    model.train()
    avg_loss = AverageMeter()
    
    bar = Bar('training', max=len(loader))

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            if is_virtual_node:
                batch = add_virtualnode(batch, device)    
            
            batch = batch.to(device)
            batch_size = batch.y.shape[0]
            y = batch.y
            
            pred = model(batch)
            
            optimizer.zero_grad()
            loss = multicls_criterion(pred.to(torch.float32), y.view(-1,))
            avg_loss.update(loss.item())

            loss.backward()
            optimizer.step()


        bar.suffix = '({step}/{size}) Total: {total:} | ETA: {eta:} | avg_loss: {avg_loss.val:.3f} ({avg_loss.avg:.3f})'.format(
                      step=step,
                      size=len(loader),
                      total=bar.elapsed_td,
                      eta=bar.eta_td,
                      avg_loss=avg_loss)
        if step % 10 == 0:
            print(bar.suffix)
        bar.next()
    bar.finish()
def eval(model, device, loader, evaluator, is_virtual_node = False):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        if batch.x.shape[0] == 1:
            pass
        else:
            if is_virtual_node:
                batch = add_virtualnode(batch, device)    
            
            #
            #    pred = model(x, edge_index, edge_attr, vn_batch, batch_size)
            #else:
            batch = batch.to(device)
            batch_size = batch.y.shape[0]

            with torch.no_grad():
                pred = model(batch)
            y = batch.y

            y_true.append(y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa)')

    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting

    dataset = PygGraphPropPredDataset(name = args.dataset, transform = add_zeros)

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    is_virtual_node = False
    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gat':
        #def __init__(self, num_class, num_layer = 5, emb_dim = 256, num_heads=8):
        model = GAT(num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, num_heads = 8).to(device)
        #is_virtual_node = True
    #elif args.gnn == 'gat_v2':
    #   model = GAT(num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, num_heads = 8).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, is_virtual_node)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator, is_virtual_node)
        valid_perf = eval(model, device, valid_loader, evaluator, is_virtual_node)
        test_perf = eval(model, device, test_loader, evaluator, is_virtual_node)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if not args.filename == '':
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = max(train_curve)
            torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()