import argparse
import torch
import heapq
import random
import pylab
import copy
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from operator import itemgetter
from torch.autograd import Variable
from model_code import ADDSTCN
from model_code import train
from model_code import VAL
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import acf

# os.chdir(os.path.dirname(sys.argv[0])) #uncomment this line to run in VSCode

class StoreDictKeyPair(argparse.Action):
    """Creates dictionary containing datasets as keys and ground truth files as values."""
    def __call__(self, namespace, values):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

class CDCNN():

    def __init__(self, signi_2, sig_2, sig_3, sig_4, valepoc, levels, dilatation_c, nrepochs, 
                 kernel_size, loginterval, learningrate, valp, lags,
                 significance=0.95, seed=1234, optimizername="Adam", cuda=True, plot=True):
        self.signi_2 = signi_2
        self.sig_2 = sig_2
        self.sig_3 = sig_3
        self.sig_4 = sig_4
        self.valepoc = valepoc
        self.levels = levels
        self.dilatation_c = dilatation_c
        self.nrepochs = nrepochs
        self.kernel_size = kernel_size
        self.loginterval = loginterval
        self.learningrate = learningrate
        self.valp = valp
        self.lags = lags
        self.significance = significance
        self.seed = seed
        self.optimizername = optimizername
        self.cuda = cuda
        self.plot = plot

    def check_positive(self, value):
        """Checks if argument is positive integer (larger than zero)."""
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("%s should be positive" % value)
        return ivalue

    def check_zero_or_positive(self, value):
        """Checks if argument is positive integer (larger than or equal to zero)."""
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError("%s should be positive" % value)
        return ivalue

    def gen_allcauses(self, gtfile, columns):
        """Collects the total delay of indirect causal relationships."""
        gtdata = gtfile
        readgt = dict()
        for n in gtdata:
            if len(gtdata[n]) == 0:
                pass
            else:
                aux_df = self.dict_to_df(gtdata[n])
                effects = aux_df.iloc[:,0]
                causes = aux_df.iloc[:,1]
                if n==1:
                    for k in range(len(columns)):
                        readgt[k]=[]
                for i in range(len(effects)):
                    key=effects.iloc[i]
                    value=causes.iloc[i]
                    readgt[key].append(value)
        return readgt

    def gen_allcauses_2(self, gtfile, columns):
        """Collects the total delay of indirect causal relationships."""
        gtdata = gtfile
        readgt = dict()
        for n in gtdata:
            if len(gtdata[n]) == 0:
                pass
            else:
                aux_df = self.dict_to_df(gtdata[n])
                effects = aux_df.iloc[:,0]
                causes = aux_df.iloc[:,1]
                delays = aux_df.iloc[:,2]
                if n==1:
                    for k in range(len(columns)):
                        readgt[k]=[]
                for i in range(len(effects)):
                    key=0
                    value=causes.iloc[i]
                    readgt[key].append(value)
        return readgt

    def getextendeddelays(self, gtfile, columns):
        """Collects the total delay of indirect causal relationships."""
        gtdata = gtfile
        readgt_big=dict()
        extendreadgt = dict()
        extendgtdelays = dict()
        for n in gtdata:
            readgt = dict()
            aux_df = self.dict_to_df(gtdata[n])
            effects = aux_df.iloc[:,0]
            causes = aux_df.iloc[:,1]
            delays = aux_df.iloc[:,2]
            gtnrrelations = 0
            pairdelays = dict()
            for k in range(len(columns)):
                readgt[k]=[]
            for i in range(len(effects)):
                key=effects.iloc[i]
                value=causes.iloc[i]
                readgt[key].append(value)
                pairdelays[(key, value)]=delays.iloc[i]
                gtnrrelations+=1

            g = nx.DiGraph()
            g.add_nodes_from(readgt.keys())
            for e in readgt:
                cs = readgt[e]
                for c in cs:
                    g.add_edge(c, e)

            extendedreadgt = copy.deepcopy(readgt)

            for c1 in range(len(columns)):
                for c2 in range(len(columns)):
                    #indirect path max length 3, no cycles
                    paths = list(nx.all_simple_paths(g, c1, c2, cutoff=2)) 

                    if len(paths)>0:
                        for path in paths:
                            for p in path[:-1]:
                                if p not in extendedreadgt[path[-1]]:
                                    extendedreadgt[path[-1]].append(p)

            extendedgtdelays = dict()
            for effect in extendedreadgt:
                causes = extendedreadgt[effect]
                for cause in causes:
                    if (effect, cause) in pairdelays:
                        delay = pairdelays[(effect, cause)]
                        extendedgtdelays[(effect, cause)]=[delay]
                    else:
                        #find extended delay
                        paths = list(nx.all_simple_paths(g, cause, effect, cutoff=2))
                        extendedgtdelays[(effect, cause)]=[]
                        for p in paths:
                            delay=0
                            for i in range(len(p)-1):
                                delay+=pairdelays[(p[i+1], p[i])]
                            extendedgtdelays[(effect, cause)].append(delay)
            readgt_big[n] = readgt
            extendreadgt[n] = extendedreadgt
            extendgtdelays[n] = extendedgtdelays

        return extendgtdelays, readgt_big, extendreadgt

    def evaluate(self, gtfile, validatedcauses, columns, testcauses):
        # Read and process the ground truth and test data
        extendedgtdelays, readgt, extendedreadgt = self.getextendeddelays(gtfile, columns)
        print("extendedgtdelays: ", extendedgtdelays)
        print("readgt: ", readgt)
        print("extendedreadgt: ", extendedreadgt)

        # Initialize counters and lists for metrics for both directed and undirected evaluations
        FP, FPdirect, TPdirect, FNdirect, TP, FN, TN, TNdirect = 0, 0, 0, 0, 0, 0, 0, 0
        FPs, FPsdirect, TPsdirect, FNsdirect, TPs = [], [], [], [], []
        FPu, TPu, FNu, TNu = 0, 0, 0, 0
        FPus, TPus = [], []

        # Create sets of all possible cause-effect pairs for 6 variables
        all_possible_direct_causes = set((i, j) for i in columns for j in columns) 
        all_possible_undirected_causes = set((min(i, j), max(i, j)) for i in columns for j in columns)

        # Maps to store extended and direct ground truths
        gt_extended, gt_direct = {}, {}
        gt_extended_undirected, gt_direct_undirected = {}, {}
        for i in extendedreadgt:
            for key in extendedreadgt[i]:
                for cause in extendedreadgt[i][key]:
                    # Store directed and undirected pairs
                    directed_pair = (key, cause)
                    undirected_pair = (min(key, cause), max(key, cause))
                    gt_extended.setdefault(directed_pair, []).append(directed_pair)
                    gt_extended_undirected.setdefault(undirected_pair, []).append(undirected_pair)
            for key in readgt[i]:
                for cause in readgt[i][key]:
                    # Store directed and undirected pairs
                    directed_pair = (key, cause)
                    undirected_pair = (min(key, cause), max(key, cause))
                    gt_direct.setdefault(directed_pair, []).append(directed_pair)
                    gt_direct_undirected.setdefault(undirected_pair, []).append(undirected_pair)

        # Identified causes and calculate metrics for both directed and undirected
        predicted_causes = set()
        predicted_causes_undirected = set()
        for key, causes in validatedcauses.items():
            for cause in causes:
                directed_pair = (key, cause)
                undirected_pair = (min(key, cause), max(key, cause))

                # Directed calculations
                if directed_pair not in gt_extended:
                    FP += 1
                    FPs.append(directed_pair)
                else:
                    TP += 1
                    TPs.append(directed_pair)
                    gt_extended[directed_pair].remove(directed_pair)

                if directed_pair not in gt_direct:
                    FPdirect += 1
                    FPsdirect.append(directed_pair)
                else:
                    TPdirect += 1
                    TPsdirect.append(directed_pair)
                    gt_direct[directed_pair].remove(directed_pair)
                predicted_causes.add(directed_pair)

                # Undirected calculations
                if undirected_pair not in gt_extended_undirected:
                    FPu += 1
                    FPus.append(undirected_pair)
                else:
                    TPu += 1
                    TPus.append(undirected_pair)
                    gt_extended_undirected[undirected_pair].remove(undirected_pair)
                predicted_causes_undirected.add(undirected_pair)

        # Calculate false negatives and true negatives for both directed and undirected
        TN = len(all_possible_direct_causes) - len(gt_extended) - FP
        TNdirect = TN  # Simplification, they should be equivalent if calculated accurately
        TNu = len(all_possible_undirected_causes - predicted_causes_undirected)

        # Metrics calculation... (similar to previous calculations shown)
        # Include precision, recall, F1-score, and accuracy for both directed and undirected
        precision = TPu / (TPu + FPu) if TPu + FPu > 0 else 0
        recall = TPu / (TPu + FNu) if TPu + FNu > 0 else 0
        accuracy = (TPu + TNu) / (TPu + TNu + FPu + FNu) if (TPu + TNu + FPu + FNu) > 0 else 0
        F1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        precision_direct = TPdirect / (TPdirect + FPdirect) if TPdirect + FPdirect > 0 else 0
        recall_direct = TPdirect / (TPdirect + FNdirect) if TPdirect + FNdirect > 0 else 0
        accuracy_direct = (TPdirect + TNdirect) / (TPdirect + TNdirect + FPdirect + FNdirect) \
            if (TPdirect + TNdirect + FPdirect + FNdirect) > 0 else 0
        F1_direct = 2 * (precision_direct * recall_direct) / (precision_direct + recall_direct) \
            if precision_direct + recall_direct > 0 else 0

        # Reporting
        print(f"Total False Positives: {FPu}")
        print(f"Total True Positives: {TPu}")
        print(f"Total False Negatives: {FNu}")
        print(f"Total True Negatives: {TNu}")
        print(f"Total Direct False Positives: {FPdirect}")
        print(f"Total Direct True Positives: {TPdirect}")
        print(f"Total Direct False Negatives: {FNdirect}")
        print(f"Total Direct True Negatives: {TNdirect}")
        print(f"Percentage of Good Directions: {100 * (TPdirect / TPu) if TPu > 0 else 0:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision (direct): {precision_direct:.2f}")
        print(f"Recall (direct): {recall_direct:.2f}")
        print(f"Accuracy (direct): {accuracy_direct:.2f}")

        return FPu, TPu, FNu, TNu, FPdirect, TPdirect, FNdirect, TNdirect, F1, F1_direct, FPus, TPus, FNsdirect, FPsdirect, TPsdirect

    def evaluatedelay(self, extendedgtdelays, alldelays, TPs, receptivefield, Test):
        """Evaluates the delay discovery of TCDF by comparing the discovered time delays with the ground truth."""
        tot = 0
        for i in Test:
            tot += len(Test[i])
        zeros = 0
        total = 0.
        aux2 = extendedgtdelays.copy()
        for i in range(len(TPs)):
            tp=TPs[i]
            aux = []
            for k in alldelays:
                if tp in alldelays[k]:
                    aux.append(alldelays[k][tp])
            discovereddelay = aux
            for j in aux2:
                if tp in aux2[j]:
                    gtdelays = aux2[j][tp]
                    aux2[j].pop(tp)
                    for d in gtdelays:
                        if d <= receptivefield:
                            total+=1.
                            error = [d-x for x in discovereddelay]
                            try:
                                if min(error) == 0:
                                    zeros+=1
                            except:
                                pass
                        else:
                            next

        if zeros==0:
            return 0.
        else:
            return zeros/float(tot)

    def runTCDF(self, datafile):
        """Loops through all variables in a dataset and return the discovered causes, 
        time delays, losses, attention scores and variable names."""
        df_data = datafile

        allcauses = dict()
        for i in range(len(datafile.columns)):
            allcauses[i] = []
        alldelays = dict()
        allde_1 = dict()
        allde_2 = dict()
        allde_3 = dict()
        allreallosses=dict()
        allscores=dict()

        columns = list(df_data)
        for c in columns:
            idx = df_data.columns.get_loc(c)
            causes, causeswithdelay, realloss, scores = self.findcauses(c, 
                                        cuda=self.cuda, epochs=self.nrepochs,
                                        kernel_size=self.kernel_size, layers=self.levels, 
                                        log_interval=self.loginterval, lr=self.learningrate, 
                                        optimizername=self.optimizername, seed=self.seed, 
                                        dilation_c=self.dilation_c, significance=self.significance, 
                                        file=datafile, val_p=self.val_p, lags=self.lags)
            causes_aux=[]
            for j in range(len(causes)):
                for i in causeswithdelay:
                    if causes[j] in [x[1] for x in causeswithdelay[i].keys()]:
                        causes_aux.append(causes[j])
            try:
                allcauses[df_data.columns.get_loc(c)].extend(self.gen_allcauses_2(causeswithdelay, [c])[0])
            except:
                pass
            allscores[idx]=scores
            allde_1.update(causeswithdelay[1])
            allde_2.update(causeswithdelay[2])
            allde_3.update(causeswithdelay[3])
            allreallosses[idx]=realloss
        alldelays[1] = allde_1
        alldelays[2] = allde_2
        alldelays[3] = allde_3
        print("allcauses", allcauses)
        return allcauses, alldelays, allreallosses, allscores, columns

    def plotgraph(self, stringdatafile,alldelays,columns):
        """Plots a temporal causal graph showing all discovered causal relationships annotated 
        with the time delay between cause and effect."""
        all_aux = dict()
        lista = dict()
        for i in alldelays:
            lista_i = []
            for t in alldelays[i]:
                lista_i.append(t)
            lista[i] = lista_i
        for j in lista:
            for i in range(len(lista[j])):
                tp=lista[j][i]
                aux = []
                for k in alldelays:
                    if tp in alldelays[k]:
                        aux.append(alldelays[k][tp])
                all_aux[tp] = aux
        G = nx.DiGraph()
        for c in columns:
            G.add_node(c)
        for pair in all_aux:
            p1,p2 = pair
            nodepair = (columns[p2], columns[p1])

            G.add_edges_from([nodepair],weight=all_aux[pair])

        edge_labels=dict([((u,v,),d['weight'])
                        for u,v,d in G.edges(data=True)])

        pos=nx.circular_layout(G)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        nx.draw(G,pos, node_color = 'white', edge_color='black',node_size=1000,with_labels = True)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000")

    def df_to_dict(self, df):
        aux = {}
        for i in range(len(df)):
            a = (df.iloc[i,1], df.iloc[i,0])
            aux[a] = df.iloc[i,2]
        return aux

    def dict_to_df(self, dic):
        aux = []
        for i in dic:
            aux.append([i[0],i[1],dic[i]])
        aux = pd.DataFrame(aux)
        return aux

    def main(self, datafiles, evaluation, test=None):

        if evaluation:
            totalF1direct = [] #contains F1-scores of all datasets
            totalF1 = [] #contains F1'-scores of all datasets

            receptivefield=1+1
            for l in range(0, self.levels):
                receptivefield+=(self.kernel_size-1) * self.dilation_c**(l)
        
        #results of TCDF containing indices of causes and effects
        allcauses, alldelays, _, _, columns = self.runTCDF(datafiles) 

        print("\n===================Results for", datafiles.columns,"==================================")
        print("------------------------------------------------")
        for i in alldelays:
            print("Efecto número: ", i)
            print("------------------------------------------------")
            for pair in alldelays[i]:
                print(columns[pair[1]], "causes", columns[pair[0]],"with a delay of",alldelays[i][pair],"time steps.")
            print("------------------------------------------------")

        if evaluation:
            # evaluate TCDF by comparing discovered causes with ground truth
            testcauses = self.gen_allcauses(test, datafiles.columns)
            print("\n===================Evaluation for", datafiles.columns,"===============================")
            FP, TP, FN, TN, _, _, _, _, F1, F1_direct, _, TPs, _, _, _ = self.evaluate(test, allcauses, columns,testcauses)
            totalF1.append(F1)
            totalF1direct.append(F1_direct)

            # evaluate delay discovery
            extendeddelays, readgt, extendedreadgt = self.getextendeddelays(test, columns)
            percentagecorrect = self.evaluatedelay(extendeddelays, alldelays, TPs, receptivefield, test)*100
            print("Percentage of delays that are correctly discovered: ", percentagecorrect,"%")

        print("==================================================================================")

        if self.plot:
            self.plotgraph(datafiles.columns, alldelays, columns)
            if evaluation:
                self.plotgraph(datafiles.columns, test, columns)

    def preparedata(file, target):
        """Reads data from csv file and transforms it to two PyTorch tensors: 
        dataset x and target time series y that has to be predicted."""
        df_data = file
        df_y = df_data.copy(deep=True)[[target]]
        df_x = df_data.copy(deep=True)
        df_yshift = df_y.copy(deep=True).shift(periods=1, axis=0)
        df_yshift[target]=df_yshift[target].fillna(0.)
        df_x[target] = df_yshift
        data_x = df_x.values.astype('float32').transpose()
        data_y = df_y.values.astype('float32').transpose()
        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)

        x, y = Variable(data_x), Variable(data_y)
        return x, y
    
    def corr(self, data, targ, pot, lags, val_p):
        val = False
        cont = 0
        lag = []
        pvals = []
        if targ == pot:
            _, _, _, pval = acf(data[data.columns[targ]], qstat=True, fft=False, alpha=.05, nlags=lags)
            if pval[1] <= val_p:
                val = True
                cont +=1
                lag.append(1)
                pvals.append(pval[1])
        else:
            _,p = spearmanr(data[data.columns[targ]], data[data.columns[pot]])
            if p <= val_p:
                val = True
                cont +=1
                pvals.append(p)
            for k in range(1,lags+1):
                y = data[data.columns[pot]].shift(k)
                _,p = spearmanr(data[data.columns[targ]], y.fillna(0))
                if p <= val_p:
                    val = True
                    cont +=1
                    pvals.append(p)
            numbers_sort = sorted(enumerate(pvals), key=itemgetter(1),  reverse=False)
            for l in range(len(pvals)):
                index, _ = numbers_sort[l]
                lag.append(index)
        return val,cont,lag

    def findcauses(self, target, cuda, epochs, kernel_size, layers,
               log_interval, lr, optimizername, seed, dilation_c, 
               significance, file, val_p, lags):
        """Discovers potential causes of one target time series, validates these potential 
        causes with PIVM and discovers the corresponding time delays"""
        
        print("\n", "Analysis started for target: ", target)
        torch.manual_seed(seed)

        X_train, Y_train = self.preparedata(file, target)
        X_train = X_train.unsqueeze(0).contiguous()
        Y_train = Y_train.unsqueeze(2).contiguous()

        input_channels = X_train.size()[1]
        targetidx = file.columns.get_loc(target)
        model = ADDSTCN(targetidx, input_channels, layers, kernel_size=kernel_size, 
                        cuda=cuda, dilation_c=dilation_c)
        if cuda:
            model.cuda()
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()

        optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)

        scores, firstloss = train(1, X_train, Y_train, model, optimizer,log_interval,epochs)
        firstloss = firstloss.cpu().data.item()
        for ep in range(2, epochs+1):
            scores, realloss = train(ep, X_train, Y_train, model, optimizer,log_interval,epochs)
        realloss = realloss.cpu().data.item()
        s = sorted(np.abs(scores.view(-1).cpu().detach().numpy()), reverse=True)
        me=np.mean(s)
        st=np.std(s)
        indices = np.argsort(-1 *np.abs(scores.view(-1).cpu().detach().numpy()))
        corr_pot = []
        corr_pot_lags = {}
        potentials = []
        gaps = []
        for i in range(len(s)-1):
            if i==0:
                if s[i]<s[0]*self.signi_2 and s[i]<(me-st):
                    break
            else:
                if s[i]<s[i-1]*self.signi_2 and s[i]<(me-st):
                    break
            gap = (s[i]-s[i+1])/s[i]
            gaps.append(gap)
        sortgaps = sorted(gaps, reverse=True)
        ind = -1
        for i in range(0, len(gaps)):
            largestgap = sortgaps[i]
            index = gaps.index(largestgap)
            ind = -1
            if len(s)<=6:
                if index<=((len(s))):
                    if index>0:
                        ind=index
                        break
            else:
                if index<=((len(s)/2)+1):
                    if index>0:
                        ind=index
                        break
        if ind<0:
            ind = 0

        potentials = indices[:ind+1].tolist()
        for i in range(len(file.columns)):
            if i not in potentials and self.corr(file, targetidx, i, lags, val_p)[0]:
                potentials.append(i)
                corr_pot.append(i)
                corr_pot_lags[i] = (self.corr(file, targetidx, i, lags, val_p)[1], 
                                    self.corr(file, targetidx, i, lags, val_p)[2])
        print("Potential causes: ", potentials)
        validated = copy.deepcopy(potentials)
        for idx in potentials:
            wei = []
            for layer in range(layers):
                weight = model.dwn.network[layer].net[0].weight.abs().view(model.dwn.network[layer].net[0].weight.size()[0], 
                                                                           model.dwn.network[layer].net[0].weight.size()[2])
                wei.append(weight)
            totaldel_1=0
            for k in range(len(wei)):
                w=wei[k]
                row = w[idx]
                index_max_1 = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
                delay_1 = index_max_1 *(dilation_c**k)
                totaldel_1+=delay_1

            random.seed(seed)
            X_test2 = X_train.clone().cpu().numpy()
            random.shuffle(X_test2[:,idx,:][0])
            shuffled = torch.from_numpy(X_test2)
            if cuda:
                shuffled=shuffled.cuda()
            model.eval()
            output = model(shuffled)
            testloss = F.mse_loss(output, Y_train)
            testloss = testloss.cpu().data.item()

            diff = (firstloss-realloss)/firstloss
            testdiff = (firstloss-testloss)/firstloss
            if testdiff>(diff*(significance)):
                loss_dir_1, loss_inv_1 = VAL(file, targetidx, idx, lags, lr, self.valepoc)
                loss_dir = loss_dir_1
                loss_inv = loss_inv_1
                if self.sig_4*loss_dir>loss_inv or idx not in corr_pot:
                    validated.remove(idx)
        weights = []
        for layer in range(layers):
            weight = model.dwn.network[layer].net[0].weight.abs().view(model.dwn.network[layer].net[0].weight.size()[0], 
                                                                       model.dwn.network[layer].net[0].weight.size()[2])
            weights.append(weight)
        causeswithdelay = dict()
        causeswithdelay_1 = dict()
        causeswithdelay_2 = dict()
        causeswithdelay_3 = dict()
        vali = validated.copy()
        for v in validated:
            totaldelay_1=0
            totaldelay_2=0
            totaldelay_3=0
            if v not in corr_pot:
                for k in range(len(weights)):
                    w=weights[k]
                    row = w[v]
                    tlargest = heapq.nlargest(3, row)
                    m = tlargest[0]
                    m2 = tlargest[1]
                    m3 = tlargest[2]
                    if m > m2:
                        index_max_1 = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
                        if m2 >= self.sig_2*m:
                            numbers_sort = sorted(enumerate(row), key=itemgetter(1),  reverse=True)
                            index, _ = numbers_sort[1]
                            index_max_2 = len(row) - 1 - index
                            if m3 >= self.sig_3*m2:
                                numbers_sort = sorted(enumerate(row), key=itemgetter(1),  reverse=True)
                                index, _ = numbers_sort[2]
                                index_max_3 = len(row) - 1 - index
                            else:
                                index_max_3 = 0
                        else:
                            index_max_2 = 0
                            index_max_3 = 0
                    else:
                        index_max_1=0
                        index_max_2 = 0
                        index_max_3 = 0
                    delay_1 = index_max_1 *(dilation_c**k)
                    delay_2 = index_max_2 *(dilation_c**k)
                    delay_3 = index_max_3 *(dilation_c**k)
                    totaldelay_1+=delay_1
                    totaldelay_2+=delay_2
                    totaldelay_3+=delay_3
            else:
                for k in range(len(weights)):
                    w=weights[k]
                    row = w[v]
                    tlargest = heapq.nlargest(3, row)
                    m = tlargest[0]
                    m2 = tlargest[1]
                    m3 = tlargest[2]
                    index_max_1 = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
                    if corr_pot_lags[v][0]>1:
                        numbers_sort = sorted(enumerate(row), key=itemgetter(1),  reverse=True)
                        index, _ = numbers_sort[1]
                        index_max_2 = len(row) - 1 - index
                        if corr_pot_lags[v][0]>2:
                            numbers_sort = sorted(enumerate(row), key=itemgetter(1),  reverse=True)
                            index, _ = numbers_sort[2]
                            index_max_3 = len(row) - 1 - index
                        else:
                            index_max_3 = 0
                    else:
                        index_max_2 = 0
                        index_max_3 = 0
                    delay_1 = index_max_1 *(dilation_c**k)
                    delay_2 = index_max_2 *(dilation_c**k)
                    delay_3 = index_max_3 *(dilation_c**k)
                    totaldelay_1+=delay_1
                    totaldelay_2+=delay_2
                    totaldelay_3+=delay_3

            if targetidx != v:
                causeswithdelay_1[(targetidx, v)]=totaldelay_1
                if totaldelay_2!=0:
                    causeswithdelay_2[(targetidx, v)]=totaldelay_2
                    if totaldelay_3!=0:
                        causeswithdelay_3[(targetidx, v)]=totaldelay_3
            else:
                causeswithdelay_1[(targetidx, v)]=totaldelay_1+1
                if totaldelay_2!=0:
                    causeswithdelay_2[(targetidx, v)]=totaldelay_2+1
                    if totaldelay_3!=0:
                        causeswithdelay_3[(targetidx, v)]=totaldelay_3+1
        print("Validated Lags per cause:   PRIMERA CAUSA: ", causeswithdelay_1,"SEGUNDA CAUSA: ", 
              causeswithdelay_2, "TERCERA CAUSA: ", causeswithdelay_3)
        print("Validated causes: ", vali)
        causeswithdelay[1] = causeswithdelay_1
        causeswithdelay[2] = causeswithdelay_2
        causeswithdelay[3] = causeswithdelay_3

        return vali, causeswithdelay, realloss, scores.view(-1).cpu().detach().numpy().tolist()