import os
import argparse
import numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon

def get_args():
    '''
    Function that gets the command-line arguments

    Parameters
    ----------
    None

    Returns
    -------
    A parsers with the input arguments
    '''

    parser = argparse.ArgumentParser(usage='Loads the results of the ensemble methods.')

    parser.add_argument('-input_folder', help='Folder where the results are located.', type=str)
    parser.add_argument('-output_folder', help='Folder where the LaTex table will be saved.', type=str) 
    parser.add_argument('-divergence_mode', help='Indicates the divergence mode for OPF ensemble. The default is \'countclass\'', type=str, default='countclass')   

    return parser.parse_args()

def load_ensemble_results(folder,n_classifiers):
    '''
    Function that loads the results of the cross-validation folds and computes the average of the MAE, MSE and execution time

    Parameters
    ----------
    folder: str
        Folder where the results of the cross-validation folds are located
    n_classifiers: list
        List with the number of models in the ensemble methods
        Each cross-validation fold was evaluated with a specific number of baseline models in the ensemble methods

    Returns
    -------
    mean_metrics: dict
        Dictionary with the average results computed for each dataset and each ensemble method for a specific number of baseline models
    '''

    if (not os.path.exists(folder)):
        raise SystemError('Folder {} does not exist'.format(folder))

    # A dictionary to store the mean and std of all n number of classifiers
    mean_metrics = dict()

    # For each number of baseline classifiers
    for n in n_classifiers:
        folds = os.listdir(folder)

        results_all = []
        
        for f in folds:
            # Check whether the folder associated with the number of classifiers exists or not
            if (not os.path.exists('{}/{}/{}'.format(folder,f,n))):
                print('Folder {} does not exist'.format('{}/{}/{}'.format(folder,f,n)))
                continue

            results = np.loadtxt('{}/{}/{}/results.txt'.format(folder,f,n),delimiter=',')
            results_all.append(results)

        # Computed the mean and the std for this n_th number of classifiers
        mean_metrics[n] = [np.mean(results_all,axis=0),np.std(results_all,axis=0),np.array(results_all)]

    # The dictionary is structured with nested dictionaries as follows:
    # -----------------------------------------------------------------
    # {dataset: {ensemble_method: {n: [[avg_accuracy,avg_f1, avg_time], [std_accuracy, std_f1, std_time]], ...}, ...}, ...}, 
    #      where 'n' is the number of baseline models.
    # In summary, for each number 'n' of baseline models, we have the average and standard deviation for the accuracy, f1-score and execution time.
    return mean_metrics

def latex_table_ensembles_metrics(mean_metrics,n_classifiers,metric='accuracy'):
    '''
    Function that constructs the LaTex table representation of the average results for each dataset and ensemble method.

    Parameters
    ----------
    folder: dict
        Dictionary with average accuracy and F1 scores
    n_classifiers: list
        List with the number of models in the ensemble methods
        Each cross-validation fold was evaluated with a specific number of baseline models in the ensemble methods

    Returns
    -------
    table: str
        String representation of the LaTex table
    '''

    if (metric != 'accuracy' and metric != 'f1'):
        raise SystemError('The parameter metric must be \'accuracy\' or \'f1\'')

    # Getting the ensemble methods from the dictionary
    # It is important to estimate the number of columns of the table
    ensembles = list(mean_metrics[next(iter(mean_metrics))].keys())
    n_colunmns = len(ensembles) + 1

    table='\\begin{table} \n'
    table+='\\centering \n'
    table+='\\caption{{Average {} for each dataset and ensemble approach.}} \n'.format(metric)
    table+='\\label{t.average_metrics_ensemble} \n'
    table+='\\begin{adjustbox}{width=1\\textwidth}{\\def\\arraystretch{1.5}\\tabcolsep=12pt \n'
    table+='\\begin{}{} \n'.format('{tabular}','{||l'+'||l'*len(ensembles)+'||}' )

    for n in n_classifiers:
        table+='\\hhline{|t:='+'='*len(ensembles)+':t|} \n'
        table+='\\multicolumn{}{}{}'.format('{'+str(n_colunmns)+'}','{||c||}','{'+str(n)+' classifiers} \\\\ \n')
        table+='\\hhline{|:=:'+'t:='*len(ensembles)+'|} \n'

        table+='\\multicolumn{1}{||c||}{}' +' '.join(['  &  \\multicolumn{}{}{}'.format('{1}','{c||}','{'+e+'}') for e in ensembles])+'\\\\ \n'
        table+='\\hline \n'     

        for d in mean_metrics:
            table+=d
            for m in mean_metrics[d]:
                if (metric=='accuracy'):
                    avg_ = '{:.4f}'.format(mean_metrics[d][m][n][0][0])
                    std_ = '{:.4f}'.format(mean_metrics[d][m][n][1][0])
                    idx_ = 0
                else:
                    avg_ = '{:.4f}'.format(mean_metrics[d][m][n][0][1])
                    std_ = '{:.4f}'.format(mean_metrics[d][m][n][1][1])
                    idx_ = 1
                
                # Checking the metric with the highest value for highlighting in bold
                highest_avg = -np.inf
                highest_avg_name = ''                
                for m1 in mean_metrics[d]:
                    avg1 = mean_metrics[d][m1][n][0][idx_]
                    if (avg1 > highest_avg):
                        highest_avg = avg1
                        highest_avg_name = m1

                if (m == highest_avg_name):              
                    table+='  &  \\textbf{{ {}$\\pm${} }}  '.format(avg_,std_)
                else:
                    table+='  &  {}$\\pm${}  '.format(avg_,std_)

            table+='\\\\ \n'


        table+='\n'

    table+='\\hhline{|b:=:'+'b:=:'*len(ensembles)+'b|} \n'
    table+='\\end{tabular}}\n'
    table+='\\end{adjustbox}\n'
    table+='\\end{table}'

    return table

def latex_table_ensemble_time(mean_metrics,n_classifiers):
    '''
    Function that constructs the LaTex table representation of the average execution time of each ensemble method.

    Parameters
    ----------
    folder: dict
        Dictionary with average and standard deviation values of the execution time
    n_classifiers: list
        List with the number of models in the ensemble methods
        Each cross-validation fold was evaluated with a specific number of baseline models in the ensemble methods

    Returns
    -------
    table: str
        String representation of the LaTex table.
        Note that this table is composed of two columns only: one for the Stacking and another for the OPF ensemble
    '''
        
    ensembles = mean_metrics[next(iter(mean_metrics))].keys()
    opf_key = [key for key in ensembles if 'OPF' in key][0]
    stacking_key = [key for key in ensembles if 'Stacking' in key][0]
    
    table='\\begin{table} \n'
    table+='\\centering \n'
    table+='\\caption{Average execution time for each dataset and ensemble approach.} \n'
    table+='\\label{t.average_time_ensemble} \n'
    table+='\\begin{adjustbox}{width=1\\textwidth}{\\def\\arraystretch{1.5}\\tabcolsep=12pt \n'
    table+='\\begin{tabular}{||l|c|c||} \n' 

    for n in n_classifiers:
        table+='\\hhline{|t:===:t|} \n'
        table+='\\multicolumn{}{}{}'.format('{3}','{||c||}','{'+str(n)+' classifiers} \\\\ \n')
        table+='\\hhline{|:===:|} \n'

        table+='Dataset  '+'  &   Stacking   &   OPF \\\\ \n'
        table+='\\hline \n' 

        for d in mean_metrics:
            table+=d

            exec_time_stack = '{:.4f}'.format(mean_metrics[d][stacking_key][n][0][2])
            std_exec_time_stack = '{:.4f}'.format(mean_metrics[d][stacking_key][n][1][2])

            exec_time_opf = '{:.4f}'.format(mean_metrics[d][opf_key][n][0][2])
            std_exec_time_opf = '{:.4f}'.format(mean_metrics[d][opf_key][n][1][2])

            table+='  &  {}$\\pm${}  &   {}$\\pm${}  \\\\ \n'.format(exec_time_stack,std_exec_time_stack,exec_time_opf,std_exec_time_opf)

    table+='\\hhline{|b:===:b|} \n'
    table+='\\end{tabular}}\n'
    table+='\\end{adjustbox}\n'
    table+='\\end{table}'
    
    return table

def latex_table_difference_divergence(input_folder,n_classifiers,divergence_metric,metric='accuracy'):
    '''
    Function that constructs the LaTex table with the difference between divergence and no divergence.

    Parameters
    ----------
    input_folder: str
        Folder where the OPF variation results are located
    n_classifiers: list
        A list with the number of classifiers in the ensemble
    diversity_metric: str
        Name of the OPF divergence metric
    metric: str
        The metric to be loaded (accuracy or f1). The defaults is \'accuracy\''

    Returns
    -------
    table: str
        A LaTex table representation
    '''

    if (metric != 'accuracy' and metric != 'f1'):
        raise SystemError('The parameter metric must be \'accuracy\' or \'f1\'') 

    if (divergence_metric == 'opf_nodivergence'):
        print('Divergence metric must be a valid metric and different from \'opf_nodivergence\'')
        return None

    dv_folder = None
    ndv_folder = None

    # Gets the divergence and no divergence folders
    dv_folder = [f for f in os.listdir(input_folder) if f==divergence_metric]
    ndv_folder = [f for f in os.listdir(input_folder) if f=='opf_nodivergence']

    # Checks if both folders exist
    if (dv_folder is None or ndv_folder is None):
        raise Exception('LaTex table with differences between divergence and no divergence requires that both folders exist.')
        return None
    else:
        dv_folder = '{}/{}'.format(input_folder,dv_folder[0])
        ndv_folder = '{}/{}'.format(input_folder,ndv_folder[0])
        
        # Gets the number of OPF ensembles
        ensembles = os.listdir(dv_folder)
        n_ensembles = len(ensembles)

        # Builds the string that stands for the LaTex table
        table='\\begin{table} \n'
        table+='\\centering \n'
        table+='\\caption{{Difference between divergence and no divergence considering the {} metric.}} \n'.format(metric)
        table+='\\label{t.difference_divergence_nodivergence} \n'
        table+='\\begin{adjustbox}{width=1\\textwidth}{\\def\\arraystretch{1.5}\\tabcolsep=12pt \n'
        table+='\\begin{}{} \n'.format('{tabular}','{||l'+'||c'*(n_ensembles*2)+'||}' )        

        # Iterates over each number of ensemble's classifiers
        for n in n_classifiers:
            # Gets the lists of datasets
            datasets1 = os.listdir('{}/{}'.format(dv_folder,os.listdir(dv_folder)[0]))
            datasets2 = os.listdir('{}/{}'.format(ndv_folder,os.listdir(ndv_folder)[0]))

            # Forms the row for n classifiers
            table+='\\hhline{|'+'='*(n_ensembles*2+1)+'|} \n'
            table+='\\multicolumn{}{}{}\n'.format('{'+str(n_ensembles*2+1)+'}','{||c||}','{}'.format('{\\textbf{'+str(n)+' classifiers}}\\\\\n'))
            table+='\\hhline{|'+'='*(n_ensembles*2+1)+'|} \n'
            table+=' ' +' '.join(['  &  \\multicolumn{}{}{}'.format('{2}','{c||}','{'+e.replace('_',' ').capitalize()+'}') for e in ensembles])+'\\\\ \n'
            table+='\\hhline{|'+'='*(n_ensembles*2+1)+'|} \n'
            table+=' ' +' '.join(['  &  DV  &   NDV  ' for e in ensembles])+'\\\\ \n'
            table+='\\hhline{|'+'='*(n_ensembles*2+1)+'|} \n'        

            # Iterates over each dataset folder
            for (d1,d2) in zip(datasets1,datasets2):
                table+='{} '   .format(d1.replace('_',' ').capitalize())

                # Gets the pruning methods of both folders, i.e., divergence and no divergence
                for (pr1,pr2) in zip(os.listdir(dv_folder),os.listdir(ndv_folder)):

                    folds1 = '{}/{}/{}'.format(dv_folder,pr1,d1)
                    folds2 = '{}/{}/{}'.format(ndv_folder,pr2,d2)
                    #print('Dataset {}   pruning {} '.format(d1,pr1)) 

                    list_d1 = []
                    list_d2 = []                  

                    # Iterates over each fold
                    for (f1,f2) in zip(os.listdir(folds1),os.listdir(folds2)):
                        # Loads the results file
                        results1 = np.loadtxt('{}/{}/{}/results.txt'.format(folds1,f1,n))
                        results2 = np.loadtxt('{}/{}/{}/results.txt'.format(folds2,f2,n))

                        if (metric=='accuracy'):
                            list_d1.append(results1[0])
                            list_d2.append(results2[0])
                        else:
                            list_d1.append(results1[1])
                            list_d2.append(results2[1])
                    
                    # Computes the average
                    avg1 = np.mean(list_d1)
                    avg2 = np.mean(list_d2)

                    # Computes the statistics according to Wilcoxon
                    st = wilcoxon(list_d1,list_d2,zero_method='zsplit')

                    # Highlights the highest average value
                    if(avg1 > avg2):
                        if (st.pvalue < 0.05):
                            table+='&  \\underline{{\\textbf{{{}}}}}  &   {}  '.format('{:.4f}'.format(avg1),'{:.4f}'.format(avg2))
                        else:
                            table+='&  \\textbf{{{}}}  &   {}  '.format('{:.4f}'.format(avg1),'{:.4f}'.format(avg2))
                    else:
                        if (st.pvalue < 0.05):
                            table+='&  {}  &   \\textbf{{{}}}  '.format('{:.4f}'.format(avg1),'{:.4f}'.format(avg2))
                        else:
                            table+='&  {}  &   \\underline{{\\textbf{{{}}}}}  '.format('{:.4f}'.format(avg1),'{:.4f}'.format(avg2))

                table+='\\\\ \n'
        
        table+='\\hhline{|'+'='*(n_ensembles*2+1)+'|} \n'
        table+='\\end{tabular}}\n'
        table+='\\end{adjustbox}\n'
        table+='\\end{table}'

        return table

# Main function
if __name__ == '__main__':
    
    args = get_args()

    # Results folder
    results_folder = args.input_folder

    # Output folder for LaTex tables
    output_folder = args.output_folder

    # Indicates the OPF with or without divergence
    opf_div = args.divergence_mode    

    # Getting the list of all methods
    methods = os.listdir(results_folder)

    # Number of baseline classifiers
    n_classifiers = [10,30,50]

    # OPFsemble variantes
    opf_variants = ['intracluster','mode']

    # Dictionary to store the mean metrics for all datasets and ensembles
    results_ensemble = defaultdict(dict)

    for m in methods:
        if (not m == 'baseline'):
            # Check if the method is a OPF variant
            if (m.startswith('opf')):
                if (not '{}_{}'.format('opf',opf_div) == m):
                    continue

                print(m)
                #opf_variants = os.listdir('{}/{}'.format(results_folder,m))
                
                for v in opf_variants:
                    ds = os.listdir('{}/{}/{}'.format(results_folder,m,v))

                    for d in ds:
                        mean_metrics = load_ensemble_results('{}/{}/{}/{}'.format(results_folder,m,v,d),n_classifiers)
                        results_ensemble[d.replace('_',' ')]['{} {}'.format('OPF',v.replace('_',' '))] = mean_metrics
            else:
                ds = os.listdir('{}/{}'.format(results_folder,m))
                for d in ds:
                    mean_metrics = load_ensemble_results('{}/{}/{}'.format(results_folder,m,d),n_classifiers)
                    results_ensemble[d.replace('_',' ')]['Stacking'] = mean_metrics

    # Gets the LaTex tables
    table_acc = latex_table_ensembles_metrics(results_ensemble,n_classifiers,'accuracy')
    table_f1 = latex_table_ensembles_metrics(results_ensemble,n_classifiers,'f1')
    table_time = latex_table_ensemble_time(results_ensemble,n_classifiers)
    # 
    table_acc_dv_ndv = latex_table_difference_divergence(results_folder,n_classifiers,'{}_{}'.format('opf',opf_div),'accuracy')
    table_f1_dv_ndv = latex_table_difference_divergence(results_folder,n_classifiers,'{}_{}'.format('opf',opf_div),'f1')

    # Creates the output folder if it does not exists
    divergence_folder = '{}/{}'.format(output_folder,opf_div)
    if (not os.path.exists(divergence_folder)):
        os.makedirs(divergence_folder)
    
    # Saving the LaTex tables
    with open('{}/table_acc.txt'.format(divergence_folder),'w') as f:
        f.write(table_acc)
    
    with open('{}/table_f1.txt'.format(divergence_folder),'w') as f:
        f.write(table_f1)

    with open('{}/table_time.txt'.format(divergence_folder),'w') as f:
        f.write(table_time)
    
    if (not table_acc_dv_ndv is None and not table_f1_dv_ndv is None):
        with open('{}/table_acc_dv_vs_ndv.txt'.format(divergence_folder),'w') as f:
            f.write(table_acc_dv_ndv) 

        with open('{}/table_f1_dv_vs_ndv.txt'.format(divergence_folder),'w') as f:
            f.write(table_f1_dv_ndv)
    else:
        print('It wasn\'t possible to build the LaTex table with the difference between divergence and no divergence.')