import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LeakyReLU

Condition_inputs=pd.read_csv('Phi_T_Sampling_uniform.csv')

name_11='State_space_cte_pressure_T' 
name_12='_phi_'
name_21='Reaction_rates_cte_pressure_T'
name_22='_phi_'

end='.csv'

def hydrogen_data_clean_shift_sandiego_cantera(cantera_species,cantera_sources,maximum_values):
    cantera_sources=cantera_sources.add_suffix('w')
    cantera_sources=cantera_sources.iloc[:,1:] #for taking out the timestep as data 
    
    cantera_time=cantera_species.iloc[:,1]
    cantera_temperature=cantera_species.iloc[:,2]
    cantera_pressure=cantera_species.iloc[:,3]
    
    cantera_species_fractions=cantera_species.iloc[:,4:12]
    
    n_columns_mass_fraction=np.shape(cantera_species_fractions)[1]
    
    cantera_sources=cantera_sources.loc[:,(cantera_sources!=0).any(axis=0)]
    cantera_sources=cantera_sources.loc[:, (cantera_sources != cantera_sources.iloc[0]).any()]
    
    n_columns_source=np.shape(cantera_sources)[1]
    
    cantera_data=pd.concat([cantera_time, cantera_temperature,cantera_species_fractions,cantera_sources],axis=1)

    maximum_values=maximum_values.iloc[0,1:]
    maximum_values=maximum_values.to_numpy()
    
    iterations=np.shape(cantera_data)[1]
    
    #cantera_data.divide(maximum_values)
    for j in range(iterations):
        cantera_data.iloc[:,j]=cantera_data.iloc[:,j]/(maximum_values[j])
    
    cantera_data_shift=cantera_data.loc[1:,:]
    cantera_data_shift=cantera_data_shift.add_suffix('shift')
    
    cantera_data=cantera_data.reset_index()
    cantera_data_shift=cantera_data_shift.reset_index()
    
    cantera_data=cantera_data.iloc[:,1:]
    cantera_data_shift=cantera_data_shift.iloc[:,1:]

    cantera_data=cantera_data.iloc[0:(np.shape(cantera_data_shift)[0]),:]
    
    data_all=pd.concat([cantera_data, cantera_data_shift], axis=1)

    columns=data_all.columns.to_list()
    
    return data_all, n_columns_source, n_columns_mass_fraction, columns

iterations=np.shape(Condition_inputs)[0]
for i in range(iterations):
    cantera_species=pd.read_csv(name_11+str(round(Condition_inputs.iloc[i,2],3))+name_12+str(round(Condition_inputs.iloc[i,1],3))+end)
    cantera_species=pd.DataFrame(cantera_species)
    
    cantera_sources=pd.read_csv(name_21+str(round(Condition_inputs.iloc[i,2],3))+name_22+str(round(Condition_inputs.iloc[i,1],3))+end)
    cantera_sources=pd.DataFrame(cantera_sources)
    
    maximum_values=pd.read_csv('maximum_values_Phi_T_Sampling.csv')
    maximum_values=pd.DataFrame(maximum_values)
    
    dataset, n_columns_source, n_columns_mass_fraction, columns=hydrogen_data_clean_shift_sandiego_cantera(cantera_species,cantera_sources,maximum_values)
    
    if i==0:
        n_samples=np.shape(dataset)[0]
        dataset_copy=dataset
    else:
        dataset_copy=pd.concat([dataset_copy,dataset],axis=0)
seed=57
tf.random.set_seed(seed)
np.random.seed(seed)

alpha=0.001

dropout_p=0 #dropout layer percentage

initializer = tf.keras.initializers.GlorotNormal()

initializer_b='zeros'

input_size=n_columns_mass_fraction+1 #the number of mass fractions plus one more column for the temperature

output_size=input_size #we want the same shape of the input vector 

reduced_size=4 #this is just for starting, we probably will change it later 

architecture_e=[70]

architecture_d=architecture_e[::-1]

class NN(Model): #The part inside the parenthesis is a standard syntaxis function for defining the class object
    
    def __init__(self):
        super(NN,self).__init__(input_size,output_size,reduced_size,dropout_p,initializer,initializer_b, architecture_e,architecture_d, alpha)
        
        #inputs=tf.keras.Input(shape=(input_size,None))
        
        self.encoder=Sequential()
        self.encoder.add(tf.keras.Input(shape=(input_size)))
        
        for i in range(len(architecture_e)):
            self.encoder.add(Dense(units=architecture_e[i], activation='linear',kernel_initializer=initializer, use_bias=True,bias_initializer=initializer_b))
            self.encoder.add(Dropout(dropout_p))
        #the for cycle describes the general structure 
        self.encoder.add(Dense(units=reduced_size)) #reduced size output
        
        self.lat_activation=Sequential()
        self.lat_activation.add(tf.keras.layers.Activation('linear')) #lat activation function, just because it appears in the paper
        
        
        self.decoder=Sequential()
        self.decoder.add(tf.keras.Input(shape=(reduced_size)))
        
        for i in range(len(architecture_d)):
            self.decoder.add(Dense(units=architecture_d[i], activation='linear',kernel_initializer=initializer, use_bias=True,bias_initializer=initializer_b))
            self.decoder.add(Dropout(dropout_p))
       
        self.decoder.add(Dense(units=output_size))
        
    def call(self,x):
        encoded=self.encoder(x)
        mid=self.lat_activation(encoded)
        output=self.decoder(mid)
        return output

optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

def loss_MSE(y_true, y_pred):
        
    # MSE loss function
    y_true=tf.cast(y_true, dtype=tf.float32)
    y_pred=tf.cast(y_pred, dtype=tf.float32)
    
    mse = tf.reduce_mean(tf.square(y_true-y_pred))
    
    #std=transformation_values[1,output_start:output_end]
    #mean=transformation_values[0,output_start:output_end]
    max_val=maximum_values.iloc[0:,2:3+n_columns_mass_fraction].to_numpy()
    
    # rescaling the data 
    y_pred = tf.convert_to_tensor(y_pred*max_val) # or the form you scaling it
    y_true= tf.convert_to_tensor(y_true*max_val)
    # Regularization: the sum of mass fractions equals 1
    reg = tf.square(tf.reduce_sum(y_true[:,1:]) -  tf.reduce_sum(y_pred[:,1:])) 
        
    # custom loss
    loss = mse 
    return loss

#output_start=3+n_columns_mass_fraction+n_columns_source
#output_end=output_start+n_columns_mass_fraction+1

Inputs=dataset_copy.iloc[:,1:2+n_columns_mass_fraction].to_numpy()
Outputs=dataset_copy.iloc[:,1:2+n_columns_mass_fraction].to_numpy()

monitors=['loss','accuracy']

repetitions=range(7)

import os
from pathlib import Path
from os.path import join
from os.path import isfile
from pathlib import Path

header=columns[1:2+n_columns_mass_fraction]
header=header+columns[1:2+n_columns_mass_fraction]

header2=monitors

for i in range(len(monitors)):
    early_stopping=tf.keras.callbacks.EarlyStopping(monitor=monitors[i],
                   min_delta=0.0001,
                   patience=5,
                   restore_best_weights=True)
    print(monitors[i])
    os.mkdir(monitors[i])
    
    from sklearn.model_selection import RepeatedKFold

    for j in range(len(repetitions)):
        n_split=5
        n_repeats=repetitions[j]+1
        counter=0
        total_c=n_split*n_repeats
        
        directory_1=monitors[i]
        directory_2='repeats_'+str(repetitions[j]+1)
        
        address=join(directory_1,directory_2)
        os.mkdir(address)

        for train_index,test_index in RepeatedKFold(n_splits=n_split, n_repeats=n_repeats,random_state=42).split(Inputs):
            x_train,x_test=Inputs[train_index],Inputs[test_index]
            y_train,y_test=Outputs[train_index],Outputs[test_index]
    
            Autoencoder=NN()
            Autoencoder.compile(optimizer=optimizer, loss=loss_MSE, run_eagerly=True, metrics = ['accuracy'])

            Autoencoder.fit(x_train, y_train,batch_size=64,
                        epochs=2000,
                        callbacks=[early_stopping],
                        validation_data=(x_test, y_test),
                        verbose=0)
        
            print('Model evaluation ',Autoencoder.evaluate(x_test,y_test))
            
        
        
            name1=join(address,'train'+str(counter)+'.csv')
            name2=join(address,'test'+str(counter)+'.csv')
            name3=join(address,'evaluate'+str(counter)+'.csv')
            
            evaluate=Autoencoder.evaluate(x_train,y_train)
            evaluate=np.asarray(evaluate)
            print(evaluate)
            evaluate=pd.DataFrame(evaluate)
            evaluate=evaluate.transpose()
            print(evaluate)
            
            x_train=pd.DataFrame(x_train)
            y_train=pd.DataFrame(y_train)
            
            x_test=pd.DataFrame(x_test)
            y_test=pd.DataFrame(y_test)

            Train=pd.concat([x_train, y_train],axis=1)
            Test=pd.concat([x_test,y_test],axis=1)
        
            Train.to_csv(name1, header=header)
            Test.to_csv(name2, header=header)
            evaluate.to_csv(name3, header=header2)
            
            counter=counter+1
        
        name=join(address,f'k_fold_best_model_new_architecture_test{reduced_size}')
        target=Path(address)
        
        condition=Condition_inputs.sample()
        condition
        T=condition.iloc[0,2] #temperature to be checked 
        Phi=condition.iloc[0,1] #equivalence ratio to be checked
        print(T)
        print(Phi)
        
        cantera_species=pd.read_csv(name_11+str(round(T,3))+name_12+str(round(Phi,3))+end)
        cantera_species=pd.DataFrame(cantera_species)
        
        cantera_sources=pd.read_csv(name_21+str(round(T,3))+name_22+str(round(Phi,3))+end)
        cantera_sources=pd.DataFrame(cantera_sources)
        
        maximum_values=pd.read_csv('maximum_values_Phi_T_Sampling.csv')
        maximum_values=pd.DataFrame(maximum_values)
        
        dataset, n_columns_source, n_columns_mass_fraction, columns=hydrogen_data_clean_shift_sandiego_cantera(cantera_species,cantera_sources,maximum_values)
    
    
        results=Autoencoder.decoder(Autoencoder.lat_activation(Autoencoder.encoder(dataset.iloc[:,1:2+n_columns_mass_fraction].to_numpy()))).numpy()
    
        Autoencoder.save(name)
    
        interest_vector=['H2O','O2','H2','T[K]','OH','HO2','H2O2']
        
        t_index=columns.index('t[s]shift')
        t_trans=maximum_values.columns.get_loc('t[s]')
        time_plot=(dataset.iloc[:,t_index])*maximum_values.iloc[0,t_trans]
        
        for k in range(len(interest_vector)):
            original_index=columns.index(interest_vector[k])
            #print(columns[original_index])
            results_index=columns.index(interest_vector[k]) #minues one due to the time column presence 
            #print(columns[results_index])
            #print(results_index-1)
            transformation_index=maximum_values.columns.get_loc(interest_vector[k])
            plot_name=interest_vector[k]+'.png'
            
            input_label=interest_vector[k]+' Dataset'
            output_label=interest_vector[k]+' Reconstruction'
            
            original=(dataset.iloc[:,original_index]).to_numpy()
            original=original*maximum_values.iloc[0,transformation_index]
    
            output=(results[:,results_index-1])
            output=output*maximum_values.iloc[0,transformation_index]
            
            fig_name=join(address,interest_vector[k])
            
            plt.figure(k)
            plt.scatter(time_plot,original, label=input_label)
            plt.scatter(time_plot,output, label=output_label)
            plt.title(interest_vector[k]+' plot results')
            plt.xlabel('Time [S]')
            plt.ylabel(interest_vector[k])
            plt.legend()
            plt.savefig(fig_name)
            plt.clf()
