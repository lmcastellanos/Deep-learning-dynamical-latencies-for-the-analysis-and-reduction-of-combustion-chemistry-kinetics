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

reduced_size=3 #this is just for starting, we probably will change it later 

ataset_copy2=dataset_copy
dataset_copy=dataset_copy.sample(frac=1, random_state=seed)
dataset_copy=dataset_copy.sample(frac=0.75, random_state=seed)

n_val = int(len(dataset_copy)*.8)
training=dataset_copy.iloc[0:n_val,:]
validation=dataset_copy.iloc[n_val:,:]

output_start=3+n_columns_mass_fraction+n_columns_source
output_end=output_start+n_columns_mass_fraction+1
training_fit_out=training.iloc[:,output_start:output_end]

training_fit_in=training.iloc[:,1:2+n_columns_mass_fraction]

training_fit_out=training.iloc[:,1:2+n_columns_mass_fraction].to_numpy()
training_fit_in=training.iloc[:,1:2+n_columns_mass_fraction].to_numpy()

validation_fit_out=validation.iloc[:,1:2+n_columns_mass_fraction].to_numpy()
validation_fit_in=validation.iloc[:,1:2+n_columns_mass_fraction].to_numpy()

def create_encoder(hp): 
    
    #hp doesn't need to be imported before it is used in architecture definition
    
    encoding_units=[] #vector for allocating the number of neurones per layer
    
    #encoding_dp=[] #vector for allocating the amount of dropout per layer 
    
    
    inputs=tf.keras.Input(shape=(input_size))
    x=inputs
    #x will be the variable that enters the autoencoder
    
    for j in range(num_layers):
        
        #with the argument in range we are defining a tunning hyperparameter which is the number of layers
        
        u=hp.Int(f'encoding_units_{j}',25,350,step=1)
        #this line if for defining the number of neurones per layer as variable
        #the letter f befor the variable name allows to allocate j as variable
        
        encoding_units.append(u)
        #for allocating the value in the vector
        
        x=Dense(u,kernel_initializer=initializer, use_bias=True,bias_initializer=initializer_b, activation='linear')(x)
        #x=tf.keras.layers.LeakyReLU(alpha)(x)
        #selection of alpha activation function
        
        #dp=hp.Float(f'encoding_dp{j}',0,0.5)
        #for having a dropout values sampling from 0 to 0.5
        #encoding_dp.append(dp)
        
        #x=Dropout(dp)(x)
    
    x=Dense(units=reduced_size,activation='linear',kernel_initializer=initializer, use_bias=True,bias_initializer=initializer_b)(x)
    #imposing the desired reduced size
    
    for u in encoding_units[::-1]:
        
        #with this we use zip for a double iteration, we are creating a tuple
        #which will have paired the values  of units and dropout per layer
        
        x=Dense(u,kernel_initializer=initializer, use_bias=True,bias_initializer=initializer_b,activation='linear')(x)
        #x=tf.keras.layers.LeakyReLU(alpha)(x)
        #x=Dropout(dp)(x)
    
    outputs=Dense(output_size,kernel_initializer=initializer, use_bias=True,bias_initializer=initializer_b)(x)
    
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    error_f=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    
    
    autoencoder = tf.keras.Model(inputs,outputs)
    
    autoencoder.compile(optimizer=optimizer, loss=error_f, metrics=['accuracy'], run_eagerly=True)
    
    return autoencoder 

import keras_tuner as kt
from pathlib import Path
from os.path import join
from os.path import isfile
from pathlib import Path

directory=f"H2_PCA_optimization_{reduced_size}"

num_layers=1
    
autoencoder=create_encoder(kt.HyperParameters())
    
project_name=f'random_search_layers_{num_layers}'
    
tuner = kt.RandomSearch(
hypermodel=create_encoder,
objective="accuracy",
max_trials=10,
seed=None,
hyperparameters=None,
tune_new_entries=True,
allow_new_entries=True,
overwrite=True,
directory=directory,
project_name=project_name,
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=1e-9)
    
models=tuner.search(training_fit_in, training_fit_out, epochs=500, validation_data=(validation_fit_in, validation_fit_out))

models= tuner.get_best_models(num_models=5)

for k in range(5):
    best_model=models[k]
    model_name='model'+str(k+1)
    
    address=join(directory,model_name)
    target=Path(address)
    
    print(best_model.summary())
    best_model.save(target)

validation=pd.DataFrame(validation)
training=pd.DataFrame(training)

address=join(directory,'training.csv')
target=Path(address)

training.to_csv(target)

address=join(directory,'validation.csv')
target=Path(address)

best_model= tuner.get_best_models(num_models=1)
    
best_hps = tuner.get_best_hyperparameters(1)
model = create_encoder(best_hps[0])

validation.to_csv(target)

best_model= tuner.get_best_models(num_models=1)
    
best_hps = tuner.get_best_hyperparameters(1)
model = create_encoder(best_hps[0])

early_stopping=tf.keras.callbacks.EarlyStopping(
                   min_delta=0.0001,
                   patience=5,
                   restore_best_weights=True)

history = model.fit(training_fit_in, training_fit_out,
                validation_data=(validation_fit_in, validation_fit_out),
                batch_size=64,
                epochs=2000,
                callbacks=[early_stopping],
                verbose=0 #turns off training log 
                )
history_df = pd.DataFrame(history.history)

plt.figure()
plt.plot(history_df['loss'],label='training loss')
plt.plot(history_df['val_loss'], label='validation loss')
plt.title('Error plot results')
plt.xlabel('N Epoch')
plt.ylabel('Error')
plt.legend()

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

interest_vector=['H2O','O2','H2','T[K]','OH','HO2','H2O2']

t_index=columns.index('t[s]shift')
t_trans=maximum_values.columns.get_loc('t[s]')
time_plot=(dataset.iloc[:,t_index])*maximum_values.iloc[0,t_trans]
    
for k in range(len(interest_vector)):
    original_index=columns.index(interest_vector[k]+'shift')
    #print(columns[original_index])
    results_index=columns.index(interest_vector[k]) #minues one due to the time column presence 
    #print(columns[results_index])
    #print(results_index-1)
    transformation_index=maximum_values.columns.get_loc(interest_vector[k])
    plot_name=interest_vector[k]+'.png'
    
    input_label=interest_vector[k]+' Dataset'
    output_label=interest_vector[k]+' Reconstruction'
    
    original=(dataset.iloc[:,original_index]).to_numpy()
    #original=original*transformation_values[1,original_index]
    #original=original+transformation_values[0,original_index]
    #original=np.exp(original)-1
    original=original*maximum_values.iloc[0,transformation_index]
    
    output=(results[:,results_index-1])
    #output=output*transformation_values[1,results_index]
    #output=output+transformation_values[0,results_index]
    #output=np.exp(output)-1
    output=output*maximum_values.iloc[0,transformation_index]
    
    plt.figure(k)
    plt.scatter(time_plot,original, label=input_label)
    plt.scatter(time_plot,output, label=output_label)
    plt.title(interest_vector[k]+' plot results')
    plt.xlabel('Time [S]')
    plt.ylabel(interest_vector[k])
    plt.legend()