import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

import gcsfs
fs = gcsfs.GCSFileSystem()
import os
import cftime

def load_vars_xarray(input_vars, output_vars, downsample=True, chunks = True):
    # raw files, not interpolated according to Yu suggestion
    if(chunks):
        mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.input.zarr')
        inp = xr.open_dataset(mapper, engine='zarr', chunks={'sample' : 720})
        mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.output.zarr')
        output = xr.open_dataset(mapper, engine='zarr', chunks={'sample' : 720})
    else:
        mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.input.zarr')
        inp = xr.open_dataset(mapper, engine='zarr')
        mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.output.zarr')
        output = xr.open_dataset(mapper, engine='zarr')
        
    ds = inp[input_vars]
    for var in output_vars:
        ds['out_'+var] = output[var]
        
    if downsample: # might as well do first
        N_samples = len(inp.sample)
        inp = inp.isel(sample = np.arange(36,N_samples,72)) #  every 1 day
        output = output.isel(sample = np.arange(36,N_samples,72))
        
        if(chunks): # can afford to do?
            print("Daily average")
            ds = ds.coarsen(sample = 72).mean()
        else:
            print("Noon each day")
            ds = ds.isel(sample = np.arange(36,N_samples,72))
            
    # reformat, add time dimension
    time = pd.DataFrame({"ymd":inp.ymd, "tod":inp.tod})
    # rename sample to reformatted time column 
    f = lambda ymd, tod : cftime.DatetimeNoLeap(ymd//10000, ymd%10000//100, ymd%10000%100, tod // 3600, tod%3600 // 60)
    time = time.apply(lambda x: f(x.ymd, x.tod), axis=1)
    ds['sample'] = list(time)
    ds = ds.rename({'sample':'time'})
    ds = ds.assign_coords({'ncol' : ds.ncol})
    
    # Load spatial latlon info
    mapper = fs.get_mapper("gs://leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.grid-info.zarr")
    ds_grid = xr.open_dataset(mapper, engine='zarr')
    lat = ds_grid.lat.values.round(2) 
    lon = ds_grid.lon.values.round(2)  
    lon = ((lon + 180) % 360) - 180 # convert from 0-360 to -180 to 180
    
    ds['lat'] = (('ncol'),lat.T)
    ds['lon'] = (('ncol'),lon.T)
    
    ds = ds.assign_coords({'lat' : ds.lat, 'lon' : ds.lon})
    
    return(ds)


def load_model(name, baseDir = 'saved_data/models', load_params=True):
    path = os.path.join(baseDir, name)
    params = json.load(open(path + '.json'))
    model_params = params['model parameters']
    architecture = model_params['architecture']
    match architecture.lower():
        case "ved":
            model = VariationalEncoderDecoder(
                model_params['beta'], model_params['data_dims'], model_params['label_dims'], model_params['latent_dims'], 
                model_params['input_layers'], model_params['output_layers'], model_params['dropout'], model_params['device']
            )
        case "vae":
            print("VAE loading not implemented")
        case "cvae":
            model = ConditionalVAE(model_params['beta'], model_params['data_dims'], model_params['label_dims'], dropout=model_params['dropout'],
                 latent_dims=model_params['latent_dims'], hidden_dims=model_params['hidden_dims'], layers=model_params['layers']).to(model_params['device'])
        case _:
            print("Unknown architecture")
    if(load_params): # load not just architecture but weights. Do unless training went terribly
        model.load_state_dict(torch.load(path + params['save parameters']['filetype']))
    model.eval()

    return(model, params['training_parameters'])


def split_input_output(ds):
    inp = []
    out = []
    for var in ds.data_vars:
        if(var[:3] == 'out'):
            out.append(var)
        else:
            inp.append(var)
    return(ds[inp], ds[out])

def get_item(ds, index):
    # t * ds.ncol.size + col == i
    # given an index, wrap around (time x ncol) grid selecting specific variable
    # converting linear indexing into structured
    assert index < ds.time.size * ds.ncol.size, "Index is outside of range"
    t, col = index // ds.ncol.size, index % ds.ncol.size
    return(ds.isel(time=t, ncol=col))

def get_batch(ds, batch_num, batch_size = 32, dim = 'ncol'):
    #same kind of linear index interpretation, except over a batch size
    # doing over ncol because 384 = 3 * 2**7, which splits nicely over powers of 2
    n_batch = ds[dim].size / batch_size
    other_dim_batch, dim_batch = int(batch_num // n_batch), batch_num % n_batch
    start, stop = int(dim_batch * batch_size), int((dim_batch+1) * batch_size)
    if(dim == 'ncol'):
        print(f"ncol from {start}-{stop}; time={other_dim_batch}")
        return(ds.isel(ncol=slice(start, stop), time=other_dim_batch))
    elif(dim=='time'):
        return(ds.isel(time=slice(start, stop), ncol=other_dim_batch))
        
        
def list_all_vars():
    mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.input.zarr')
    ds = xr.open_dataset(mapper, engine='zarr')
    all_input_vars = list(ds.data_vars)[:-2]
    
    mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.output.zarr')
    ds = xr.open_dataset(mapper, engine='zarr')
    all_output_vars = list(ds.data_vars)[:-2]
    return(all_input_vars, all_output_vars)


def split_vars_by_leveled(ds, var_list, out=False):
    v = []
    leveled = []
    for var in var_list:
        if out:
            var = 'out_' + var
        if(len(ds[var].shape) > 2):
            leveled.append(var)
        else:
            v.append(var)
    return(v, leveled)