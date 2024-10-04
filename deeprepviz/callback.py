from copy import deepcopy
from einops import rearrange
import json
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import os
from os.path import join
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class DeepRepViz(Callback):
    def __init__(self, dataloader_class,
                 dataset_kwargs,
                 datasets_kwargs_test={},
                 hook_layer=-1,
                 best_ckpt_by=None, best_ckpt_metric_should_be='min',
                 verbose=0):
        '''
        dataloader_class(**dataloader_kwargs) should return all data points as X,y on which we want
        to generate DeepRepViz logs.
        expected_IDs: the IDs of the subjects in the dataset in the order that they are expected
        to be returned by the dataloader.
        expected_labels: the labels of the subjects in the dataset in the order that they are expected
        to be returned by the dataloader.
        '''
        self.verbose = verbose

        ### prepare the train dataloader
        dataset_kwargs = self._check_dataset_kwargs(dataset_kwargs, split='train')

        # convert the dataloader to a custom DeepRepViz dataloader that returns X,y AND the subject ID
        self.dataloader = create_dataloader(dataloader_class,
                                            dataloader_kwargs=dataset_kwargs["dataloader_kwargs"],
                                            expected_IDs=dataset_kwargs["expected_IDs"],
                                            expected_labels=dataset_kwargs["expected_labels"])

        # also save all the provided data IDs,labels, splits in a table to store in the deeprepvizlog folder later
        df_data = pd.DataFrame({"IDs"      : dataset_kwargs["expected_IDs"], # TODO can exclude this neccesity to provide expected_IDs and expected_labels
                                "labels"   : dataset_kwargs["expected_labels"], # TODO can exclude this neccesity to provide expected_IDs and expected_labels
                                "datasplit": dataset_kwargs["datasplit"]
                                })

        ### prepare the test dataloaders if provided
        self.dataloaders_test = {}
        dfs_data_test = []

        # if no test datasets are provided then throw a warning that the metrics will be reported on the train data
        if len(datasets_kwargs_test)==0:
            warnings.warn("No test datasets provided. \
The metrics reported in DeepRepViz will be computed on the train data.", stacklevel=2)
        else:
            for split_name, dataset_kwargs_test in datasets_kwargs_test.items():
                assert split_name != "train", "The split name 'train' is reserved for the train dataset. Please provide a different split name for the test dataset."
                dataset_kwargs_test = self._check_dataset_kwargs(dataset_kwargs_test, split=split_name)
                dataloader_test = create_dataloader(dataloader_class,
                                                dataloader_kwargs=dataset_kwargs_test["dataloader_kwargs"],
                                                expected_IDs=dataset_kwargs_test["expected_IDs"],
                                                expected_labels=dataset_kwargs_test["expected_labels"])
                self.dataloaders_test.update({split_name: dataloader_test})
                # save all the provided data IDs,labels, splits in a table to store in the deeprepvizlog folder later
                df_data_test = pd.DataFrame({"IDs"    : dataset_kwargs_test["expected_IDs"],
                                             "labels"   : dataset_kwargs_test["expected_labels"],
                                             "datasplit": dataset_kwargs_test["datasplit"]
                                             })
                dfs_data_test.append(df_data_test)

            # save all the provided data IDs,labels, splits in a table to store in the deeprepvizlog folder later
            df_data = pd.concat([df_data]+dfs_data_test)
            self.df_data = df_data.set_index("IDs")


        # Save the IDs and labels in the global log
        self.deeprepvizlog = {**df_data.to_dict(orient='list'),
                              "checkpoints" : []}
        # store all DeepRepViz results in a dict for each checkpoint
        self.hook_layer = hook_layer

        self.hooked_layer = None
        self._reset_hooked_vals()
        self.verbose = verbose
        self.ckpts_dir = None
        self.ckpt_idx = 0

        self.best_ckpt_by = best_ckpt_by
        self.best_ckpt_metric_should_be = best_ckpt_metric_should_be
        assert self.best_ckpt_metric_should_be in ['min', 'max'], "best_ckpt_metric_should_be should be either 'min' or 'max'"
        self.previous_best_ckpt_score = 0 if self.best_ckpt_metric_should_be=='max' else np.inf


    def _check_dataset_kwargs(self, dataset_kwargs, split='train'):
        # check all necessary keys are provided
        for k in ["dataloader_kwargs","expected_IDs", "expected_labels"]:
            assert k in dataset_kwargs, f"{k} should be provided for the train dataset in dataset_kwargs"
        assert len(dataset_kwargs["expected_IDs"]) == len(dataset_kwargs["expected_labels"]), "\
expected_IDs and expected_labels should have the same length"
        # prepare the datasplit names
        dataset_kwargs["datasplit"] = len(dataset_kwargs["expected_IDs"])*[split]

        return dataset_kwargs

################################    hook functions    ######################################

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        '''everytime a check point is called (either by a ModelCheckpoint callback or the EarlyStopping callback) run the DeepRepViz inference and save states in the same folder.'''
        if self.verbose: print('[DeepRepViz] Creating snapshot at', 'epoch=',checkpoint['epoch'], 'global_step=',checkpoint['global_step'])
        ckpt_name = "epoch{:02d}-step{:06d}".format(checkpoint['epoch'],checkpoint['global_step'])
        model_ckpt_callbacks = [checkpoint['callbacks'][k] for k in checkpoint['callbacks'] if 'ModelCheckpoint' in k]
        assert len(model_ckpt_callbacks)>0, "DeepRepViz only works when the ModelCheckpoint callback is enabled. Ensure that `enable_checkpointing=True` when calling the pytorch lightning Trainer()."

        # use the the same folder to store DeepRepViz logs as the ModelCheckpoint
        if self.ckpts_dir is None:
            self.ckpts_dir = model_ckpt_callbacks[0]['dirpath'].replace('checkpoints', 'deeprepvizlog')
            # create the folder if it doesn't exist
            if not os.path.isdir(self.ckpts_dir):
                os.makedirs(self.ckpts_dir)
            # when creating the logdir also save the data IDs and labels in a global csv file in the same folder
            self.df_data.to_csv(f'{self.ckpts_dir}/data.csv')

        # create a copy of the trainer and model not to interfere
        # with the main training and remove any callbacks and loggers for this
        model_hooked = deepcopy(pl_module)
        model_hooked = self.apply_hook(model=model_hooked,
                                       hook_layer=self.hook_layer)

        trainer_copy = deepcopy(trainer)
        trainer_copy.callbacks = []
        trainer_copy.logger = None
        # create a dataloader with shuffle off
        dataloader = DataLoader(
                    dataset=self.dataloader,
                    batch_size=trainer.val_dataloaders.batch_size,
                    num_workers=trainer.val_dataloaders.num_workers,
                    shuffle=False, drop_last=False)

        dataloaders_test = []
        dataset_splits = ['train']
        for test_data_name, dataloader_test in self.dataloaders_test.items():
            dataset_splits.append(test_data_name)
            dataloaders_test.append(DataLoader(
                    dataset=dataloader_test,
                    batch_size=trainer.val_dataloaders.batch_size,
                    num_workers=trainer.val_dataloaders.num_workers,
                    shuffle=False, drop_last=False))
        # turn on the DeepRepViz hook first
        # generate predictions over all training samples in the provided dataloader
        predict_outputs = trainer_copy.predict(
                                    model_hooked,
                                    dataloaders=[dataloader]+dataloaders_test,
                                    ckpt_path=None)

        # join the outputs from all batches and dataloaders
        all_preds = []
        all_metrics = {}
        for i, split in enumerate(dataset_splits):
            # extract ID, label, pred, and metrics  from the output
            predict_outputs_i = predict_outputs[i]
            IDs, labels, preds, metrics_list = zip(*predict_outputs_i)
            IDs = torch.cat(IDs).numpy().astype(int)
            labels = torch.cat(labels).numpy()
            preds = torch.cat(preds).numpy()
            if preds.ndim==1: preds = np.expand_dims(preds,1)

            # sanity checks
            assert (IDs == self.df_data[self.df_data.datasplit == split].index).all(), f"IDs returned by \
    the dataloader (n={len(IDs)}) does not match with the IDs provided in the init (n={len(self.df_data[self.df_data.datasplit == 'train'])}) at {(IDs!=self.df_data.index).sum()}/{len(IDs)} instances"
            assert len(labels) == len(preds), f"len(labels)={len(labels)} != len(preds)={len(preds)}"

            all_preds.append(preds)
            # calculate the average metrics across all batches
            for k in metrics_list[0].keys():
                vals = [m[k] for m in metrics_list]
                # append the datasplit name to the metrics keys
                all_metrics[f"{k}_{split}"] = float(np.mean(vals))

        all_preds = np.concatenate(all_preds, axis=0)
        num_classes = all_preds.shape[-1]

        # only save the logs for this checkpoint if the best_ckpt_by metric has improved as expected
        if self.best_ckpt_by is not None:
            if self.best_ckpt_by not in all_metrics.keys():
                raise RuntimeError(f"Provided metric name for best_ckpt_by ('{self.best_ckpt_by}')  is not in the metrics list. \
Available metrics are {list(all_metrics.keys())}. Please provide one of these (and set the 'best_ckpt_metric_should_be' appropriately) \
to check for improvements over the checkpoints.")
            # check if this checkpoint is better than the previous best checkpoint
            ckpt_metric = all_metrics[self.best_ckpt_by]
            if ((self.best_ckpt_metric_should_be == 'min') and (ckpt_metric > self.previous_best_ckpt_score)) or \
               ((self.best_ckpt_metric_should_be == 'max') and (ckpt_metric < self.previous_best_ckpt_score)):
                    if self.verbose: print(f"[DeepRepViz] Checkpoint performance at {ckpt_name} ({self.best_ckpt_by}={ckpt_metric:.2f}) is not better \
than the previous best checkpoint score ({self.previous_best_ckpt_score:.2f}). Skipping.")
                    # delete all states saved in this hook and exit
                    self._reset_hooked_vals()
                    del model_hooked, trainer_copy
                    return
            # if not then update the previous best checkpoint score
            self.previous_best_ckpt_score = ckpt_metric

        ### collect the activations, weights and biases
        # get the activation arrays collected by the hook fn into numpy arrays
        acts = torch.stack(self.acts).numpy()
        # TODO implement models with different representation bottlenecks for each class
        # such that different activations are created for each class output
        # for only 1 class, put it explicitly in the last dim
        # if acts.ndim==2: acts = np.expand_dims(acts, axis=-1)
        # sanity check
        assert all_preds.shape[0] == acts.shape[0], f"len(all_preds) ={all_preds.shape[0]} != len(acts)={acts.shape[0]}"
        assert acts.shape[0] == len(self.df_data), f"len(acts)={acts.shape[0]} != total data points = {len(self.df_data)}"

        # transpose weights and biases to (nfeatures, nclasses) from (nclasses, nfeatures)
        weights = self.weights.numpy().T
        assert num_classes == weights.shape[-1], f"number of classes in the model according to the activations = ({num_classes}) does not match with the number of classes in the weights ({weights.shape})"
        biases = []
        if len(self.biases)>0:
            biases = self.biases.numpy().T
            assert num_classes == biases.shape[-1], f"number of classes in the model according to the activations = ({num_classes}) does not match with the number of classes in the biases ({biases.shape})"
        # reset all states obtained from the hook
        self._reset_hooked_vals()
        del model_hooked, trainer_copy
        # print(acts.shape, preds.shape, weights.shape, self.biases.shape)

        ### Save the results specific to this checkpoint
        ckpt_results = {"metrics": all_metrics,
                        "acts"   : acts}
        # save act, weights and biases for each class separately
        for i in range(num_classes):
            ckpt_results.update({f'preds_{i}' :all_preds[:,i].squeeze().tolist(),
                                 f'weights_{i}':weights[:,i].squeeze().tolist()})
            if len(biases)>0:
                ckpt_results.update({f'biases_{i}':biases[i].squeeze().tolist()})

        self.deeprepvizlog["checkpoints"] += [(ckpt_name, ckpt_results)]

        # create a metadata for the tensorboard with IDs, labels and preds
        metadata_dict = {k:v for k,v in self.deeprepvizlog.items() if k in ["IDs", "labels"]}
        metadata_dict.update({k:v for k,v in ckpt_results.items() if 'preds' in k})

        metadata, metadata_header = [], []
        for key,val in metadata_dict.items():
            assert np.array(val).ndim==1, "tensorboard accepts metadata that are 1D arrays only"
            metadata.append(val)
            metadata_header.append(key)
        # change shape of metadata to have nfeatures X datapoints as expected by tensorboard
        metadata = list(zip(*metadata))

        # get the tensorboard logger object
        tensorboard_writer = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tensorboard_writer = logger.experiment
                break
        assert tensorboard_writer is not None, "TensorboardLogger must be provided for DeepRepViz to work. \
In PyTorch Lightning you can provide multiple loggers as a list." #TODO test if this works with multiple loggers
        # start_time = time.time()
        # save activations to the logs dir using Tensorboard data structure
        tensorboard_writer.add_embedding(
                mat = ckpt_results[f'acts'],
                metadata=metadata, metadata_header= metadata_header,
                global_step = f"deeprepvizlog/{ckpt_name}",
                tag = ckpt_name,
                label_img = None)
        # print("time to write tensorboard", time.time()-start_time)
        ## TODO tensorboard is slow, main bottleneck, maybe better to switch back to h5 files
        ## TODO saving as .tsv files is also slow, maybe better to switch back to h5 files
        # old code where I saved the checkpoints as h5 files instead of tensorboard logs
        # with h5py.File(f'{self.ckpts_dir}/{ckpt_name}/{ckpt_name}/tensors.h5', 'w') as h5:
        #     for k,v in result.items():
        #         # write dicts into the attributes instead of datasets
        #         if not isinstance(v,dict):
        #             h5[k] = v
        #         else:
        #             h5.attrs[k] = str(v)

        # finally save the metrics as a yaml file
        metrics_file = f'{self.ckpts_dir}/{ckpt_name}/{ckpt_name}/metametadata.json'

        with open(metrics_file, 'w') as fp:
            out_dict = {"metrics": ckpt_results["metrics"]}
            for i in range(num_classes):
                out_dict.update({f'weights_{i}': ckpt_results[f'weights_{i}']})
                if len(biases)>0:
                    out_dict.update({f'biases_{i}': ckpt_results[f'biases_{i}']})

            json.dump(out_dict, fp, indent=4)


        # also save the current checkpoint as the 'best_checkpoint.json' for the tool
        with open(join(self.ckpts_dir, 'best_checkpoint.json'), 'w') as fp:
            if 'train' in self.best_ckpt_by:
                # throw a warning message that the best checkpoint is not decided on a test metric
                warnings.warn(f"The 'best_checkpoint' is decided based on the '{self.best_ckpt_by}' metric \
which is calculated on the training data. It is recommended to configure DeepRepVizBackend(best_ckpt_by=..) \
to a test or validation data metric instead.")
            json.dump({
                'ckpt_idx' : self.ckpt_idx,
                'ckpt_name': ckpt_name,
                'metrics'  : ckpt_results["metrics"]
            }, fp, indent=4)

            self.ckpt_idx += 1


    def apply_hook(self, model, hook_layer=-1):
        ''' Set a deeprepviz hook to read the activations from the
        specific layer of the model and return the hooked model'''
        layers = get_all_model_layers(model)
        # if layer name is given then iterate from the last and select the first layername that matches
        if isinstance(hook_layer, str):
            for layer in layers[::-1]:
                if hook_layer.lower() == layer[0].lower():
                    self.hooked_layer = layer
            if self.hooked_layer is None:
                raise ValueError(
                    f"[Error] Requested layer name '{hook_layer}' doesn't exist in the model.\
Existing layers in the model (name, Module): \n{layers}")
        # if layer index is given then get the layer by index
        else:
            self.hooked_layer  = layers[hook_layer]

        if self.verbose:
            print(f"[DeepRepViz] Applying DeepRepViz hook to \n\
    Model name     : {model.__class__.__name__}\n\
    Layer name     : {self.hooked_layer[0]}\n\
    Layer          : {self.hooked_layer[1]}\n\
    Layer n(params): {get_param_count(self.hooked_layer[1])}")

        #src references: https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
        self.hook = self.hooked_layer[1].register_forward_hook(self._hook_fn)
        return model


    # define the hook function
    def _hook_fn(self, layer, inp, out):
        if self.verbose>1: print(f'[DeepRepViz] Hook triggered with input.shape={inp[0].shape}')
        # get the input activations of all subjects
        acts = inp[0].detach().float().cpu()
        self.acts.extend(self._reshape_acts(acts))

        # get the weights and biases of the hooked layer and store them once
        if len(self.weights)==0:
            self.weights = layer.state_dict()['weight'].detach().float().cpu()
            if 'bias' in layer.state_dict():
                self.biases = layer.state_dict()['bias'].detach().float().cpu()

    def _reset_hooked_vals(self):
        # reset/initialize the lists that save model activations using the forward hook fn
        self.acts, self.weights, self.biases = [], [], []

    def _reshape_acts(self, act):
        """
        Reshape the input activations into a list shape.
        https://einops.rocks/1-einops-basics/
        """
        shape_dict = {
                      5 : 'b c h w d -> b (c h w d)',
                      4 : 'b c h w -> b (c h w)',
                      3 : 'b h w -> b (h w)',
                      2 : 'b h -> b (h)'
                     }
        if act.ndim > 5:
            raise ValueError(f"Representation tensor shape is expected to be lower than 6, but it is {act.ndim}.")
        act = rearrange(act, shape_dict[act.ndim])

        if act.shape[-1]<3:
            raise ValueError(f"The number of input features (activations) to the selected DL layer should be >=3 but it is {act.shape[-1]}")
        return act

def get_all_model_layers(module: torch.nn.Module, name=''):
    """
    Recursively get all children layers with trainable parameters
    from the architecture as a flat list

    PARAMETER
    --------
    module : torch.nn.module()

    RETURN
    ------
    layers : list
        list of all layers; element is a torch.nn.module
    """
    layers_list = []
    children = tuple(module.named_children())
    # if a layer has no more children module then this is
    # the child layer which we need.
    if (len(children)==0) and (
        # Exclude activation functions & BatchNorm layers
        len(list(module.parameters()))>0) and (
        not isinstance(module, nn.modules.batchnorm._BatchNorm)):
        # return layer name and the layer object as a tuple
        return [(name, module)]
    else:
       # recurse from child to children until we get to the last child
        for name, child in children:
            try:
                layers_list.extend(get_all_model_layers(child, name))
            except TypeError:
                layers_list.append(get_all_model_layers(child, name))
    return layers_list

def get_param_count(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def create_dataloader(dataloader_class, dataloader_kwargs,
                      expected_IDs, expected_labels=None):
        """Returns an instance of the dataloader such that it also provides the ID along with X,y when __getitem__() is called"""

        # perform some sanity checks on the provided dataloader
        dataloader_dummy = dataloader_class(**dataloader_kwargs)
        assert len(dataloader_dummy) == len(expected_IDs)
        labels_gen = np.array([dataloader_dummy[i][-1] for i in range(len(dataloader_dummy))])
        # cross check that the dataloader y matches with the conf_table
        if expected_labels is not None:
            assert (labels_gen==expected_labels).all(), "the labels in the conf_table and the ones generated by the dataloader do not match."

        class DatasetWithID(dataloader_class):
            def __init__(self, IDs, **dataloader_kwargs):
                super().__init__(**dataloader_kwargs)
                self.IDs = IDs

            def __getitem__(self, idx):
                X_y = super().__getitem__(idx)
                i = self.IDs[idx]
                return i,X_y

        dataset = DatasetWithID(IDs=expected_IDs,
                                **dataloader_kwargs)
        return dataset