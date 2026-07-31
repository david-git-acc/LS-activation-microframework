from __future__ import annotations
import torch
import pandas as pd

### CUSTOM
from dataclass_objects.input_objects import expInput, testInput
from support.config import config
from support.processing_helpers import dfs_settings2tensors, nan_long
from support.parsing_helpers import safe_asdict
from categories.base_definitions import categoryExperimentTracker, measure_type2dim


class metricsExperimentTracker(categoryExperimentTracker) :
    
    """metricsExperimentTracker implementation for test prediction data. Stores epochs and test samples.
    This implementation is currently identical to testpredsExperimentTracker except the _category = "metrics".
    However, it is kept separately to avoid unnecessary dependency and allow for future isolation of any changes.
    """
    
    def __init__(self, xpi : expInput) :
        super().__init__(xpi)
        self._category = "metrics"
        self._data = torch.full(size = (xpi.n_captures, len(xpi.Y_test_tensor)), fill_value = nan_long, dtype = torch.long) 
    
    def track(self) -> torch.Tensor :
        return torch.argmax(self.xpi.anet_model(self.xpi.X_test_tensor), dim = 1).to(torch.long).view(-1).detach().cpu()
    
    def record(self, record_index : int = 0) -> None :
        self._data[record_index, :] = self.track()  


def post_experiment_test_metrics(ti : testInput) -> pd.DataFrame :
    
    """Post-experiment test function for metrics. 
    
    Params:
        testInput dataclass, containing the dataframes, reference to the experiment configuration (may be None),
        and all necessary parameters. See the TestInput type annotation for more details.
    
    Returns:
        DataFrame: DataFrame containing all (metric, kf_reducer) combinations and the corresponding output data.
    
    """
    
    # We need an experiment reference so we can identify which data to compare to
    if ti.xpc is None :
        raise ValueError("Metrics category requires a parent expConfig reference (xpc) in testInput dataclass")
    elif "kfold_data" not in ti.metadata : 
        raise ValueError("Test input metadata should have a kfold_data attribute if you want to record metrics over data")
    
    # Ground truths are Y test, which will be the 4th and final element of the processing
    ground_truths = [Y_test.numpy() for X_train, X_test, Y_train, Y_test in ti.metadata["kfold_data"] ]
    
    nepochs = ti.X.shape[measure_type2dim("epochs")]
    nfolds = ti.X.shape[-1]
    
    result_dict = { metric_name : torch.full(size = (nepochs, nfolds), fill_value = torch.nan) 
                   for metric_name in config["eval_metric_names"] }
    
    for metric_name, metric in zip(config["eval_metric_names"], config["eval_metrics"]) :
        
        # The ground truths only change by fold, never by epochs or samples, so we iterate only over folds
        for k in range(nfolds) :
            
            # 1D array containing true sizes, without padding of the correct values for given testpreds
            y = ground_truths[k]
            testpreds_this_fold = ti.X[:, :, k] # Now of shape (n_epochs, n_testsamples)
            
            # Number of valid samples is len(y), since it always gives the exact number of groundtruths and therefore preds
            unpadded_testpreds = testpreds_this_fold[:, :len(y)].numpy()

            # Forced to iterate manually because metrics are not vectorisable
            result = torch.tensor([metric(y, y_hat) for y_hat in unpadded_testpreds])
            result_dict[metric_name][:, k] = result
          
        for kfold_aggfunc_name, kfold_aggfunc in zip(ti.kf_reducer_names, ti.kf_reducers ) :
            result = result_dict[metric_name]
            collapsed_result = kfold_aggfunc(result, dim = 1 ) # Always 2D so last dim is always = 1
            result_dict[(metric_name, kfold_aggfunc_name)] = collapsed_result.view(-1).numpy()
        
        # Once we have applied all agg funcs over all folds, no longer need the original uncollapsed kfold data
        del result_dict[metric_name]
            
    result_df = pd.DataFrame(result_dict)
    result_df.columns = pd.Index(result_df.columns) # Multiindex will break the code later
    
    if (diff := len(result_df.columns) - len(set(result_df.columns)) ) != 0 :    
        raise ValueError(f"{diff} duplicate df columns detected. Please check eval_metrics and kf_reducers")
    
    # Name the index based on if we measure epochs or otherwise
    result_df.index.name = "epoch" 
        
    return result_df    