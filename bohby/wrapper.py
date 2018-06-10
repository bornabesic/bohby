from hpbandster.optimizers.bohb import BOHB
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.result import json_result_logger
import ConfigSpace
import ConfigSpace.hyperparameters as CSH

_runid = "bohby"

class WrapWorker(Worker):

    def __init__(self, model_class, train_and_validate_fn, working_directory, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_class = model_class
        self.train_and_validate_fn = train_and_validate_fn

    def compute(self, config, budget, **kwargs):
        """
        The actual function that is minimized!

        This method is repeatedly called with different configurations and 
        budgets during the optimization. The return value has to be a dict
        with the mandatory fields 'loss' and 'info'. The former has to be a
        single number, the latter can be any build in python type
        (i.e. no numpy arrays! -> .tolist() below)
        """

        # Instantiate the new model
        model = self.model_class(**config)

        # Choose a portion of data depending on the budget [0, 1]
        # dataset_size = len(self.train_data)
        # indices = np.random.choice(dataset_size, size = int(budget * dataset_size), replace = False)
        # subset = Subset(self.train_data, indices)

        # Train
        validation_loss = self.train_and_validate_fn(model, budget)

        print("Current loss: {:4.3f}".format(validation_loss))

        return {
            "loss": validation_loss,
            "info": {} # no interesting information to store here
        }


def generate_configspace(parameters):
    config_space = ConfigSpace.ConfigurationSpace()

    types = {
        int: CSH.UniformIntegerHyperparameter,
        float: CSH.UniformFloatHyperparameter,
        list: CSH.CategoricalHyperparameter
    }

    for parameter_name in parameters:
        attrs = parameters[parameter_name]
        param_type = attrs["type"]
        del attrs["type"]

        config_space.add_hyperparameter(types[param_type](
            parameter_name,
            **attrs
        ))

    return config_space

def optimize_hyperparameters(model_class, parameters, train_and_validate_fn, num_iterations, min_budget = 0.01, working_dir = "./bohby/"):

    # Generate a configspace from the given parameters
    config_space = generate_configspace(parameters)

    # Start a local nameserver for communication
    NS = hpns.NameServer(run_id = _runid, nic_name = "lo", working_directory = working_dir)
    ns_host, ns_port = NS.start()

    # Define the worker
    worker = WrapWorker(model_class, train_and_validate_fn,  working_directory = working_dir,  nameserver = ns_host, nameserver_port = ns_port, run_id = _runid)
    worker.run(background = True)
 
    # Enable live logging so a run can be canceled at any time and we can still recover the results
    result_logger = json_result_logger(directory = working_dir, overwrite = True)

    # Optimization
    bohb = BOHB(configspace = config_space,
			working_directory = working_dir,
			run_id = _runid,
			eta = 2, min_budget = min_budget, max_budget = 1,
			host = ns_host,
			nameserver=ns_host, 
			nameserver_port = ns_port,
			ping_interval = 3600,
			result_logger = result_logger
	)

    res = bohb.run(n_iterations = num_iterations)

    # Clean up
    bohb.shutdown(shutdown_workers = True)
    NS.shutdown()

    # Best found config
    run_results = hpres.logged_results_to_HB_result(working_dir)
    id2conf = run_results.get_id2config_mapping()

    incumbent_id = run_results.get_incumbent_id()
    incumbent_config = id2conf[incumbent_id]['config']
    incumbent_runs = run_results.get_runs_by_id(incumbent_id)

    val_loss = incumbent_runs[-1].loss

    return val_loss, incumbent_config
 