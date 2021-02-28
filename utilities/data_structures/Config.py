from multiprocessing import Value


class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self, config = None):
        if(config != None):
            self.debug_mode = config.get_debug_mode()
            self.environment = config.get_environment()
            self.file_to_save_data_results = config.get_file_to_save_data_results()
            self.file_to_save_results_graph = config.get_file_to_save_results_graph()
            self.num_episodes_to_run = config.get_num_episodes_to_run()
            self.overwrite_existing_results_file = config.get_overwrite_existing_results_file()
            self.randomise_random_seed = config.get_randomise_random_seed()
            self.requirements_to_solve_game = config.get_requirements_to_solve_game()
            self.runs_per_agent = config.get_runs_per_agent()
            self.visualise_individual_results = config.get_visualise_individual_results()
            self.visualise_overall_results = config.get_visualise_overall_results()
            self.save_model = config.get_save_model()
            self.seed = config.get_seed()
            self.show_solution_score = config.get_show_solution_score()
            self.standard_deviation_results = config.get_standard_deviation_results()
            self.use_GPU = config.get_use_GPU()
        else:          
            self.debug_mode = False
            self.environment = None
            self.file_to_save_data_results = None
            self.file_to_save_results_graph = None
            self.hyperparameters = None
            self.num_episodes_to_run = None
            self.overwrite_existing_results_file = None
            self.randomise_random_seed = True
            self.requirements_to_solve_game = None
            self.runs_per_agent = None
            self.visualise_individual_results = False
            self.visualise_overall_results = False
            self.save_model = False
            self.seed = None
            self.show_solution_score = False
            self.standard_deviation_results = 1.0
            self.use_GPU = None
        
        
        
    def get_seed(self):
        if(self.seed == None):
            raise ValueError("Seed Not Defined")
        return self.seed

    def get_environment(self):
        if(self.environment == None):
            raise ValueError("Environment Not Defined")
        return self.environment

    def get_requirements_to_solve_game(self):
        return self.requirements_to_solve_game

    def get_num_episodes_to_run(self):
        if(self.num_episodes_to_run == None):
            raise ValueError("Num Episodes Not Defined")
        return self.num_episodes_to_run
    
    def get_file_to_save_data_results(self):
        if(self.file_to_save_data_results == None):
            raise ValueError("File to save data Not Defined")
        return self.file_to_save_data_results
    
    def get_file_to_save_results_graph(self):
        if(self.file_to_save_results_graph == None):
            raise ValueError("File to save results graph Not Defined")
        return self.file_to_save_results_graph  

    def get_runs_per_agent(self):
        if(self.runs_per_agent == None):
            raise ValueError("Runs Per Agent Not Defined")
        return self.runs_per_agent

    def get_visualise_overall_results(self):
        if(self.visualise_overall_results == None):
            raise ValueError("Visualise Overall results Not Defined")
        return self.visualise_overall_results
    
    def get_visualise_individual_results(self):
        if(self.visualise_individual_results == None):
            raise ValueError("Visualise Individual results Not Defined")
        return self.visualise_individual_results

    def get_hyperparameters(self):
        raise ValueError("Not supposed to access hyperparameters anymore")

    def get_use_GPU(self):
        if(self.use_GPU == None):
            raise ValueError("Use GPU is Not Defined")
        return self.use_GPU

    def get_overwrite_existing_results_file(self):
        if(self.overwrite_existing_results_file == None):
            raise ValueError("Overwrite Existing results file Not Defined")
        return self.overwrite_existing_results_file

    def get_save_model(self):
        if(self.save_model == None):
            raise ValueError("Save Model Not Defined")
        return self.save_model

    def get_standard_deviation_results(self):
        if(self.standard_deviation_results == None):
            raise ValueError("Standard Deviation results Not Defined")
        return self.standard_deviation_results

    def get_randomise_random_seed(self):
        if(self.randomise_random_seed == None):
            raise ValueError("Randomise Random Seed Not Defined")
        return self.randomise_random_seed

    def get_show_solution_score(self):
        if(self.show_solution_score == None):
            raise ValueError("Show Solution Score Not Defined")
        return self.show_solution_score

    def get_debug_mode(self):
        if(self.debug_mode == None):
            raise ValueError("Debug Mode Not Defined")
        return self.debug_mode
    

    
    

    
    





